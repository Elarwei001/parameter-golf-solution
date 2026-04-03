"""
Modal XSA Training - Exclusive Self Attention

XSA removes the projection of attention output onto self value vector,
forcing attention to focus on contextual information rather than self.

Paper: "Exclusive Self Attention" (2026)
Key insight: Standard attention has "similarity bias" - output is too 
similar to self value. XSA fixes this with 2 lines of code.

Usage:
    modal run modal_xsa.py::train_xsa --steps 5000
"""
import modal
import os
import math

app = modal.App("parameter-golf-xsa")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.0",
        "numpy",
    ])
)

data_volume = modal.Volume.from_name("parameter-golf-data", create_if_missing=True)


@app.function(
    image=image,
    gpu="H100",
    volumes={"/data": data_volume},
    timeout=1800,
)
def train_xsa(
    steps: int = 5000,
    dim: int = 512,
    n_layers: int = 9,
    n_heads: int = 8,
    n_kv_heads: int = 4,
    window_size: int = 192,
    lr: float = 1e-3,
    batch_size: int = 64,
    seq_len: int = 256,
    use_xsa: bool = True,  # Toggle XSA vs standard attention
):
    """Train GPT with XSA (Exclusive Self Attention)"""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import time
    
    DEVICE = torch.device("cuda")
    VOCAB_SIZE = 8192
    DATA_DIR = "/data/datasets/fineweb10B_sp8192"
    
    print(f"Device: {DEVICE}")
    print(f"CUDA: {torch.cuda.get_device_name()}")
    print(f"Config: dim={dim}, layers={n_layers}, heads={n_heads}")
    print(f"XSA: {'✅ ENABLED' if use_xsa else '❌ DISABLED (baseline)'}")
    
    # ── Load Data ─────────────────────────────────────────────
    print("\nLoading BPE-8192 tokenized data...")
    
    train_files = sorted([f for f in os.listdir(DATA_DIR) if 'train' in f])
    val_files = sorted([f for f in os.listdir(DATA_DIR) if 'val' in f])
    
    train_data = []
    for f in train_files[:10]:
        path = os.path.join(DATA_DIR, f)
        data = np.fromfile(path, dtype=np.uint16)
        train_data.append(data)
    train_data = np.concatenate(train_data)
    
    val_data = []
    for f in val_files:
        path = os.path.join(DATA_DIR, f)
        data = np.fromfile(path, dtype=np.uint16)
        val_data.append(data)
    val_data = np.concatenate(val_data)
    
    print(f"Train: {len(train_data)/1e6:.1f}M tokens | Val: {len(val_data)/1e6:.1f}M tokens")
    
    def get_batch(data, seq_len=seq_len, batch_size=batch_size):
        starts = np.random.randint(0, len(data) - seq_len - 1, batch_size)
        batch = np.stack([data[i:i+seq_len+1] for i in starts])
        return torch.from_numpy(batch.astype(np.int64)).to(DEVICE)
    
    # ── Model Components ──────────────────────────────────────
    class RMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))
        def forward(self, x):
            rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
            return x / rms * self.weight
    
    class RotaryEmbedding(nn.Module):
        def __init__(self, dim, max_seq_len=4096, base=10000.0):
            super().__init__()
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
            self.register_buffer("inv_freq", inv_freq)
            self._extend(max_seq_len)
        
        def _extend(self, seq_len):
            t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat([freqs, freqs], dim=-1)
            self.register_buffer("cos", emb.cos())
            self.register_buffer("sin", emb.sin())
            self._max_seq_len = seq_len
        
        def forward(self, x, seq_len):
            if seq_len > self._max_seq_len:
                self._extend(seq_len * 2)
            return self.cos[:seq_len], self.sin[:seq_len]
    
    def apply_rotary_pos_emb(q, k, cos, sin):
        def rotate_half(x):
            x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
            return torch.cat([-x2, x1], dim=-1)
        q_embed = q * cos + rotate_half(q) * sin
        k_embed = k * cos + rotate_half(k) * sin
        return q_embed, k_embed
    
    class MLP_ReLU2(nn.Module):
        def __init__(self, dim, mult=4):
            super().__init__()
            hidden = int(dim * mult)
            self.w1 = nn.Linear(dim, hidden, bias=False)
            self.w2 = nn.Linear(hidden, dim, bias=False)
        
        def forward(self, x):
            h = F.leaky_relu(self.w1(x), 0.01)
            return self.w2(h.square())
    
    class AttentionXSA(nn.Module):
        """Attention with optional XSA (Exclusive Self Attention)"""
        def __init__(self, dim, n_heads, n_kv_heads=None, window_size=128, use_xsa=True):
            super().__init__()
            self.n_heads = n_heads
            self.n_kv_heads = n_kv_heads or n_heads
            self.head_dim = dim // n_heads
            self.n_rep = n_heads // self.n_kv_heads
            self.window_size = window_size
            self.use_xsa = use_xsa
            
            self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
            self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
            self.wv = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
            self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)
        
        def forward(self, x, cos, sin):
            B, L, _ = x.shape
            
            q = self.wq(x).view(B, L, self.n_heads, self.head_dim).transpose(1,2)
            k = self.wk(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1,2)
            v = self.wv(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1,2)
            
            # Apply RoPE
            q, k = apply_rotary_pos_emb(q, k,
                                        cos.unsqueeze(0).unsqueeze(0),
                                        sin.unsqueeze(0).unsqueeze(0))
            
            # Repeat KV for GQA
            if self.n_rep > 1:
                k = k.repeat_interleave(self.n_rep, dim=1)
                v = v.repeat_interleave(self.n_rep, dim=1)
            
            # Standard attention
            scale = self.head_dim ** -0.5
            attn = (q @ k.transpose(-2,-1)) * scale
            
            # Causal + sliding window mask
            rows = torch.arange(L, device=x.device).unsqueeze(1)
            cols = torch.arange(L, device=x.device).unsqueeze(0)
            causal_mask = rows < cols
            window_mask = (rows - cols) > self.window_size
            attn = attn.masked_fill((causal_mask | window_mask).unsqueeze(0).unsqueeze(0), float('-inf'))
            
            attn = F.softmax(attn, dim=-1)
            y = attn @ v  # [B, H, L, D]
            
            # ═══════════════════════════════════════════════════════
            # XSA: Remove projection onto self value vector
            # This is the KEY innovation - just 2 lines!
            # ═══════════════════════════════════════════════════════
            if self.use_xsa:
                # Get self value (diagonal of v for each position)
                # v is [B, H, L, D], we want v_i for each position i
                v_self = v  # v[:, :, i, :] is the self value for position i
                
                # Normalize self value
                v_norm = F.normalize(v_self, dim=-1)  # [B, H, L, D]
                
                # Remove projection: z = y - (y·v_norm) * v_norm
                proj = (y * v_norm).sum(dim=-1, keepdim=True)  # [B, H, L, 1]
                z = y - proj * v_norm  # [B, H, L, D]
                
                out = z.transpose(1,2).reshape(B, L, -1)
            else:
                out = y.transpose(1,2).reshape(B, L, -1)
            
            return self.wo(out)
    
    class TransformerBlock(nn.Module):
        def __init__(self, dim, n_heads, n_kv_heads=None, window_size=128, use_xsa=True):
            super().__init__()
            self.attn = AttentionXSA(dim, n_heads, n_kv_heads, window_size, use_xsa)
            self.mlp = MLP_ReLU2(dim)
            self.norm1 = RMSNorm(dim)
            self.norm2 = RMSNorm(dim)
        
        def forward(self, x, cos, sin):
            x = x + self.attn(self.norm1(x), cos, sin)
            x = x + self.mlp(self.norm2(x))
            return x
    
    class GPT_XSA(nn.Module):
        def __init__(self, vocab_size, dim, n_layers, n_heads, n_kv_heads, max_seq_len, window_size, use_xsa):
            super().__init__()
            self.vocab_size = vocab_size
            self.tok_emb = nn.Embedding(vocab_size, dim)
            self.rope = RotaryEmbedding(dim // n_heads, max_seq_len)
            self.layers = nn.ModuleList([
                TransformerBlock(dim, n_heads, n_kv_heads, window_size, use_xsa)
                for _ in range(n_layers)
            ])
            self.norm = RMSNorm(dim)
            self.head = nn.Linear(dim, vocab_size, bias=False)
            self.tok_emb.weight = self.head.weight  # Weight tying
        
        def forward(self, idx):
            B, L = idx.shape
            x = self.tok_emb(idx)
            cos, sin = self.rope(x, L)
            for layer in self.layers:
                x = layer(x, cos, sin)
            x = self.norm(x)
            return self.head(x)
        
        def loss(self, batch):
            logits = self(batch[:, :-1])
            return F.cross_entropy(logits.reshape(-1, self.vocab_size),
                                   batch[:, 1:].reshape(-1))
    
    # ── BPB Calculation ───────────────────────────────────────
    def calculate_bpb(model, data, num_batches=50):
        model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for _ in range(num_batches):
                batch = get_batch(data)
                loss = model.loss(batch)
                total_loss += loss.item() * (batch.shape[0] * batch.shape[1])
                total_tokens += batch.shape[0] * batch.shape[1]
        
        avg_loss = total_loss / total_tokens
        BYTES_PER_TOKEN = 4.0
        bpb = (avg_loss / math.log(2)) * (1.0 / BYTES_PER_TOKEN)
        
        model.train()
        return bpb, avg_loss
    
    # ── Create Model ──────────────────────────────────────────
    model = GPT_XSA(
        vocab_size=VOCAB_SIZE,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        max_seq_len=seq_len + 64,
        window_size=window_size,
        use_xsa=use_xsa,
    ).to(DEVICE)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel params: {n_params/1e6:.2f}M")
    
    # ── Optimizer ─────────────────────────────────────────────
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    
    # ── Training Loop ─────────────────────────────────────────
    LOG_EVERY = 100
    mode_str = "XSA" if use_xsa else "Standard"
    print(f"\n{'='*60}")
    print(f"Training GPT with {mode_str} Attention")
    print(f"Steps: {steps} | Batch: {batch_size}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    best_bpb = float('inf')
    
    for step in range(1, steps + 1):
        batch = get_batch(train_data)
        loss = model.loss(batch)
        
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()
        
        if step % LOG_EVERY == 0:
            elapsed = time.time() - start_time
            val_bpb, val_loss = calculate_bpb(model, val_data, num_batches=30)
            best_bpb = min(best_bpb, val_bpb)
            print(f"Step {step:5d} [{mode_str}] | Loss {loss.item():.4f} | Val BPB {val_bpb:.4f} | "
                  f"Val Loss {val_loss:.4f} | LR {scheduler.get_last_lr()[0]:.2e} | "
                  f"Time {elapsed:.0f}s")
    
    # ── Final Evaluation ──────────────────────────────────────
    final_bpb, final_loss = calculate_bpb(model, val_data, num_batches=100)
    total_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"  Mode: {mode_str}")
    print(f"  Final Val Loss: {final_loss:.4f}")
    print(f"  Final Val BPB:  {final_bpb:.4f}")
    print(f"  Best Val BPB:   {best_bpb:.4f}")
    print(f"  Total Time:     {total_time:.0f}s")
    print(f"{'='*60}")
    
    # ── Save Checkpoint ───────────────────────────────────────
    checkpoint_dir = "/data/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_name = f"{'xsa' if use_xsa else 'baseline'}_dim{dim}_L{n_layers}_bpb{final_bpb:.3f}.pt"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'vocab_size': VOCAB_SIZE,
            'dim': dim,
            'n_layers': n_layers,
            'n_heads': n_heads,
            'n_kv_heads': n_kv_heads,
            'use_xsa': use_xsa,
        },
        'metrics': {
            'final_bpb': final_bpb,
            'best_bpb': best_bpb,
        },
    }, checkpoint_path)
    data_volume.commit()
    print(f"\n💾 Checkpoint saved to: {checkpoint_path}")
    
    return {
        "mode": mode_str,
        "final_bpb": final_bpb,
        "best_bpb": best_bpb,
        "params": n_params,
        "time_seconds": total_time,
    }


@app.local_entrypoint()
def main():
    print("Parameter Golf - XSA (Exclusive Self Attention)")
    print("=" * 50)
    print("Compare XSA vs Standard Attention:")
    print("  modal run modal_xsa.py::train_xsa --use-xsa true")
    print("  modal run modal_xsa.py::train_xsa --use-xsa false")

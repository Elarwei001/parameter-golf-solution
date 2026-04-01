"""
Modal XSA + LoRA TTT - Best of Both Worlds

Combines:
1. XSA (Exclusive Self Attention) - removes self-similarity bias
2. LoRA TTT (Test-Time Training) - adapts to each document

Target: Beat our previous best of 1.40 BPB!

Usage:
    modal run modal_xsa_ttt.py::train_and_eval --steps 5000
"""
import modal
import os
import math

app = modal.App("parameter-golf-xsa-ttt")

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
    timeout=3600,
)
def train_and_eval(
    steps: int = 5000,
    dim: int = 512,
    n_layers: int = 12,  # Use more layers
    n_heads: int = 8,
    n_kv_heads: int = 4,
    window_size: int = 192,
    lr: float = 1e-3,
    batch_size: int = 64,
    seq_len: int = 256,
    # TTT params
    lora_rank: int = 8,
    ttt_lr: float = 0.01,
    ttt_epochs: int = 2,
    chunk_size: int = 256,
):
    """Train XSA model, then evaluate with LoRA TTT"""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import time
    
    DEVICE = torch.device("cuda")
    VOCAB_SIZE = 8192
    DATA_DIR = "/data/datasets/fineweb10B_sp8192"
    
    print("="*70)
    print("XSA + LoRA TTT Experiment")
    print(f"Config: dim={dim}, layers={n_layers}, steps={steps}")
    print(f"TTT: LoRA rank={lora_rank}, epochs={ttt_epochs}")
    print("="*70)
    
    # ══════════════════════════════════════════════════════════════════
    # MODEL WITH XSA + LORA
    # ══════════════════════════════════════════════════════════════════
    
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
    
    class LoRALinear(nn.Module):
        """Linear with optional LoRA adaptation"""
        def __init__(self, in_features, out_features, bias=False):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.weight = nn.Parameter(torch.empty(out_features, in_features))
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
            self.lora_A = None
            self.lora_B = None
        
        def init_lora(self, rank, device=None):
            device = device or self.weight.device
            self.lora_A = nn.Parameter(torch.randn(rank, self.in_features, device=device) * 0.01)
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank, device=device))
            return [self.lora_A, self.lora_B]
        
        def reset_lora(self):
            if self.lora_A is not None:
                nn.init.normal_(self.lora_A, std=0.01)
                nn.init.zeros_(self.lora_B)
        
        def forward(self, x):
            out = F.linear(x, self.weight, self.bias)
            if self.lora_A is not None and self.lora_B is not None:
                out = out + F.linear(F.linear(x, self.lora_A), self.lora_B)
            return out
    
    class MLP_ReLU2(nn.Module):
        def __init__(self, dim, mult=4):
            super().__init__()
            hidden = int(dim * mult)
            self.w1 = nn.Linear(dim, hidden, bias=False)
            self.w2 = nn.Linear(hidden, dim, bias=False)
        
        def forward(self, x):
            h = F.leaky_relu(self.w1(x), 0.01)
            return self.w2(h.square())
    
    class AttentionXSA_LoRA(nn.Module):
        """XSA Attention with LoRA support"""
        def __init__(self, dim, n_heads, n_kv_heads=None, window_size=128):
            super().__init__()
            self.n_heads = n_heads
            self.n_kv_heads = n_kv_heads or n_heads
            self.head_dim = dim // n_heads
            self.n_rep = n_heads // self.n_kv_heads
            self.window_size = window_size
            
            # Q and V get LoRA
            self.wq = LoRALinear(dim, n_heads * self.head_dim)
            self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
            self.wv = LoRALinear(dim, self.n_kv_heads * self.head_dim)
            self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)
        
        def forward(self, x, cos, sin):
            B, L, _ = x.shape
            
            q = self.wq(x).view(B, L, self.n_heads, self.head_dim).transpose(1,2)
            k = self.wk(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1,2)
            v = self.wv(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1,2)
            
            q, k = apply_rotary_pos_emb(q, k,
                                        cos.unsqueeze(0).unsqueeze(0),
                                        sin.unsqueeze(0).unsqueeze(0))
            
            if self.n_rep > 1:
                k = k.repeat_interleave(self.n_rep, dim=1)
                v = v.repeat_interleave(self.n_rep, dim=1)
            
            scale = self.head_dim ** -0.5
            attn = (q @ k.transpose(-2,-1)) * scale
            
            # Causal + sliding window mask
            rows = torch.arange(L, device=x.device).unsqueeze(1)
            cols = torch.arange(L, device=x.device).unsqueeze(0)
            causal_mask = rows < cols
            window_mask = (rows - cols) > self.window_size
            attn = attn.masked_fill((causal_mask | window_mask).unsqueeze(0).unsqueeze(0), float('-inf'))
            
            attn = F.softmax(attn, dim=-1)
            y = attn @ v
            
            # XSA: Remove projection onto self value
            v_norm = F.normalize(v, dim=-1)
            proj = (y * v_norm).sum(dim=-1, keepdim=True)
            z = y - proj * v_norm
            
            out = z.transpose(1,2).reshape(B, L, -1)
            return self.wo(out)
        
        def init_lora(self, rank, device=None):
            params = []
            params.extend(self.wq.init_lora(rank, device))
            params.extend(self.wv.init_lora(rank, device))
            return params
        
        def reset_lora(self):
            self.wq.reset_lora()
            self.wv.reset_lora()
    
    class TransformerBlock(nn.Module):
        def __init__(self, dim, n_heads, n_kv_heads=None, window_size=128):
            super().__init__()
            self.attn = AttentionXSA_LoRA(dim, n_heads, n_kv_heads, window_size)
            self.mlp = MLP_ReLU2(dim)
            self.norm1 = RMSNorm(dim)
            self.norm2 = RMSNorm(dim)
        
        def forward(self, x, cos, sin):
            x = x + self.attn(self.norm1(x), cos, sin)
            x = x + self.mlp(self.norm2(x))
            return x
        
        def init_lora(self, rank, device=None):
            return self.attn.init_lora(rank, device)
        
        def reset_lora(self):
            self.attn.reset_lora()
    
    class GPT_XSA_LoRA(nn.Module):
        def __init__(self, vocab_size, dim, n_layers, n_heads, n_kv_heads, max_seq_len, window_size):
            super().__init__()
            self.vocab_size = vocab_size
            self.tok_emb = nn.Embedding(vocab_size, dim)
            self.rope = RotaryEmbedding(dim // n_heads, max_seq_len)
            self.layers = nn.ModuleList([
                TransformerBlock(dim, n_heads, n_kv_heads, window_size)
                for _ in range(n_layers)
            ])
            self.norm = RMSNorm(dim)
            self.head = LoRALinear(dim, vocab_size)
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
        
        def init_all_lora(self, rank, device=None):
            device = device or next(self.parameters()).device
            params = []
            for layer in self.layers:
                params.extend(layer.init_lora(rank, device))
            params.extend(self.head.init_lora(rank, device))
            return params
        
        def reset_all_lora(self):
            for layer in self.layers:
                layer.reset_lora()
            self.head.reset_lora()
        
        def freeze_base(self):
            for name, param in self.named_parameters():
                if 'lora' not in name:
                    param.requires_grad = False
        
        def unfreeze_all(self):
            for param in self.parameters():
                param.requires_grad = True
    
    # ══════════════════════════════════════════════════════════════════
    # DATA LOADING
    # ══════════════════════════════════════════════════════════════════
    
    print("\nLoading BPE-8192 data...")
    train_files = sorted([f for f in os.listdir(DATA_DIR) if 'train' in f])
    val_files = sorted([f for f in os.listdir(DATA_DIR) if 'val' in f])
    
    train_data = []
    for f in train_files[:10]:
        data = np.fromfile(os.path.join(DATA_DIR, f), dtype=np.uint16)
        train_data.append(data)
    train_data = np.concatenate(train_data)
    
    val_data = []
    for f in val_files:
        data = np.fromfile(os.path.join(DATA_DIR, f), dtype=np.uint16)
        val_data.append(data)
    val_data = np.concatenate(val_data)
    
    print(f"Train: {len(train_data)/1e6:.1f}M | Val: {len(val_data)/1e6:.1f}M tokens")
    
    def get_batch(data, seq_len=seq_len, batch_size=batch_size):
        starts = np.random.randint(0, len(data) - seq_len - 1, batch_size)
        batch = np.stack([data[i:i+seq_len+1] for i in starts])
        return torch.from_numpy(batch.astype(np.int64)).to(DEVICE)
    
    # ══════════════════════════════════════════════════════════════════
    # PHASE 1: TRAINING WITH XSA
    # ══════════════════════════════════════════════════════════════════
    
    print("\n" + "="*70)
    print("Phase 1: Training XSA Model")
    print("="*70)
    
    model = GPT_XSA_LoRA(
        vocab_size=VOCAB_SIZE,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        max_seq_len=seq_len + 64,
        window_size=window_size,
    ).to(DEVICE)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params/1e6:.2f}M")
    
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    
    LOG_EVERY = 500
    start_time = time.time()
    
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
            print(f"Step {step}/{steps} | Loss {loss.item():.4f} | "
                  f"LR {scheduler.get_last_lr()[0]:.2e} | Time {elapsed:.0f}s")
    
    train_time = time.time() - start_time
    print(f"\nTraining complete in {train_time:.0f}s")
    
    # ══════════════════════════════════════════════════════════════════
    # PHASE 2: STANDARD EVALUATION (no TTT)
    # ══════════════════════════════════════════════════════════════════
    
    def calculate_bpb(model, data, num_batches=100):
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
        return bpb, avg_loss
    
    print("\n" + "="*70)
    print("Phase 2: Standard Evaluation (no TTT)")
    print("="*70)
    
    pre_ttt_bpb, pre_ttt_loss = calculate_bpb(model, val_data)
    print(f"Pre-TTT BPB: {pre_ttt_bpb:.4f} | Loss: {pre_ttt_loss:.4f}")
    
    # ══════════════════════════════════════════════════════════════════
    # PHASE 3: LoRA TTT EVALUATION
    # ══════════════════════════════════════════════════════════════════
    
    print("\n" + "="*70)
    print(f"Phase 3: LoRA TTT Evaluation (rank={lora_rank})")
    print("="*70)
    
    # Initialize LoRA
    model.init_all_lora(lora_rank, DEVICE)
    model.freeze_base()
    
    def eval_document_with_ttt(model, doc_tokens, chunk_size=256, epochs=2, lr=0.01):
        model.reset_all_lora()
        
        doc_len = len(doc_tokens)
        if doc_len < 512:
            return None
        
        lora_params = [p for p in model.parameters() if p.requires_grad]
        ttt_opt = torch.optim.Adam(lora_params, lr=lr, betas=(0.9, 0.95))
        
        chunks = []
        for i in range(0, doc_len - chunk_size, chunk_size // 2):
            chunk = doc_tokens[i:i+chunk_size+1]
            if len(chunk) == chunk_size + 1:
                chunks.append(chunk)
        
        if len(chunks) < 2:
            return None
        
        total_loss = 0
        total_tokens = 0
        
        for epoch in range(epochs):
            for i, chunk in enumerate(chunks):
                chunk_tensor = torch.tensor(chunk, dtype=torch.long, device=DEVICE).unsqueeze(0)
                loss = model.loss(chunk_tensor)
                
                if epoch == epochs - 1:
                    total_loss += loss.item() * (len(chunk) - 1)
                    total_tokens += len(chunk) - 1
                
                if not (epoch == epochs - 1 and i == len(chunks) - 1):
                    ttt_opt.zero_grad()
                    loss.backward()
                    ttt_opt.step()
        
        return total_loss / total_tokens if total_tokens > 0 else None
    
    # Evaluate with TTT
    doc_length = 2048
    num_docs = min(100, len(val_data) // doc_length)
    
    ttt_losses = []
    start_ttt = time.time()
    model.train()
    
    for i in range(num_docs):
        doc_start = i * doc_length
        doc_tokens = val_data[doc_start:doc_start + doc_length].astype(np.int64)
        doc_tokens = np.clip(doc_tokens, 0, VOCAB_SIZE - 1).tolist()
        
        loss = eval_document_with_ttt(model, doc_tokens, chunk_size, ttt_epochs, ttt_lr)
        
        if loss is not None:
            ttt_losses.append(loss)
        
        if (i + 1) % 20 == 0:
            avg_loss = sum(ttt_losses) / len(ttt_losses)
            current_bpb = (avg_loss / math.log(2)) * (1.0 / 4.0)
            print(f"  Doc {i+1}/{num_docs} | Avg BPB {current_bpb:.4f}")
    
    ttt_time = time.time() - start_ttt
    
    if ttt_losses:
        avg_ttt_loss = sum(ttt_losses) / len(ttt_losses)
        post_ttt_bpb = (avg_ttt_loss / math.log(2)) * (1.0 / 4.0)
    else:
        post_ttt_bpb = pre_ttt_bpb
    
    # ══════════════════════════════════════════════════════════════════
    # RESULTS
    # ══════════════════════════════════════════════════════════════════
    
    improvement = (pre_ttt_bpb - post_ttt_bpb) / pre_ttt_bpb * 100
    
    print("\n" + "="*70)
    print("🏆 FINAL RESULTS")
    print("="*70)
    print(f"  Model:        XSA, {n_layers} layers, dim={dim}")
    print(f"  Training:     {steps} steps, {train_time:.0f}s")
    print(f"  Parameters:   {n_params/1e6:.2f}M")
    print(f"")
    print(f"  Pre-TTT BPB:  {pre_ttt_bpb:.4f}")
    print(f"  Post-TTT BPB: {post_ttt_bpb:.4f}")
    print(f"  Improvement:  {improvement:.2f}%")
    print(f"  TTT Time:     {ttt_time:.0f}s for {num_docs} docs")
    print("="*70)
    
    # Compare with previous best
    PREVIOUS_BEST = 1.396  # 12L experiment
    if post_ttt_bpb < PREVIOUS_BEST:
        print(f"\n🎉 NEW RECORD! Beat previous best of {PREVIOUS_BEST:.4f}")
    else:
        print(f"\n📊 Previous best: {PREVIOUS_BEST:.4f}")
    
    # Save checkpoint
    checkpoint_dir = "/data/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"xsa_ttt_bpb{post_ttt_bpb:.3f}.pt")
    
    # Reset LoRA before saving (only save base weights)
    model.reset_all_lora()
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'dim': dim, 'n_layers': n_layers, 'n_heads': n_heads,
            'n_kv_heads': n_kv_heads, 'vocab_size': VOCAB_SIZE,
        },
        'metrics': {
            'pre_ttt_bpb': pre_ttt_bpb,
            'post_ttt_bpb': post_ttt_bpb,
            'improvement': improvement,
        },
    }, checkpoint_path)
    data_volume.commit()
    print(f"\n💾 Saved to {checkpoint_path}")
    
    return {
        'pre_ttt_bpb': pre_ttt_bpb,
        'post_ttt_bpb': post_ttt_bpb,
        'improvement': improvement,
        'train_time': train_time,
        'ttt_time': ttt_time,
    }


@app.local_entrypoint()
def main():
    print("XSA + LoRA TTT - Going for a new record!")
    result = train_and_eval.remote(steps=5000, n_layers=12)
    print(f"\n🏁 Final: {result['post_ttt_bpb']:.4f} BPB")

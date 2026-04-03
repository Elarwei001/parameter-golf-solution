"""
Modal QAT Training v2 - Larger model to use 16MB budget

Config: dim=560, 10 layers, ~39M params, ~16MB quantized

Usage:
    modal run modal_qat_v2.py::train_qat_v2 --steps 5000
"""
import modal
import os
import math

app = modal.App("parameter-golf-qat-v2")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.0",
        "numpy",
        "tqdm",
    ])
)

data_volume = modal.Volume.from_name("parameter-golf-data", create_if_missing=True)


@app.function(
    image=image,
    gpu="H100",
    volumes={"/data": data_volume},
    timeout=1800,
)
def train_qat_v2(
    steps: int = 5000,
    dim: int = 560,        # Increased from 512
    n_layers: int = 10,    # Increased from 9
    n_heads: int = 8,
    n_kv_heads: int = 4,
    window_size: int = 192,
    lr: float = 1e-3,
    batch_size: int = 64,
    seq_len: int = 256,
    warmup_steps: int = 500,
):
    """Train larger GPT with Ternary QAT on H100"""
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
    print(f"QAT: Ternary (1.58-bit) weights")
    
    # ── Load Data ─────────────────────────────────────────────
    print("\nLoading BPE-8192 tokenized data...")
    
    train_files = sorted([f for f in os.listdir(DATA_DIR) if 'train' in f])
    val_files = sorted([f for f in os.listdir(DATA_DIR) if 'val' in f])
    
    train_data = []
    for f in train_files[:10]:
        path = os.path.join(DATA_DIR, f)
        data = np.fromfile(path, dtype=np.uint16)
        train_data.append(data)
        print(f"  Loaded {f}: {len(data)/1e6:.1f}M tokens")
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
    
    # ── Ternary Quantization with STE ─────────────────────────
    class TernaryQuantize(torch.autograd.Function):
        @staticmethod
        def forward(ctx, weight):
            scale = weight.abs().mean() + 1e-8
            w_quant = torch.clamp(torch.round(weight / scale), -1, 1)
            ctx.save_for_backward(weight)
            ctx.scale = scale
            return w_quant * scale
        
        @staticmethod
        def backward(ctx, grad_output):
            return grad_output
    
    def ternary_quantize(weight):
        return TernaryQuantize.apply(weight)
    
    # ── QAT Linear Layer ──────────────────────────────────────
    class QATLinear(nn.Module):
        def __init__(self, in_features, out_features, bias=False, qat_enabled=False):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.qat_enabled = qat_enabled
            self.weight = nn.Parameter(torch.empty(out_features, in_features))
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            if bias:
                self.bias = nn.Parameter(torch.zeros(out_features))
            else:
                self.register_parameter('bias', None)
        
        def forward(self, x):
            if self.qat_enabled:
                w = ternary_quantize(self.weight)
            else:
                w = self.weight
            return F.linear(x, w, self.bias)
        
        def enable_qat(self):
            self.qat_enabled = True
    
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
        def __init__(self, dim, mult=4, qat_enabled=False):
            super().__init__()
            hidden = int(dim * mult)
            self.w1 = QATLinear(dim, hidden, bias=False, qat_enabled=qat_enabled)
            self.w2 = QATLinear(hidden, dim, bias=False, qat_enabled=qat_enabled)
        
        def forward(self, x):
            h = F.relu(self.w1(x))
            return self.w2(h.square())
        
        def enable_qat(self):
            self.w1.enable_qat()
            self.w2.enable_qat()
    
    class AttentionSW(nn.Module):
        def __init__(self, dim, n_heads, n_kv_heads=None, window_size=128, qat_enabled=False):
            super().__init__()
            self.n_heads = n_heads
            self.n_kv_heads = n_kv_heads or n_heads
            self.head_dim = dim // n_heads
            self.n_rep = n_heads // self.n_kv_heads
            self.window_size = window_size
            
            self.wq = QATLinear(dim, n_heads * self.head_dim, bias=False, qat_enabled=qat_enabled)
            self.wk = QATLinear(dim, self.n_kv_heads * self.head_dim, bias=False, qat_enabled=qat_enabled)
            self.wv = QATLinear(dim, self.n_kv_heads * self.head_dim, bias=False, qat_enabled=qat_enabled)
            self.wo = QATLinear(n_heads * self.head_dim, dim, bias=False, qat_enabled=qat_enabled)
        
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
            
            rows = torch.arange(L, device=x.device).unsqueeze(1)
            cols = torch.arange(L, device=x.device).unsqueeze(0)
            causal_mask = rows < cols
            window_mask = (rows - cols) > self.window_size
            attn = attn.masked_fill((causal_mask | window_mask).unsqueeze(0).unsqueeze(0), float('-inf'))
            
            attn = F.softmax(attn, dim=-1)
            out = attn @ v
            out = out.transpose(1,2).reshape(B, L, -1)
            return self.wo(out)
        
        def enable_qat(self):
            self.wq.enable_qat()
            self.wk.enable_qat()
            self.wv.enable_qat()
            self.wo.enable_qat()
    
    class TransformerBlock(nn.Module):
        def __init__(self, dim, n_heads, n_kv_heads=None, window_size=128, qat_enabled=False):
            super().__init__()
            self.attn = AttentionSW(dim, n_heads, n_kv_heads, window_size, qat_enabled)
            self.mlp = MLP_ReLU2(dim, qat_enabled=qat_enabled)
            self.norm1 = RMSNorm(dim)
            self.norm2 = RMSNorm(dim)
        
        def forward(self, x, cos, sin):
            x = x + self.attn(self.norm1(x), cos, sin)
            x = x + self.mlp(self.norm2(x))
            return x
        
        def enable_qat(self):
            self.attn.enable_qat()
            self.mlp.enable_qat()
    
    class GPT_QAT(nn.Module):
        def __init__(self, vocab_size, dim, n_layers, n_heads, n_kv_heads, max_seq_len, window_size):
            super().__init__()
            self.vocab_size = vocab_size
            self.tok_emb = nn.Embedding(vocab_size, dim)
            self.rope = RotaryEmbedding(dim // n_heads, max_seq_len)
            self.layers = nn.ModuleList([
                TransformerBlock(dim, n_heads, n_kv_heads, window_size, qat_enabled=False)
                for _ in range(n_layers)
            ])
            self.norm = RMSNorm(dim)
            self.head = nn.Linear(dim, vocab_size, bias=False)
            self.tok_emb.weight = self.head.weight
        
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
        
        def enable_qat(self):
            for layer in self.layers:
                layer.enable_qat()
            print("✅ QAT enabled for all transformer layers")
        
        def count_ternary_params(self):
            ternary = 0
            fp = 0
            for name, param in self.named_parameters():
                if 'tok_emb' in name or 'head' in name or 'norm' in name:
                    fp += param.numel()
                else:
                    ternary += param.numel()
            return ternary, fp
    
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
    model = GPT_QAT(
        vocab_size=VOCAB_SIZE,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        max_seq_len=seq_len + 64,
        window_size=window_size,
    ).to(DEVICE)
    
    n_params = sum(p.numel() for p in model.parameters())
    ternary_params, fp_params = model.count_ternary_params()
    
    print(f"\nModel params: {n_params/1e6:.2f}M")
    print(f"  Ternary (1.58-bit): {ternary_params/1e6:.2f}M")
    print(f"  Full precision: {fp_params/1e6:.2f}M")
    
    ternary_size_mb = (ternary_params * 1.58) / 8 / 1e6
    fp_size_mb = (fp_params * 16) / 8 / 1e6
    total_size_mb = ternary_size_mb + fp_size_mb
    print(f"\nEstimated quantized size: {total_size_mb:.2f} MB")
    
    # ── Optimizer ─────────────────────────────────────────────
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    
    # ── Training Loop ─────────────────────────────────────────
    LOG_EVERY = 100
    print(f"\n{'='*60}")
    print(f"Training GPT v2 with QAT")
    print(f"Config: dim={dim}, layers={n_layers}, ~{total_size_mb:.1f}MB")
    print(f"Steps: {steps} | Warmup: {warmup_steps} | Batch: {batch_size}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    best_bpb = float('inf')
    qat_enabled = False
    
    for step in range(1, steps + 1):
        if step == warmup_steps + 1 and not qat_enabled:
            model.enable_qat()
            qat_enabled = True
            print(f"\n🔄 Step {step}: Switching to QAT mode\n")
        
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
            mode = "QAT" if qat_enabled else "FP16"
            print(f"Step {step:5d} [{mode}] | Loss {loss.item():.4f} | Val BPB {val_bpb:.4f} | "
                  f"Val Loss {val_loss:.4f} | LR {scheduler.get_last_lr()[0]:.2e} | "
                  f"Time {elapsed:.0f}s")
    
    # ── Final Evaluation ──────────────────────────────────────
    final_bpb, final_loss = calculate_bpb(model, val_data, num_batches=100)
    total_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"  Config: dim={dim}, layers={n_layers}")
    print(f"  Final Val Loss: {final_loss:.4f}")
    print(f"  Final Val BPB:  {final_bpb:.4f}")
    print(f"  Best Val BPB:   {best_bpb:.4f}")
    print(f"  Total Time:     {total_time:.0f}s")
    print(f"  Estimated Size: {total_size_mb:.2f} MB")
    print(f"{'='*60}")
    
    # ── Save Checkpoint ───────────────────────────────────────
    checkpoint_dir = "/data/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_name = f"qat_v2_dim{dim}_L{n_layers}_bpb{final_bpb:.3f}.pt"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': {
            'vocab_size': VOCAB_SIZE,
            'dim': dim,
            'n_layers': n_layers,
            'n_heads': n_heads,
            'n_kv_heads': n_kv_heads,
            'window_size': window_size,
        },
        'metrics': {
            'final_bpb': final_bpb,
            'best_bpb': best_bpb,
        },
    }
    
    import torch
    torch.save(checkpoint, checkpoint_path)
    data_volume.commit()
    print(f"\n💾 Checkpoint saved to: {checkpoint_path}")
    
    return {
        "config": f"dim={dim}, layers={n_layers}",
        "final_bpb": final_bpb,
        "best_bpb": best_bpb,
        "params": n_params,
        "estimated_size_mb": total_size_mb,
        "time_seconds": total_time,
        "checkpoint_path": checkpoint_path,
    }


@app.local_entrypoint()
def main():
    print("Parameter Golf - QAT v2 (Larger Model)")
    print("=" * 50)
    print("Configs to try:")
    print("  modal run modal_qat_v2.py::train_qat_v2 --dim 560 --n-layers 10")
    print("  modal run modal_qat_v2.py::train_qat_v2 --dim 576 --n-layers 9")
    print("  modal run modal_qat_v2.py::train_qat_v2 --dim 512 --n-layers 12")

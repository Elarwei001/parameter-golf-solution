"""
Experiment: dim=128 + Whitening vs dim=512 baseline

Hypothesis: If PR collapses to ~100, we can use dim=128 with whitening
and achieve similar performance while saving 75% embedding params.

Usage:
    modal run modal_whitening_exp.py::run_experiment
"""
import modal
import os

app = modal.App("parameter-golf-whitening")

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
def run_experiment(steps: int = 3000):
    """Compare dim=128 vs dim=512, with and without whitening"""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import math
    import time
    
    DEVICE = torch.device("cuda")
    VOCAB_SIZE = 8192
    SEQ_LEN = 256
    BATCH_SIZE = 64
    DATA_DIR = "/data/datasets/fineweb10B_sp8192"
    
    print("="*70)
    print("Experiment: Embedding Dimension vs Whitening")
    print("="*70)
    
    # ── Load Data ─────────────────────────────────────────────
    train_files = sorted([f for f in os.listdir(DATA_DIR) if 'train' in f])
    val_files = sorted([f for f in os.listdir(DATA_DIR) if 'val' in f])
    
    train_data = []
    for f in train_files[:5]:
        data = np.fromfile(os.path.join(DATA_DIR, f), dtype=np.uint16)
        train_data.append(data)
    train_data = np.concatenate(train_data)
    
    val_data = []
    for f in val_files:
        data = np.fromfile(os.path.join(DATA_DIR, f), dtype=np.uint16)
        val_data.append(data)
    val_data = np.concatenate(val_data)
    
    print(f"Train: {len(train_data)/1e6:.1f}M | Val: {len(val_data)/1e6:.1f}M tokens")
    
    def get_batch(data):
        starts = np.random.randint(0, len(data) - SEQ_LEN - 1, BATCH_SIZE)
        batch = np.stack([data[i:i+SEQ_LEN+1] for i in starts])
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
    
    class Attention(nn.Module):
        def __init__(self, dim, n_heads, window_size=192):
            super().__init__()
            self.n_heads = n_heads
            self.head_dim = dim // n_heads
            self.window_size = window_size
            self.wqkv = nn.Linear(dim, 3 * dim, bias=False)
            self.wo = nn.Linear(dim, dim, bias=False)
        
        def forward(self, x):
            B, L, _ = x.shape
            qkv = self.wqkv(x).chunk(3, dim=-1)
            q, k, v = [t.view(B, L, self.n_heads, self.head_dim).transpose(1,2) for t in qkv]
            
            scale = self.head_dim ** -0.5
            attn = (q @ k.transpose(-2,-1)) * scale
            
            # Causal + sliding window mask
            rows = torch.arange(L, device=x.device).unsqueeze(1)
            cols = torch.arange(L, device=x.device).unsqueeze(0)
            mask = (rows < cols) | ((rows - cols) > self.window_size)
            attn = attn.masked_fill(mask, float('-inf'))
            
            attn = F.softmax(attn, dim=-1)
            out = (attn @ v).transpose(1,2).reshape(B, L, -1)
            return self.wo(out)
    
    class MLP(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.w1 = nn.Linear(dim, 4 * dim, bias=False)
            self.w2 = nn.Linear(4 * dim, dim, bias=False)
        def forward(self, x):
            h = F.leaky_relu(self.w1(x), 0.01)
            return self.w2(h * h)  # LeakyReLU²
    
    class Block(nn.Module):
        def __init__(self, dim, n_heads):
            super().__init__()
            self.attn = Attention(dim, n_heads)
            self.mlp = MLP(dim)
            self.norm1 = RMSNorm(dim)
            self.norm2 = RMSNorm(dim)
        def forward(self, x):
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
            return x
    
    class GPT(nn.Module):
        def __init__(self, vocab_size, dim, n_layers, n_heads, use_whitening=False):
            super().__init__()
            self.vocab_size = vocab_size
            self.dim = dim
            self.use_whitening = use_whitening
            
            self.tok_emb = nn.Embedding(vocab_size, dim)
            self.layers = nn.ModuleList([Block(dim, n_heads) for _ in range(n_layers)])
            self.norm = RMSNorm(dim)
            self.head = nn.Linear(dim, vocab_size, bias=False)
            
            # Whitening layer (learned)
            if use_whitening:
                self.whiten = nn.Linear(dim, dim, bias=True)
                # Initialize close to identity
                nn.init.eye_(self.whiten.weight)
                nn.init.zeros_(self.whiten.bias)
        
        def forward(self, idx):
            x = self.tok_emb(idx)
            
            if self.use_whitening:
                x = self.whiten(x)
            
            for layer in self.layers:
                x = layer(x)
            
            return self.head(self.norm(x))
        
        def loss(self, batch):
            logits = self(batch[:, :-1])
            return F.cross_entropy(logits.reshape(-1, self.vocab_size),
                                   batch[:, 1:].reshape(-1))
    
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
        return bpb
    
    def count_params(model):
        return sum(p.numel() for p in model.parameters())
    
    def estimate_size_mb(model, embed_bits=16, layer_bits=1.58):
        """Estimate size with quantization"""
        embed_params = model.tok_emb.weight.numel()
        other_params = count_params(model) - embed_params
        
        embed_size = embed_params * embed_bits / 8 / 1e6
        other_size = other_params * layer_bits / 8 / 1e6
        return embed_size + other_size
    
    def train_model(dim, n_layers, n_heads, use_whitening, name):
        print(f"\n{'='*60}")
        print(f"Training: {name}")
        print(f"  dim={dim}, layers={n_layers}, heads={n_heads}, whitening={use_whitening}")
        print(f"{'='*60}")
        
        model = GPT(VOCAB_SIZE, dim, n_layers, n_heads, use_whitening).to(DEVICE)
        
        n_params = count_params(model)
        size_mb = estimate_size_mb(model)
        print(f"  Params: {n_params/1e6:.1f}M | Est. size: {size_mb:.1f} MB")
        
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
        
        start = time.time()
        best_bpb = float('inf')
        
        for step in range(1, steps + 1):
            batch = get_batch(train_data)
            loss = model.loss(batch)
            
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()
            
            if step % 500 == 0:
                val_bpb = calculate_bpb(model, val_data, num_batches=30)
                best_bpb = min(best_bpb, val_bpb)
                elapsed = time.time() - start
                print(f"  Step {step}/{steps} | Loss {loss.item():.4f} | Val BPB {val_bpb:.4f} | Time {elapsed:.0f}s")
        
        final_bpb = calculate_bpb(model, val_data, num_batches=50)
        total_time = time.time() - start
        
        print(f"\n  ✅ {name} Complete!")
        print(f"     Final BPB: {final_bpb:.4f} | Best BPB: {best_bpb:.4f}")
        print(f"     Time: {total_time:.0f}s | Size: {size_mb:.1f} MB")
        
        return {
            'name': name,
            'dim': dim,
            'n_layers': n_layers,
            'use_whitening': use_whitening,
            'params': n_params,
            'size_mb': size_mb,
            'final_bpb': final_bpb,
            'best_bpb': best_bpb,
            'time': total_time,
        }
    
    # ── Run Experiments ───────────────────────────────────────
    results = []
    
    # Experiment A: dim=512, 9 layers (baseline)
    results.append(train_model(
        dim=512, n_layers=9, n_heads=8, 
        use_whitening=False, name="A: dim=512 (baseline)"
    ))
    
    # Experiment B: dim=128, more layers (same param budget)
    # dim=128 / 4 heads = 32 head_dim ✓
    results.append(train_model(
        dim=128, n_layers=14, n_heads=4,  # More layers with smaller dim
        use_whitening=False, name="B: dim=128, 14 layers"
    ))
    
    # Experiment C: dim=128 + whitening layer
    results.append(train_model(
        dim=128, n_layers=14, n_heads=4,
        use_whitening=True, name="C: dim=128 + whitening"
    ))
    
    # Experiment D: dim=256 (middle ground)
    # dim=256 / 8 heads = 32 head_dim ✓
    results.append(train_model(
        dim=256, n_layers=12, n_heads=8,
        use_whitening=False, name="D: dim=256, 12 layers"
    ))
    
    # ── Summary ───────────────────────────────────────────────
    print("\n" + "="*70)
    print("📊 EXPERIMENT SUMMARY")
    print("="*70)
    print(f"\n{'Name':<30} {'Dim':>5} {'Layers':>6} {'Size':>8} {'BPB':>8}")
    print("-"*70)
    
    for r in results:
        print(f"{r['name']:<30} {r['dim']:>5} {r['n_layers']:>6} {r['size_mb']:>7.1f}M {r['final_bpb']:>8.4f}")
    
    best = min(results, key=lambda x: x['final_bpb'])
    print(f"\n🏆 Best: {best['name']} with BPB = {best['final_bpb']:.4f}")
    
    # Analysis
    baseline = results[0]
    print(f"\n📈 Compared to baseline (dim=512):")
    for r in results[1:]:
        diff = (r['final_bpb'] - baseline['final_bpb']) / baseline['final_bpb'] * 100
        size_diff = (r['size_mb'] - baseline['size_mb']) / baseline['size_mb'] * 100
        print(f"  {r['name']}: BPB {diff:+.1f}%, Size {size_diff:+.1f}%")
    
    return results


@app.local_entrypoint()
def main():
    print("Whitening Experiment")
    print("Comparing dim=128/256/512 with and without whitening")
    results = run_experiment.remote(steps=3000)
    print("\nExperiment complete!")

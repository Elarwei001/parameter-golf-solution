#!/usr/bin/env python3
"""
控制变量实验：每次只改一个参数
基线配置：dim=512, n_layers=9, n_heads=8, ws=192, lr=1e-3
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ── Device ──
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── Data ──
DATA_FILE = "/tmp/tinyshakespeare.txt"
if not os.path.exists(DATA_FILE):
    import urllib.request
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        DATA_FILE
    )

with open(DATA_FILE, "rb") as f:
    raw = f.read()
data = np.frombuffer(raw, dtype=np.uint8).copy()
split = int(len(data) * 0.9)
train_data, val_data = data[:split], data[split:]
print(f"Train: {len(train_data)/1e6:.2f}M | Val: {len(val_data)/1000:.0f}K")

# ── Baseline config ──
BASELINE = {
    "dim": 512,
    "n_layers": 9,
    "n_heads": 8,
    "n_kv_heads": 4,
    "mlp_mult": 4,
    "window_size": 192,  # 之前发现 192 > 128
    "lr": 1e-3,
    "batch_size": 32,
    "seq_len": 256,
    "train_minutes": 5,  # 每个实验 5 分钟
}

# ── Model components ──
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class MLP_LeakyReLU2(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        hidden = int(dim * mult)
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
    def forward(self, x):
        return self.w2(F.leaky_relu(self.w1(x), 0.5).square())

class Attention(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, window_size):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.n_rep = n_heads // n_kv_heads
        self.window_size = window_size
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)
    
    def forward(self, x):
        B, L, _ = x.shape
        q = self.wq(x).view(B, L, self.n_heads, self.head_dim).transpose(1,2)
        k = self.wk(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1,2)
        v = self.wv(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1,2)
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2,-1)) * scale
        # Sliding window + causal mask
        rows = torch.arange(L, device=x.device).view(-1, 1)
        cols = torch.arange(L, device=x.device).view(1, -1)
        causal_mask = cols > rows
        window_mask = (rows - cols) > self.window_size
        mask = causal_mask | window_mask
        attn = attn.masked_fill(mask, float('-inf'))
        attn = torch.nan_to_num(F.softmax(attn, dim=-1), nan=0.0)
        out = (attn @ v).transpose(1,2).reshape(B, L, -1)
        return self.wo(out)

class Block(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, mlp_mult, window_size):
        super().__init__()
        self.ln1 = RMSNorm(dim)
        self.attn = Attention(dim, n_heads, n_kv_heads, window_size)
        self.ln2 = RMSNorm(dim)
        self.mlp = MLP_LeakyReLU2(dim, mlp_mult)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size=256, dim=512, n_layers=9, n_heads=8, n_kv_heads=4, mlp_mult=4, window_size=192):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            Block(dim, n_heads, n_kv_heads, mlp_mult, window_size) 
            for _ in range(n_layers)
        ])
        self.ln_f = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
    def forward(self, idx):
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))

def get_batch(data, seq_len, batch_size):
    starts = np.random.randint(0, len(data) - seq_len - 1, batch_size)
    batch = np.stack([data[i:i+seq_len+1] for i in starts])
    return torch.from_numpy(batch.astype(np.int64)).to(DEVICE)

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def run_experiment(name, config):
    print(f"\n{'='*50}")
    print(f"Experiment: {name}")
    print(f"Config: ws={config['window_size']}, lr={config['lr']}")
    print(f"{'='*50}")
    
    model = GPT(
        dim=config['dim'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        n_kv_heads=config['n_kv_heads'],
        mlp_mult=config['mlp_mult'],
        window_size=config['window_size'],
    ).to(DEVICE)
    
    print(f"Parameters: {count_params(model):,}")
    
    opt = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    
    start = time.time()
    step = 0
    train_minutes = config['train_minutes']
    
    while time.time() - start < train_minutes * 60:
        batch = get_batch(train_data, config['seq_len'], config['batch_size'])
        logits = model(batch[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, 256), batch[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        step += 1
        if step % 100 == 0:
            elapsed = time.time() - start
            print(f"  Step {step} | Loss {loss.item():.4f} | {elapsed/60:.1f}min")
    
    # Eval
    model.eval()
    val_losses = []
    with torch.no_grad():
        for _ in range(50):
            batch = get_batch(val_data, config['seq_len'], config['batch_size'])
            logits = model(batch[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, 256), batch[:, 1:].reshape(-1))
            val_losses.append(loss.item())
    
    val_bpb = sum(val_losses) / len(val_losses) / 0.6931
    print(f"\n  ✓ Val BPB: {val_bpb:.4f}")
    return val_bpb

# ── Run experiments ──
results = []

# 1. Baseline (ws=192, lr=1e-3)
config = BASELINE.copy()
bpb = run_experiment("BASELINE (ws=192, lr=1e-3)", config)
results.append(("baseline", config['window_size'], config['lr'], bpb))

# 2. Window Size 变体 (只改 ws)
for ws in [128, 256]:
    config = BASELINE.copy()
    config['window_size'] = ws
    bpb = run_experiment(f"ws={ws} (lr=1e-3)", config)
    results.append((f"ws={ws}", ws, config['lr'], bpb))

# 3. Learning Rate 变体 (只改 lr)
for lr in [5e-4, 2e-3]:
    config = BASELINE.copy()
    config['lr'] = lr
    bpb = run_experiment(f"lr={lr} (ws=192)", config)
    results.append((f"lr={lr}", config['window_size'], lr, bpb))

# ── Summary ──
print("\n" + "="*60)
print("CONTROLLED EXPERIMENT RESULTS")
print("="*60)
print(f"{'Name':<20} {'WS':<6} {'LR':<10} {'BPB':<10}")
print("-"*60)
for name, ws, lr, bpb in results:
    marker = "🏆" if bpb == min(r[3] for r in results) else ""
    print(f"{name:<20} {ws:<6} {lr:<10.0e} {bpb:<10.4f} {marker}")

# Save results
with open("controlled_sweep_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to controlled_sweep_results.json")

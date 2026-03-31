#!/usr/bin/env python3
"""Quick sweep for window size and learning rate"""
import subprocess
import sys
import re

def run_experiment(ws, lr, minutes=2):
    """Run a single experiment and return val BPB"""
    cmd = f"""python3 -c "
import sys, os, time
sys.path.insert(0, '{os.getcwd()}')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Device
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

# Data
with open('/tmp/tinyshakespeare.txt', 'rb') as f:
    raw = f.read()
data = np.frombuffer(raw, dtype=np.uint8).copy()
split = int(len(data) * 0.9)
train_data, val_data = data[:split], data[split:]

def get_batch(data, seq_len=256, batch_size=32):
    starts = np.random.randint(0, len(data) - seq_len - 1, batch_size)
    batch = np.stack([data[i:i+seq_len+1] for i in starts])
    return torch.from_numpy(batch.astype(np.int64)).to(DEVICE)

# Simple model
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class MLP(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        hidden = int(dim * mult)
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
    def forward(self, x):
        return self.w2(F.leaky_relu(self.w1(x), 0.5).square())

class Attention(nn.Module):
    def __init__(self, dim, n_heads, window_size):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.window_size = window_size
        self.wqkv = nn.Linear(dim, 3 * dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
    def forward(self, x):
        B, L, _ = x.shape
        qkv = self.wqkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2,-1)) * scale
        # Sliding window + causal mask
        mask = torch.ones(L, L, device=x.device, dtype=torch.bool).triu(1)
        if self.window_size < L:
            mask = mask | torch.ones(L, L, device=x.device, dtype=torch.bool).tril(-self.window_size)
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1,2).reshape(B, L, -1)
        return self.wo(out)

class Block(nn.Module):
    def __init__(self, dim, n_heads, window_size):
        super().__init__()
        self.ln1 = RMSNorm(dim)
        self.attn = Attention(dim, n_heads, window_size)
        self.ln2 = RMSNorm(dim)
        self.mlp = MLP(dim)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size=256, dim=384, n_layers=6, n_heads=6, window_size=192):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([Block(dim, n_heads, window_size) for _ in range(n_layers)])
        self.ln_f = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
    def forward(self, idx):
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))

# Train
model = GPT(window_size={ws}).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr={lr})

start = time.time()
step = 0
while time.time() - start < {minutes * 60}:
    batch = get_batch(train_data)
    logits = model(batch[:, :-1])
    loss = F.cross_entropy(logits.reshape(-1, 256), batch[:, 1:].reshape(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()
    step += 1

# Eval
model.eval()
val_losses = []
with torch.no_grad():
    for _ in range(20):
        batch = get_batch(val_data)
        logits = model(batch[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, 256), batch[:, 1:].reshape(-1))
        val_losses.append(loss.item())
val_bpb = sum(val_losses) / len(val_losses) / 0.6931
print(f'VAL_BPB={val_bpb:.4f}')
"
"""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
    output = result.stdout + result.stderr
    match = re.search(r'VAL_BPB=(\d+\.\d+)', output)
    if match:
        return float(match.group(1))
    return None

if __name__ == "__main__":
    print("=== Window Size Sweep (lr=1e-3) ===")
    results = []
    for ws in [192, 256, 320]:
        bpb = run_experiment(ws, 1e-3, minutes=2)
        print(f"ws={ws}: {bpb:.4f}" if bpb else f"ws={ws}: FAILED")
        if bpb:
            results.append((ws, bpb))
    
    best_ws = min(results, key=lambda x: x[1])[0] if results else 192
    
    print(f"\n=== LR Sweep (ws={best_ws}) ===")
    lr_results = []
    for lr in [5e-4, 1e-3, 2e-3]:
        bpb = run_experiment(best_ws, lr, minutes=2)
        print(f"lr={lr}: {bpb:.4f}" if bpb else f"lr={lr}: FAILED")
        if bpb:
            lr_results.append((lr, bpb))
    
    if lr_results:
        best_lr, best_bpb = min(lr_results, key=lambda x: x[1])
        print(f"\n🏆 Best: ws={best_ws}, lr={best_lr}, BPB={best_bpb:.4f}")

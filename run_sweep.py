#!/usr/bin/env python3
"""
Parameter Golf - Hyperparameter sweep on the best combo
Tests: window_size (64, 128, 256) × negative_slope (0.1, 0.3, 0.5)
Each run: 10 minutes
Goal: push below 2.2821 BPB
"""
import sys, os, time, math, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ── Device ──────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Device: {DEVICE}")

# ── Data ─────────────────────────────────────────────────────────
DATA_FILE  = "/tmp/tinyshakespeare.txt"
VOCAB_SIZE = 256
SEQ_LEN    = 256
BATCH_SIZE = 32

def load_data():
    with open(DATA_FILE, "rb") as f:
        raw = f.read()
    data = np.frombuffer(raw, dtype=np.uint8).copy()
    split = int(len(data) * 0.9)
    return data[:split], data[split:]

def get_batch(data):
    starts = np.random.randint(0, len(data) - SEQ_LEN - 1, BATCH_SIZE)
    batch  = np.stack([data[i:i+SEQ_LEN+1] for i in starts])
    return torch.from_numpy(batch.astype(np.int64)).to(DEVICE)

train_data, val_data = load_data()
print(f"Train: {len(train_data)/1e6:.1f}M | Val: {len(val_data)/1000:.0f}K tokens")

# ── Model components ─────────────────────────────────────────────
from models.standard_gpt import RMSNorm, apply_rotary_pos_emb
from run_experiments import RotaryEmbeddingDynamic


class MLP_LeakyReLU2(nn.Module):
    """leaky_relu(x, slope).square() — slope is a hyperparam."""
    def __init__(self, dim: int, mult: int = 4, negative_slope: float = 0.5):
        super().__init__()
        hidden = int(dim * mult)
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.negative_slope = negative_slope

    def forward(self, x):
        h = F.leaky_relu(self.w1(x), negative_slope=self.negative_slope)
        return self.w2(h.square())


class AttentionSW(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads=None, window_size=128):
        super().__init__()
        self.n_heads     = n_heads
        self.n_kv_heads  = n_kv_heads or n_heads
        self.head_dim    = dim // n_heads
        self.n_rep       = n_heads // self.n_kv_heads
        self.window_size = window_size
        self.wq = nn.Linear(dim, n_heads        * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim,          bias=False)

    def forward(self, x, cos, sin):
        B, L, _ = x.shape
        q = self.wq(x).view(B, L, self.n_heads,    self.head_dim).transpose(1,2)
        k = self.wk(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1,2)
        v = self.wv(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1,2)
        q, k = apply_rotary_pos_emb(q, k,
                                    cos.unsqueeze(0).unsqueeze(0),
                                    sin.unsqueeze(0).unsqueeze(0))
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        scale = self.head_dim ** -0.5
        attn  = (q @ k.transpose(-2,-1)) * scale

        rows = torch.arange(L, device=x.device).unsqueeze(1)
        cols = torch.arange(L, device=x.device).unsqueeze(0)
        mask = (rows < cols) | ((rows - cols) > self.window_size)
        attn = torch.nan_to_num(F.softmax(attn.masked_fill(mask, float('-inf')), dim=-1), nan=0.0)
        return self.wo((attn @ v).transpose(1,2).reshape(B, L, -1))


class BlockCombo(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads=None, mlp_mult=4,
                 window_size=128, negative_slope=0.5):
        super().__init__()
        self.ln1  = RMSNorm(dim)
        self.attn = AttentionSW(dim, n_heads, n_kv_heads, window_size)
        self.ln2  = RMSNorm(dim)
        self.mlp  = MLP_LeakyReLU2(dim, mlp_mult, negative_slope)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT_Combo(nn.Module):
    def __init__(self, vocab_size, dim, n_layers, n_heads, n_kv_heads,
                 mlp_mult, max_seq_len, window_size=128, negative_slope=0.5):
        super().__init__()
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.rope    = RotaryEmbeddingDynamic(dim // n_heads, max_seq_len)
        self.blocks  = nn.ModuleList([
            BlockCombo(dim, n_heads, n_kv_heads, mlp_mult, window_size, negative_slope)
            for _ in range(n_layers)
        ])
        self.ln_f = RMSNorm(dim)
        for m in self.modules():
            if isinstance(m, nn.Linear):   nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding): nn.init.normal_(m.weight, std=0.02)

    def forward(self, x):
        B, L = x.shape
        h = self.tok_emb(x)
        cos, sin = self.rope(h, L)
        for blk in self.blocks:
            h = blk(h, cos, sin)
        return self.ln_f(h) @ self.tok_emb.weight.T

    def compute_loss(self, batch):
        logits = self.forward(batch[:, :-1])
        loss   = F.cross_entropy(logits.reshape(-1, self.vocab_size), batch[:, 1:].reshape(-1))
        return {'loss': loss, 'ce_loss': loss}

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Training ─────────────────────────────────────────────────────
def train(model, name, duration_sec=600):
    model = model.to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9,0.95), weight_decay=0.01)
    n_p   = model.count_parameters()

    print(f"\n{'─'*55}")
    print(f"  {name}  ({n_p/1e6:.2f}M params, {duration_sec//60}min)")
    print(f"{'─'*55}")

    model.train()
    losses, step, start = [], 0, time.time()
    while time.time() - start < duration_sec:
        batch = get_batch(train_data)
        opt.zero_grad()
        d = model.compute_loss(batch)
        d['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(d['ce_loss'].item())
        step += 1
        if step % 200 == 0:
            bpb = np.mean(losses[-50:]) / math.log(2)
            print(f"  [{(time.time()-start)/60:.1f}min] step {step}: BPB {bpb:.4f}")

    model.eval()
    vl = []
    with torch.no_grad():
        for _ in range(20):
            vl.append(model.compute_loss(get_batch(val_data))['ce_loss'].item())

    val_bpb   = np.mean(vl) / math.log(2)
    train_bpb = np.mean(losses[-100:]) / math.log(2)
    elapsed   = time.time() - start
    print(f"  ✅ {step} steps, {elapsed/60:.1f}min | train BPB {train_bpb:.4f} | val BPB {val_bpb:.4f}")
    return {'name': name, 'steps': step, 'train_bpb': train_bpb,
            'val_bpb': val_bpb, 'params': n_p}


# ── Sweep grid ───────────────────────────────────────────────────
# Focus: window_size (most impact) × negative_slope (secondary)
# 3 × 2 = 6 runs × 10min = 60min total — too long
# Better: 2 key axes, 4 runs × 10min = 40min
SWEEP = [
    # (window_size, negative_slope, label)
    ( 64, 0.5, "ws064_ns0.5"),   # smaller window
    (128, 0.3, "ws128_ns0.3"),   # tune slope (known best window)
    (192, 0.5, "ws192_ns0.5"),   # bigger window
    (128, 0.1, "ws128_ns0.1"),   # more linear slope
]
DURATION = 10 * 60  # 10 min each

BASE_KWARGS = dict(
    vocab_size=VOCAB_SIZE, dim=256, n_layers=6,
    n_heads=8, n_kv_heads=4, mlp_mult=4,
    max_seq_len=SEQ_LEN + 64,
)

# ── Main ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    sweep_results = []

    for ws, ns, label in SWEEP:
        model = GPT_Combo(**BASE_KWARGS, window_size=ws, negative_slope=ns)
        r = train(model, label, DURATION)
        r['window_size']     = ws
        r['negative_slope']  = ns
        sweep_results.append(r)
        del model
        torch.mps.empty_cache() if hasattr(torch, 'mps') else None

    # Load all results
    prev_path = "/tmp/parameter-golf-solution/experiment_results.json"
    all_results = []
    if os.path.exists(prev_path):
        with open(prev_path) as f:
            all_results = json.load(f)

    all_results.extend(sweep_results)
    with open(prev_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary
    best_prev = 2.2821  # from last combo run

    print("\n" + "="*65)
    print("SWEEP RESULTS  (baseline combo: 2.2821)")
    print("="*65)
    print(f"  {'Name':<22} {'ws':>5} {'ns':>5}  {'Val BPB':>10}  {'vs combo':>10}")
    print("  " + "-"*55)
    for r in sorted(sweep_results, key=lambda x: x['val_bpb']):
        delta  = r['val_bpb'] - best_prev
        marker = " 🏆" if r['val_bpb'] < best_prev else ""
        print(f"  {r['name']:<22} {r['window_size']:>5} {r['negative_slope']:>5.1f}  "
              f"{r['val_bpb']:>10.4f}  {delta:>+9.4f}{marker}")
    print("="*65)

    best = min(sweep_results, key=lambda x: x['val_bpb'])
    print(f"\n🏆 Best sweep config: window_size={best['window_size']}, "
          f"negative_slope={best['negative_slope']}")
    print(f"   Val BPB: {best['val_bpb']:.4f}")
    if best['val_bpb'] < best_prev:
        print(f"   🎉 New record! Improved by {best_prev - best['val_bpb']:.4f}")
    else:
        print(f"   → Original ws=128, ns=0.5 still best")
    print(f"\nAll results saved to {prev_path}")

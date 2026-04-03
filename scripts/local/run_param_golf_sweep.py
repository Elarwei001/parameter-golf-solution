#!/usr/bin/env python3
"""
Parameter Golf Low-Hanging Fruit Sweep
Experiments: LR, Dropout, Window Size
Baseline: LeakyReLU² + Sliding Window (ws=192, ns=0.5) → 2.1752 BPB
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
DURATION   = 5 * 60  # 5 minutes per run

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

# ── Model Components ─────────────────────────────────────────────
from models.standard_gpt import RMSNorm, apply_rotary_pos_emb
from run_experiments import RotaryEmbeddingDynamic


class MLP_LeakyReLU2(nn.Module):
    def __init__(self, dim: int, mult: int = 4, negative_slope: float = 0.5, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * mult)
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.negative_slope = negative_slope
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = F.leaky_relu(self.w1(x), negative_slope=self.negative_slope)
        h = self.dropout(h.square())
        return self.w2(h)


class AttentionSW(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads=None, window_size=192, dropout: float = 0.0):
        super().__init__()
        self.n_heads     = n_heads
        self.n_kv_heads  = n_kv_heads or n_heads
        self.head_dim    = dim // n_heads
        self.n_rep       = n_heads // self.n_kv_heads
        self.window_size = window_size
        self.wq = nn.Linear(dim, n_heads         * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim,          bias=False)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x, cos, sin):
        B, L, _ = x.shape
        q = self.wq(x).view(B, L, self.n_heads,    self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k,
                                    cos.unsqueeze(0).unsqueeze(0),
                                    sin.unsqueeze(0).unsqueeze(0))
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        scale = self.head_dim ** -0.5
        attn  = (q @ k.transpose(-2, -1)) * scale

        rows = torch.arange(L, device=x.device).unsqueeze(1)
        cols = torch.arange(L, device=x.device).unsqueeze(0)
        mask = (rows < cols) | ((rows - cols) > self.window_size)
        attn = torch.nan_to_num(
            F.softmax(attn.masked_fill(mask, float('-inf')), dim=-1), nan=0.0
        )
        attn = self.attn_drop(attn)
        return self.wo((attn @ v).transpose(1, 2).reshape(B, L, -1))


class BlockCombo(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads=None, mlp_mult=4,
                 window_size=192, negative_slope=0.5, dropout=0.0):
        super().__init__()
        self.ln1  = RMSNorm(dim)
        self.attn = AttentionSW(dim, n_heads, n_kv_heads, window_size, dropout)
        self.ln2  = RMSNorm(dim)
        self.mlp  = MLP_LeakyReLU2(dim, mlp_mult, negative_slope, dropout)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT_Combo(nn.Module):
    def __init__(self, vocab_size, dim, n_layers, n_heads, n_kv_heads,
                 mlp_mult, max_seq_len, window_size=192, negative_slope=0.5, dropout=0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.rope    = RotaryEmbeddingDynamic(dim // n_heads, max_seq_len)
        self.blocks  = nn.ModuleList([
            BlockCombo(dim, n_heads, n_kv_heads, mlp_mult, window_size, negative_slope, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = RMSNorm(dim)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

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


BASE_KWARGS = dict(
    vocab_size=VOCAB_SIZE, dim=256, n_layers=6,
    n_heads=8, n_kv_heads=4, mlp_mult=4,
    max_seq_len=SEQ_LEN + 64,
    window_size=192, negative_slope=0.5,
)

# ── Training helper ───────────────────────────────────────────────
def train(model, name, lr=1e-3, duration_sec=DURATION):
    model = model.to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    n_p   = model.count_parameters()

    # LR warmup (200 steps)
    warmup = 200
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda step: min(1.0, (step + 1) / warmup)
    )

    print(f"\n{'─'*60}")
    print(f"  {name}  ({n_p/1e6:.2f}M params, lr={lr}, {duration_sec//60}min)")
    print(f"{'─'*60}")

    model.train()
    losses, step, start = [], 0, time.time()
    while time.time() - start < duration_sec:
        batch = get_batch(train_data)
        opt.zero_grad()
        d = model.compute_loss(batch)
        d['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()
        losses.append(d['ce_loss'].item())
        step += 1
        if step % 200 == 0:
            bpb = np.mean(losses[-50:]) / math.log(2)
            elapsed = (time.time() - start) / 60
            print(f"  [{elapsed:.1f}min] step {step}: BPB {bpb:.4f}")

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


def clear_device():
    if DEVICE.type == 'mps' and hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()
    elif DEVICE.type == 'cuda':
        torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT 1: Learning Rate Sweep
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("EXPERIMENT 1: Learning Rate Sweep")
print("="*65)

lr_values = [5e-4, 8e-4, 1e-3, 1.5e-3, 2e-3]
lr_results = []

for lr in lr_values:
    model = GPT_Combo(**BASE_KWARGS)
    r = train(model, f"lr={lr:.0e}", lr=lr)
    r['lr'] = lr
    lr_results.append(r)
    del model
    clear_device()

best_lr_result = min(lr_results, key=lambda x: x['val_bpb'])
best_lr = best_lr_result['lr']

print("\n📊 LR Sweep Results:")
print(f"  {'LR':<10} {'Val BPB':>10}")
for r in sorted(lr_results, key=lambda x: x['lr']):
    marker = " ← best" if r['lr'] == best_lr else ""
    print(f"  {r['lr']:<10.0e} {r['val_bpb']:>10.4f}{marker}")


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT 2: Dropout Test
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("EXPERIMENT 2: Dropout Test (using best LR)")
print("="*65)

dropout_values = [0.0, 0.05, 0.1]
dropout_results = []

for do in dropout_values:
    kwargs = dict(**BASE_KWARGS)
    kwargs['dropout'] = do  # not in BASE_KWARGS yet, handled below
    model = GPT_Combo(**BASE_KWARGS, dropout=do)
    r = train(model, f"dropout={do}", lr=best_lr)
    r['dropout'] = do
    dropout_results.append(r)
    del model
    clear_device()

best_dropout_result = min(dropout_results, key=lambda x: x['val_bpb'])
best_dropout = best_dropout_result['dropout']

print("\n📊 Dropout Results:")
print(f"  {'Dropout':<10} {'Val BPB':>10}")
for r in sorted(dropout_results, key=lambda x: x['dropout']):
    marker = " ← best" if r['dropout'] == best_dropout else ""
    print(f"  {r['dropout']:<10.2f} {r['val_bpb']:>10.4f}{marker}")


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT 3: Window Size Fine-tuning
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("EXPERIMENT 3: Window Size Sweep (using best LR + dropout)")
print("="*65)

window_sizes = [64, 96, 128, 160, 192]
ws_results = []

for ws in window_sizes:
    model = GPT_Combo(**{**BASE_KWARGS, 'window_size': ws}, dropout=best_dropout)
    r = train(model, f"ws={ws}", lr=best_lr)
    r['window_size'] = ws
    ws_results.append(r)
    del model
    clear_device()

best_ws_result = min(ws_results, key=lambda x: x['val_bpb'])
best_ws = best_ws_result['window_size']

print("\n📊 Window Size Results:")
print(f"  {'Window':>8} {'Val BPB':>10}")
for r in sorted(ws_results, key=lambda x: x['window_size']):
    marker = " ← best" if r['window_size'] == best_ws else ""
    print(f"  {r['window_size']:>8} {r['val_bpb']:>10.4f}{marker}")


# ═══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════
BASELINE_BPB = 2.1752  # ws192_ns0.5 from previous sweep

all_new = lr_results + dropout_results + ws_results
best_overall = min(all_new, key=lambda x: x['val_bpb'])

print("\n" + "="*65)
print("FINAL SUMMARY")
print("="*65)
print(f"  Baseline (ws=192, ns=0.5):  {BASELINE_BPB:.4f} BPB")
print(f"  Best LR:      {best_lr:.0e}  → {best_lr_result['val_bpb']:.4f} BPB")
print(f"  Best Dropout: {best_dropout:.2f}    → {best_dropout_result['val_bpb']:.4f} BPB")
print(f"  Best WS:      {best_ws}       → {best_ws_result['val_bpb']:.4f} BPB")
print(f"\n  🏆 Overall best config:")
print(f"     Name: {best_overall['name']}")
print(f"     Val BPB: {best_overall['val_bpb']:.4f}")

if best_overall['val_bpb'] < BASELINE_BPB:
    improvement = BASELINE_BPB - best_overall['val_bpb']
    print(f"     🎉 Improved by {improvement:.4f} BPB!")
else:
    print(f"     → Baseline still best")

# Save results
results_path = "/tmp/parameter-golf-solution/param_golf_sweep_results.json"
output = {
    "baseline_bpb": BASELINE_BPB,
    "best_lr": best_lr,
    "best_lr_bpb": best_lr_result['val_bpb'],
    "best_dropout": best_dropout,
    "best_dropout_bpb": best_dropout_result['val_bpb'],
    "best_window_size": best_ws,
    "best_ws_bpb": best_ws_result['val_bpb'],
    "best_overall_name": best_overall['name'],
    "best_overall_bpb": best_overall['val_bpb'],
    "lr_results": lr_results,
    "dropout_results": dropout_results,
    "ws_results": ws_results,
}
with open(results_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\n  Results saved to {results_path}")
print("="*65)

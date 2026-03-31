#!/usr/bin/env python3
"""
Parameter Golf - Continue Sweep
Baseline: ws=192, ns=0.5 → 2.1752 BPB

Exp 1: Window Size Sweep [192, 224, 256, 320] @ lr=1e-3
Exp 2: LR Sweep [5e-4, 8e-4, 1e-3, 1.5e-3, 2e-3] @ ws=best
"""
import sys, os, time, math, json, argparse
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
DURATION   = 3 * 60  # 3 minutes per run

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
    def __init__(self, dim, n_heads, n_kv_heads=None, window_size=192):
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
        return self.wo((attn @ v).transpose(1, 2).reshape(B, L, -1))


class BlockCombo(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads=None, mlp_mult=4,
                 window_size=192, negative_slope=0.5):
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
                 mlp_mult, max_seq_len, window_size=192, negative_slope=0.5):
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
    negative_slope=0.5,
)

# ── Training helper ───────────────────────────────────────────────
def train(model, name, lr=1e-3, duration_sec=DURATION):
    model = model.to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    n_p   = model.count_parameters()

    warmup = 200
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda step: min(1.0, (step + 1) / warmup)
    )

    print(f"\n  ▶ {name}  ({n_p/1e6:.2f}M params, lr={lr:.0e}, {duration_sec//60}min)")

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
            print(f"    [{elapsed:.1f}min] step {step}: BPB {bpb:.4f}")

    model.eval()
    vl = []
    with torch.no_grad():
        for _ in range(20):
            vl.append(model.compute_loss(get_batch(val_data))['ce_loss'].item())

    val_bpb   = np.mean(vl) / math.log(2)
    train_bpb = np.mean(losses[-100:]) / math.log(2)
    elapsed   = time.time() - start
    print(f"    ✅ {step} steps, {elapsed/60:.1f}min | train {train_bpb:.4f} | val {val_bpb:.4f}")
    return {'name': name, 'steps': step, 'train_bpb': train_bpb,
            'val_bpb': val_bpb, 'params': n_p}


def clear_device():
    if DEVICE.type == 'mps' and hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()
    elif DEVICE.type == 'cuda':
        torch.cuda.empty_cache()


BASELINE_BPB = 2.1752  # ws=192, ns=0.5, lr=1e-3

# ═══════════════════════════════════════════════════════════════
# EXPERIMENT 1: Window Size Sweep @ lr=1e-3
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("EXPERIMENT 1: Window Size Sweep (lr=1e-3)")
print("="*65)

window_sizes = [192, 224, 256, 320]
ws_results = []

for ws in window_sizes:
    model = GPT_Combo(**BASE_KWARGS, window_size=ws)
    r = train(model, f"ws={ws}", lr=1e-3)
    r['window_size'] = ws
    r['lr'] = 1e-3
    ws_results.append(r)
    del model
    clear_device()

best_ws_result = min(ws_results, key=lambda x: x['val_bpb'])
best_ws = best_ws_result['window_size']

print("\n=== Window Size Sweep (lr=1e-3) ===")
print(f"ws=192: {BASELINE_BPB:.4f} (reference baseline from prev run)")
for r in ws_results:
    marker = " ← best" if r['window_size'] == best_ws else ""
    note   = " (baseline)" if r['window_size'] == 192 else ""
    print(f"ws={r['window_size']}: {r['val_bpb']:.4f}{note}{marker}")


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT 2: LR Sweep @ ws=best
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*65)
print(f"EXPERIMENT 2: LR Sweep (ws={best_ws})")
print("="*65)

lr_values = [5e-4, 8e-4, 1e-3, 1.5e-3, 2e-3]
lr_results = []

for lr in lr_values:
    model = GPT_Combo(**BASE_KWARGS, window_size=best_ws)
    r = train(model, f"ws={best_ws}_lr={lr:.0e}", lr=lr)
    r['lr'] = lr
    r['window_size'] = best_ws
    lr_results.append(r)
    del model
    clear_device()

best_lr_result = min(lr_results, key=lambda x: x['val_bpb'])
best_lr = best_lr_result['lr']

print("\n=== LR Sweep (ws=best) ===")
for r in sorted(lr_results, key=lambda x: x['lr']):
    marker = " ← best" if r['lr'] == best_lr else ""
    print(f"lr={r['lr']:.0e}: {r['val_bpb']:.4f}{marker}")


# ═══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════
all_results = ws_results + lr_results
best_overall = min(all_results, key=lambda x: x['val_bpb'])

print("\n" + "="*65)
print("FINAL SUMMARY")
print("="*65)
print(f"  Baseline (ws=192, lr=1e-3): {BASELINE_BPB:.4f} BPB")
print(f"  Best WS: {best_ws}  → {best_ws_result['val_bpb']:.4f} BPB")
print(f"  Best LR: {best_lr:.0e}  → {best_lr_result['val_bpb']:.4f} BPB")

best_config_bpb = best_overall['val_bpb']
improvement = BASELINE_BPB - best_config_bpb
print(f"\n  Best config: ws={best_overall.get('window_size', best_ws)}, lr={best_overall.get('lr', best_lr):.0e}")
print(f"  Best BPB: {best_config_bpb:.4f}")

if improvement > 0:
    print(f"  🎉 Improved by {improvement:.4f} BPB vs baseline!")
else:
    print(f"  → Baseline still best")

# Save
out_path = "/tmp/parameter-golf-solution/continue_sweep_results.json"
output = {
    "baseline_bpb": BASELINE_BPB,
    "ws_sweep": {str(r['window_size']): r['val_bpb'] for r in ws_results},
    "lr_sweep": {f"{r['lr']:.0e}": r['val_bpb'] for r in lr_results},
    "best_ws": best_ws,
    "best_lr": best_lr,
    "best_ws_bpb": best_ws_result['val_bpb'],
    "best_lr_bpb": best_lr_result['val_bpb'],
    "best_overall_bpb": best_config_bpb,
    "best_overall_name": best_overall['name'],
}
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\n  Results saved to {out_path}")
print("="*65)

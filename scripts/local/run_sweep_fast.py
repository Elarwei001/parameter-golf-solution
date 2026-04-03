#!/usr/bin/env python3
"""
Parameter Golf - Fast Hyperparameter Sweep
LR sweep (2 min each) then Window Size sweep (2 min each)
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
DATA_FILE = "/tmp/tinyshakespeare.txt"
VOCAB_SIZE = 256
SEQ_LEN    = 256
BATCH_SIZE = 32

def load_data():
    with open(DATA_FILE, "rb") as f:
        raw = f.read()
    data = np.frombuffer(raw, dtype=np.uint8).copy()
    split = int(len(data) * 0.9)
    return data[:split], data[split:]

def get_batch(data, seq_len=SEQ_LEN, batch_size=BATCH_SIZE):
    starts = np.random.randint(0, len(data) - seq_len - 1, batch_size)
    batch  = np.stack([data[i:i+seq_len+1] for i in starts])
    return torch.from_numpy(batch.astype(np.int64)).to(DEVICE)

train_data, val_data = load_data()
print(f"Train: {len(train_data)/1e6:.1f}M tokens | Val: {len(val_data)/1000:.0f}K tokens")

# ── Model Components ─────────────────────────────────────────────
from models.standard_gpt import RMSNorm, apply_rotary_pos_emb
from run_experiments import RotaryEmbeddingDynamic, MLP_LeakyReLU2

class AttentionSW(nn.Module):
    """Sliding-window causal attention."""
    def __init__(self, dim, n_heads, n_kv_heads=None, window_size=128):
        super().__init__()
        self.n_heads    = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.head_dim   = dim // n_heads
        self.n_rep      = n_heads // self.n_kv_heads
        self.window_size = window_size
        self.wq = nn.Linear(dim, n_heads       * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim,         bias=False)

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
        causal_mask = rows < cols
        window_mask = (rows - cols) > self.window_size
        attn = attn.masked_fill(causal_mask | window_mask, float('-inf'))
        attn = torch.nan_to_num(F.softmax(attn, dim=-1), nan=0.0)

        out = (attn @ v).transpose(1,2).reshape(B, L, -1)
        return self.wo(out)


class BlockCombo(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads=None, mlp_mult=4, window_size=128):
        super().__init__()
        self.ln1  = RMSNorm(dim)
        self.attn = AttentionSW(dim, n_heads, n_kv_heads, window_size)
        self.ln2  = RMSNorm(dim)
        self.mlp  = MLP_LeakyReLU2(dim, mlp_mult)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT_Combo(nn.Module):
    def __init__(self, vocab_size, dim, n_layers, n_heads, n_kv_heads,
                 mlp_mult, max_seq_len, window_size=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.rope    = RotaryEmbeddingDynamic(dim // n_heads, max_seq_len)
        self.blocks  = nn.ModuleList([
            BlockCombo(dim, n_heads, n_kv_heads, mlp_mult, window_size)
            for _ in range(n_layers)
        ])
        self.ln_f = RMSNorm(dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x):
        B, L = x.shape
        h = self.tok_emb(x)
        cos, sin = self.rope(h, L)
        for block in self.blocks:
            h = block(h, cos, sin)
        h = self.ln_f(h)
        return h @ self.tok_emb.weight.T

    def compute_loss(self, batch):
        logits  = self.forward(batch[:, :-1])
        targets = batch[:, 1:]
        loss    = F.cross_entropy(logits.reshape(-1, self.vocab_size),
                                  targets.reshape(-1))
        return {'loss': loss, 'ce_loss': loss}

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Training ─────────────────────────────────────────────────────
def train_config(lr, window_size, duration_sec=120, name=None):
    if name is None:
        name = f"lr={lr:.0e}_ws={window_size}"
    
    model = GPT_Combo(
        vocab_size=VOCAB_SIZE, dim=256, n_layers=6,
        n_heads=8, n_kv_heads=4, mlp_mult=4,
        max_seq_len=SEQ_LEN + 64, window_size=window_size
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                             betas=(0.9, 0.95), weight_decay=0.01)

    model.train()
    losses, step, start = [], 0, time.time()

    while time.time() - start < duration_sec:
        batch = get_batch(train_data)
        opt.zero_grad()
        loss_dict = model.compute_loss(batch)
        loss_dict['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss_dict['ce_loss'].item())
        step += 1

    # Val eval
    model.eval()
    val_losses = []
    with torch.no_grad():
        for _ in range(20):
            batch = get_batch(val_data)
            d     = model.compute_loss(batch)
            val_losses.append(d['ce_loss'].item())

    val_bpb   = np.mean(val_losses) / math.log(2)
    elapsed   = time.time() - start

    print(f"  {name}: steps={step}, elapsed={elapsed:.0f}s, val_BPB={val_bpb:.4f}")
    del model
    return val_bpb


# ── Main Sweep ─────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast hyperparameter sweep")
    parser.add_argument("--duration", type=int, default=120, help="Seconds per run")
    parser.add_argument("--default-ws", type=int, default=128, help="Default window size for LR sweep")
    parser.add_argument("--default-lr", type=float, default=None, help="Override LR for WS sweep (skips LR sweep)")
    args = parser.parse_args()

    DURATION = args.duration
    DEFAULT_WS = args.default_ws

    results = {}

    # ── Experiment 1: LR Sweep ────────────────────────────────
    if args.default_lr is None:
        print("\n" + "="*50)
        print("=== LR Sweep ===")
        print("="*50)
        lr_values = [5e-4, 8e-4, 1e-3, 1.5e-3, 2e-3]
        lr_results = {}
        for lr in lr_values:
            bpb = train_config(lr=lr, window_size=DEFAULT_WS, duration_sec=DURATION,
                               name=f"lr={lr:.0e}_ws={DEFAULT_WS}")
            lr_results[lr] = bpb

        print(f"\n=== LR Sweep ===")
        for lr, bpb in sorted(lr_results.items()):
            print(f"lr={lr:.0e}:  BPB={bpb:.4f}")
        best_lr = min(lr_results, key=lr_results.get)
        print(f"Best LR: {best_lr:.0e}  (BPB={lr_results[best_lr]:.4f})")
        results['lr_sweep'] = lr_results
        results['best_lr'] = best_lr
    else:
        best_lr = args.default_lr
        print(f"\nUsing provided LR: {best_lr:.0e}")

    # ── Experiment 2: Window Size Sweep ───────────────────────
    print("\n" + "="*50)
    print("=== Window Size Sweep ===")
    print("="*50)
    ws_values = [64, 96, 128, 160]
    ws_results = {}
    for ws in ws_values:
        bpb = train_config(lr=best_lr, window_size=ws, duration_sec=DURATION,
                           name=f"lr={best_lr:.0e}_ws={ws}")
        ws_results[ws] = bpb

    print(f"\n=== Window Sweep ===")
    for ws, bpb in sorted(ws_results.items()):
        print(f"ws={ws}:  BPB={bpb:.4f}")
    best_ws = min(ws_results, key=ws_results.get)
    print(f"Best WS: {best_ws}  (BPB={ws_results[best_ws]:.4f})")
    results['ws_sweep'] = ws_results
    results['best_ws'] = best_ws

    # ── Final Summary ─────────────────────────────────────────
    best_bpb = ws_results[best_ws]
    print(f"\n=== Final ===")
    print(f"Best config: lr={best_lr:.0e}, ws={best_ws}")
    print(f"Best BPB: {best_bpb:.4f}")

    # Compare to previous best
    prev_best = 2.1752  # ws192_ns0.5
    if best_bpb < prev_best:
        improvement = prev_best - best_bpb
        print(f"\n🎉 Improvement over previous best ({prev_best:.4f}): -{improvement:.4f}")
    else:
        print(f"\n📊 Previous best: {prev_best:.4f} (ws192_ns0.5)")
        print(f"   No improvement in 2-min runs (longer training may differ)")

    # Save sweep results
    sweep_path = "/tmp/parameter-golf-solution/sweep_results.json"
    with open(sweep_path, "w") as f:
        json.dump({
            'best_lr': best_lr,
            'best_ws': best_ws,
            'best_bpb_2min': best_bpb,
            'lr_sweep': {str(k): v for k, v in results.get('lr_sweep', {}).items()},
            'ws_sweep': {str(k): v for k, v in ws_results.items()},
        }, f, indent=2)
    print(f"\nResults saved to {sweep_path}")

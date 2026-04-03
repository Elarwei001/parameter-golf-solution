#!/usr/bin/env python3
"""
Parameter Golf - Weight Sharing + Deep Layers
正确版 Weight Sharing：参数量不变，但更深！

Configs:
  baseline : 6L, dim=256, no sharing, 4.39M, depth=6
  ws_18a   : 3 unique × 6 passes, dim=352, 4.18M, depth=18
  ws_18b   : 6 unique × 3 passes, dim=256, 4.39M, depth=18

All use LeakyReLU² + Sliding Window + AdamW
"""
import sys, os, time, math, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ── Device ───────────────────────────────────────────────────────
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

# ── Primitives ───────────────────────────────────────────────────
from models.standard_gpt import RMSNorm, apply_rotary_pos_emb


class RotaryEmbeddingDynamic(nn.Module):
    """RoPE that auto-expands for longer sequences."""
    def __init__(self, dim, base_seq_len=512, base=10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(base_seq_len)

    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos", emb.cos())
        self.register_buffer("sin", emb.sin())
        self._cache_len = seq_len

    def forward(self, x, seq_len):
        if seq_len > self._cache_len:
            self._build_cache(seq_len * 2)
        return self.cos[:seq_len], self.sin[:seq_len]


class MLP_LeakyReLU2(nn.Module):
    """LeakyReLU² MLP."""
    def __init__(self, dim, mult=4):
        super().__init__()
        hidden = dim * mult
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.w2(F.leaky_relu(self.w1(x), negative_slope=0.5).square())


class AttentionSW(nn.Module):
    """Sliding-window causal attention."""
    def __init__(self, dim, n_heads, n_kv_heads=None, window_size=128):
        super().__init__()
        self.n_heads    = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.head_dim   = dim // n_heads
        self.n_rep      = n_heads // self.n_kv_heads
        self.window_size = window_size
        self.wq = nn.Linear(dim, n_heads          * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_kv_heads  * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_kv_heads  * self.head_dim, bias=False)
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
        attn = attn.masked_fill(mask, float('-inf'))
        attn = torch.nan_to_num(F.softmax(attn, dim=-1), nan=0.0)

        out = (attn @ v).transpose(1, 2).reshape(B, L, -1)
        return self.wo(out)


class BlockWS(nn.Module):
    """Sliding-Window + LeakyReLU² block."""
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


class GPT_WeightShared(nn.Module):
    """
    Weight-sharing GPT with LeakyReLU² + Sliding Window.

    If n_passes=1  → standard (no sharing)
    If n_passes>1  → n_unique_layers blocks cycled n_passes times
                      effective_depth = n_unique_layers × n_passes
    """
    def __init__(self, vocab_size, dim, n_unique_layers, n_passes,
                 n_heads, n_kv_heads, mlp_mult=4, max_seq_len=320, window_size=128):
        super().__init__()
        self.vocab_size     = vocab_size
        self.n_unique_layers = n_unique_layers
        self.n_passes       = n_passes

        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.rope    = RotaryEmbeddingDynamic(dim // n_heads, max_seq_len)
        self.blocks  = nn.ModuleList([
            BlockWS(dim, n_heads, n_kv_heads, mlp_mult, window_size)
            for _ in range(n_unique_layers)
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
        for _ in range(self.n_passes):
            for block in self.blocks:
                h = block(h, cos, sin)
        h = self.ln_f(h)
        return h @ self.tok_emb.weight.T  # tied embeddings

    def compute_loss(self, batch):
        logits  = self.forward(batch[:, :-1])
        targets = batch[:, 1:]
        loss    = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            targets.reshape(-1)
        )
        return {'loss': loss, 'ce_loss': loss}

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def effective_depth(self):
        return self.n_unique_layers * self.n_passes


# ── Training ─────────────────────────────────────────────────────
def train(model, name, duration_sec=600, log_every=100):
    model = model.to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-3,
                               betas=(0.9, 0.95), weight_decay=0.01)
    n_params = model.count_parameters()
    eff_depth = model.effective_depth
    print(f"\n{'='*65}")
    print(f"Experiment  : {name}")
    print(f"Params      : {n_params/1e6:.3f}M")
    print(f"Unique layers: {model.n_unique_layers}  passes: {model.n_passes}  eff_depth: {eff_depth}")
    print(f"Duration    : {duration_sec/60:.0f} min")
    print(f"{'='*65}")

    model.train()
    losses, step, start = [], 0, time.time()

    while time.time() - start < duration_sec:
        batch = get_batch(train_data)
        opt.zero_grad()
        out  = model.compute_loss(batch)
        out['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(out['ce_loss'].item())
        step += 1
        if step % log_every == 0:
            recent  = np.mean(losses[-50:])
            bpb     = recent / math.log(2)
            elapsed = time.time() - start
            print(f"  [{elapsed/60:.1f}min] step {step}: loss={recent:.4f} BPB={bpb:.4f}")

    # Val eval
    model.eval()
    val_losses = []
    with torch.no_grad():
        for _ in range(30):
            batch = get_batch(val_data)
            d     = model.compute_loss(batch)
            val_losses.append(d['ce_loss'].item())

    val_bpb   = np.mean(val_losses) / math.log(2)
    train_bpb = np.mean(losses[-100:]) / math.log(2) if losses else float('nan')
    elapsed   = time.time() - start

    print(f"\n✅ {name}: {step} steps, {elapsed/60:.1f}min")
    print(f"   Train BPB : {train_bpb:.4f}")
    print(f"   Val   BPB : {val_bpb:.4f}")
    return {
        'name': name,
        'steps': step,
        'train_bpb': train_bpb,
        'val_bpb': val_bpb,
        'params': n_params,
        'n_unique_layers': model.n_unique_layers,
        'n_passes': model.n_passes,
        'eff_depth': eff_depth,
    }


# ── Main ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Quick local test (60s per exp)')
    parser.add_argument('--exp', type=str, default='all', help='Which experiment: baseline|ws_18a|ws_18b|all')
    args = parser.parse_args()

    duration = 60 if args.quick else 600  # 1 min or 10 min

    # ── Sanity check ──────────────────────────────────────────────
    print("Sanity check (all 3 configs)...")
    for cfg in [
        dict(dim=64,  n_unique=2, n_passes=1, n_heads=4, n_kv_heads=2),
        dict(dim=64,  n_unique=2, n_passes=6, n_heads=4, n_kv_heads=2),
        dict(dim=64,  n_unique=4, n_passes=3, n_heads=4, n_kv_heads=2),
    ]:
        m = GPT_WeightShared(vocab_size=VOCAB_SIZE,
                             dim=cfg['dim'],
                             n_unique_layers=cfg['n_unique'],
                             n_passes=cfg['n_passes'],
                             n_heads=cfg['n_heads'],
                             n_kv_heads=cfg['n_kv_heads'],
                             max_seq_len=64,
                             window_size=32).to(DEVICE)
        x = torch.randint(0, VOCAB_SIZE, (2, 33)).to(DEVICE)
        loss = m.compute_loss(x)['loss']
        loss.backward()
        print(f"  {cfg['n_unique']}Lx{cfg['n_passes']}pass dim={cfg['dim']}: "
              f"params={m.count_parameters()/1e6:.3f}M depth={m.effective_depth} "
              f"loss={loss.item():.4f} ✅")
        del m
    print()

    results = []

    # ── Experiment configs ────────────────────────────────────────
    EXPS = {
        'baseline': dict(
            desc='baseline_no_sharing',
            dim=256, n_unique=6, n_passes=1,
            n_heads=8, n_kv_heads=4,
        ),
        'ws_18a': dict(
            desc='ws_3Lx6pass_dim352',
            dim=352, n_unique=3, n_passes=6,
            n_heads=8, n_kv_heads=4,
        ),
        'ws_18b': dict(
            desc='ws_6Lx3pass_dim256',
            dim=256, n_unique=6, n_passes=3,
            n_heads=8, n_kv_heads=4,
        ),
    }

    run_exps = list(EXPS.keys()) if args.exp == 'all' else [args.exp]

    for key in run_exps:
        cfg = EXPS[key]
        model = GPT_WeightShared(
            vocab_size=VOCAB_SIZE,
            dim=cfg['dim'],
            n_unique_layers=cfg['n_unique'],
            n_passes=cfg['n_passes'],
            n_heads=cfg['n_heads'],
            n_kv_heads=cfg['n_kv_heads'],
            mlp_mult=4,
            max_seq_len=SEQ_LEN + 64,
            window_size=128,
        )
        result = train(model, cfg['desc'], duration_sec=duration)
        results.append(result)

    # ── Save results ──────────────────────────────────────────────
    out_path = "/tmp/parameter-golf-solution/weight_sharing_results.json"
    existing = []
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
    existing.extend(results)
    with open(out_path, "w") as f:
        json.dump(existing, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "="*70)
    print("WEIGHT SHARING EXPERIMENT RESULTS")
    print("="*70)
    print(f"{'Model':<30} {'Params':>8} {'Depth':>6} {'Val BPB':>10}")
    print("-"*70)
    # Include prior best
    print(f"  {'leaky+sw (baseline)':28} {'4.39M':>8} {'6':>6} {'2.2821':>10}  (prior best)")
    for r in results:
        marker = " 🏆" if r['val_bpb'] < 2.2821 else ""
        print(f"  {r['name']:<28} {r['params']/1e6:>7.3f}M {r['eff_depth']:>6} {r['val_bpb']:>10.4f}{marker}")
    print("="*70)
    print(f"\nResults saved to {out_path}")

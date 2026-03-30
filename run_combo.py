#!/usr/bin/env python3
"""
Parameter Golf - Combo: LeakyReLU² + Sliding Window
第四组实验：叠加两个改进，看是否能突破 2.3 BPB
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

# ── Shared primitives ────────────────────────────────────────────
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
        causal_mask = rows < cols                          # future tokens
        window_mask = (rows - cols) > self.window_size    # too-far past
        attn = attn.masked_fill(causal_mask | window_mask, float('-inf'))
        attn = torch.nan_to_num(F.softmax(attn, dim=-1), nan=0.0)

        out = (attn @ v).transpose(1,2).reshape(B, L, -1)
        return self.wo(out)


class BlockCombo(nn.Module):
    """Sliding-window attention + LeakyReLU² MLP — the combo."""
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
    """LeakyReLU² + Sliding Window combined model."""
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
def train(model, name, duration_sec=900, log_every=100):
    model = model.to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-3,
                               betas=(0.9,0.95), weight_decay=0.01)
    n_params = model.count_parameters()
    print(f"\n{'='*60}")
    print(f"Experiment : {name}")
    print(f"Params     : {n_params/1e6:.2f}M")
    print(f"Duration   : {duration_sec/60:.0f} min")
    print(f"{'='*60}")

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
        if step % log_every == 0:
            recent  = np.mean(losses[-50:])
            bpb     = recent / math.log(2)
            elapsed = time.time() - start
            print(f"  [{elapsed/60:.1f}min] step {step}: loss={recent:.4f} BPB={bpb:.4f}")

    # Val eval
    model.eval()
    val_losses = []
    with torch.no_grad():
        for _ in range(20):
            batch = get_batch(val_data)
            d     = model.compute_loss(batch)
            val_losses.append(d['ce_loss'].item())

    val_bpb   = np.mean(val_losses) / math.log(2)
    train_bpb = np.mean(losses[-100:]) / math.log(2) if losses else float('nan')
    elapsed   = time.time() - start

    print(f"\n✅ {name}: {step} steps, {elapsed/60:.1f}min")
    print(f"   Train BPB : {train_bpb:.4f}")
    print(f"   Val   BPB : {val_bpb:.4f}")
    return {'name': name, 'steps': step, 'train_bpb': train_bpb,
            'val_bpb': val_bpb, 'params': n_params}


# ── Main ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Quick sanity check
    print("Sanity check...")
    m = GPT_Combo(vocab_size=VOCAB_SIZE, dim=64, n_layers=2,
                  n_heads=4, n_kv_heads=2, mlp_mult=4,
                  max_seq_len=32, window_size=16).to(DEVICE)
    x = torch.randint(0, VOCAB_SIZE, (2, 33)).to(DEVICE)
    out = m.compute_loss(x)
    print(f"  forward OK, loss={out['loss'].item():.4f} ✅")
    del m

    # Full combo run
    combo = GPT_Combo(
        vocab_size=VOCAB_SIZE, dim=256, n_layers=6,
        n_heads=8, n_kv_heads=4, mlp_mult=4,
        max_seq_len=SEQ_LEN + 64, window_size=128
    )
    result = train(combo, "leaky_relu2+sliding_window", duration_sec=900)

    # Load previous results and append
    prev_path = "/tmp/parameter-golf-solution/experiment_results.json"
    all_results = []
    if os.path.exists(prev_path):
        with open(prev_path) as f:
            all_results = json.load(f)
    all_results.append(result)
    with open(prev_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Final comparison table
    print("\n" + "="*65)
    print("FULL COMPARISON (all 4 experiments)")
    print("="*65)
    print(f"{'Model':<35} {'Params':>8} {'Val BPB':>10}")
    print("-"*65)
    for r in all_results:
        marker = " 🏆" if r['val_bpb'] == min(x['val_bpb'] for x in all_results) else ""
        print(f"  {r['name']:<33} {r['params']/1e6:>7.2f}M {r['val_bpb']:>10.4f}{marker}")
    print("="*65)
    print(f"\nResults saved to {prev_path}")

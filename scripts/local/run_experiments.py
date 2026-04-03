#!/usr/bin/env python3
"""
Parameter Golf 连续优化实验
Compare: Baseline SwiGLU vs LeakyReLU² vs Sliding Window (fixed)
"""
import sys, os, time, math, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ========================
# Device setup
# ========================
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Device: {DEVICE}")

# ========================
# Data preparation
# ========================
DATA_FILE = "/tmp/tinyshakespeare.txt"
VOCAB_SIZE = 256   # byte-level tokenizer
SEQ_LEN = 256
BATCH_SIZE = 32

def load_data():
    """Load byte-level tokenized tinyshakespeare."""
    with open(DATA_FILE, "rb") as f:
        raw = f.read()
    data = np.frombuffer(raw, dtype=np.uint8).copy()
    split = int(len(data) * 0.9)
    return data[:split], data[split:]

def get_batch(data, seq_len, batch_size):
    """Sample a random batch."""
    starts = np.random.randint(0, len(data) - seq_len - 1, batch_size)
    batch = np.stack([data[i:i+seq_len+1] for i in starts])
    batch = torch.from_numpy(batch.astype(np.int64)).to(DEVICE)
    return batch

train_data, val_data = load_data()
print(f"Train: {len(train_data)/1e6:.1f}M tokens, Val: {len(val_data)/1000:.0f}K tokens")

# ========================
# Model variants
# ========================

from models.standard_gpt import (
    RMSNorm, RotaryEmbedding, Attention, Block, apply_rotary_pos_emb, rotate_half
)


class MLP_SwiGLU(nn.Module):
    """Original SwiGLU MLP (baseline)."""
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        hidden = int(dim * mult * 2 / 3)
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MLP_LeakyReLU2(nn.Module):
    """LeakyReLU² MLP: leaky_relu(x, 0.5).square()."""
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        hidden = int(dim * mult)
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        h = F.leaky_relu(self.w1(x), negative_slope=0.5)
        return self.w2(h.square())


class Block_LeakyReLU2(nn.Module):
    """Transformer block with LeakyReLU² MLP."""
    def __init__(self, dim, n_heads, n_kv_heads=None, mlp_mult=4):
        super().__init__()
        self.ln1 = RMSNorm(dim)
        self.attn = Attention(dim, n_heads, n_kv_heads)
        self.ln2 = RMSNorm(dim)
        self.mlp = MLP_LeakyReLU2(dim, mlp_mult)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT_Base(nn.Module):
    """Base GPT model with swappable block type."""
    def __init__(self, vocab_size, dim, n_layers, n_heads, n_kv_heads,
                 mlp_mult, max_seq_len, block_cls=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        if block_cls is None:
            block_cls = Block

        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.rope = RotaryEmbedding(dim // n_heads, max_seq_len)
        self.blocks = nn.ModuleList([
            block_cls(dim, n_heads, n_kv_heads, mlp_mult)
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
        logits = self.forward(batch[:, :-1])
        targets = batch[:, 1:]
        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            targets.reshape(-1)
        )
        return {'loss': loss, 'ce_loss': loss}

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class RotaryEmbeddingDynamic(nn.Module):
    """RoPE that dynamically extends if seq_len > max_seq_len."""
    def __init__(self, dim, max_seq_len=4096, base=10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._extend(max_seq_len)

    def _extend(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos", emb.cos())
        self.register_buffer("sin", emb.sin())
        self._max_seq_len = seq_len

    def forward(self, x, seq_len):
        if seq_len > self._max_seq_len:
            self._extend(seq_len * 2)
        return self.cos[:seq_len], self.sin[:seq_len]


class GPT_SlidingWindow(nn.Module):
    """GPT with sliding window attention + fixed RoPE (dynamic extension)."""
    def __init__(self, vocab_size, dim, n_layers, n_heads, n_kv_heads,
                 mlp_mult, max_seq_len, window_size=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.window_size = window_size

        self.tok_emb = nn.Embedding(vocab_size, dim)
        # FIX: use dynamic RoPE that extends as needed
        self.rope = RotaryEmbeddingDynamic(dim // n_heads, max_seq_len)
        self.blocks = nn.ModuleList([
            BlockSW(dim, n_heads, n_kv_heads, mlp_mult, window_size)
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
        # Dynamic RoPE - handles any seq_len
        cos, sin = self.rope(h, L)
        for block in self.blocks:
            h = block(h, cos, sin)
        h = self.ln_f(h)
        return h @ self.tok_emb.weight.T

    def compute_loss(self, batch):
        logits = self.forward(batch[:, :-1])
        targets = batch[:, 1:]
        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            targets.reshape(-1)
        )
        return {'loss': loss, 'ce_loss': loss}

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AttentionSW(nn.Module):
    """Sliding-window attention."""
    def __init__(self, dim, n_heads, n_kv_heads=None, window_size=128):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.head_dim = dim // n_heads
        self.n_rep = n_heads // self.n_kv_heads
        self.window_size = window_size

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    def forward(self, x, cos, sin):
        B, L, _ = x.shape
        q = self.wq(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q, k = apply_rotary_pos_emb(
            q, k,
            cos.unsqueeze(0).unsqueeze(0),
            sin.unsqueeze(0).unsqueeze(0)
        )

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1
        )
        # Sliding window mask: positions more than window_size apart
        rows = torch.arange(L, device=x.device).unsqueeze(1)
        cols = torch.arange(L, device=x.device).unsqueeze(0)
        window_mask = (rows - cols) > self.window_size
        combined_mask = causal_mask | window_mask

        attn = attn.masked_fill(combined_mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        # Handle all-inf rows (first tokens in sliding window)
        attn = torch.nan_to_num(attn, nan=0.0)

        out = (attn @ v).transpose(1, 2).reshape(B, L, -1)
        return self.wo(out)


class BlockSW(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads=None, mlp_mult=4, window_size=128):
        super().__init__()
        self.ln1 = RMSNorm(dim)
        self.attn = AttentionSW(dim, n_heads, n_kv_heads, window_size)
        self.ln2 = RMSNorm(dim)
        self.mlp = MLP_SwiGLU(dim, mlp_mult)  # keep SwiGLU for fair comparison

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.mlp(self.ln2(x))
        return x


# ========================
# Training loop
# ========================
def train_model(model, name, duration_sec=900, log_interval=100):
    """Train for duration_sec seconds, return BPB."""
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.01
    )

    n_params = model.count_parameters()
    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    print(f"Params: {n_params/1e6:.2f}M")
    print(f"Training for {duration_sec/60:.0f} min...")
    print(f"{'='*60}")

    model.train()
    losses = []
    step = 0
    start = time.time()

    while True:
        elapsed = time.time() - start
        if elapsed >= duration_sec:
            break

        batch = get_batch(train_data, SEQ_LEN, BATCH_SIZE)
        optimizer.zero_grad()
        loss_dict = model.compute_loss(batch)
        loss_dict['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss_dict['ce_loss'].item())
        step += 1

        if step % log_interval == 0:
            recent = np.mean(losses[-50:])
            bpb = recent / math.log(2)
            mins = elapsed / 60
            print(f"  [{mins:.1f}min] step {step}: loss={recent:.4f} BPB={bpb:.4f}")

    # Final eval on val set
    model.eval()
    val_losses = []
    with torch.no_grad():
        for _ in range(20):
            batch = get_batch(val_data, SEQ_LEN, BATCH_SIZE)
            loss_dict = model.compute_loss(batch)
            val_losses.append(loss_dict['ce_loss'].item())

    val_loss = np.mean(val_losses)
    val_bpb = val_loss / math.log(2)
    train_bpb = np.mean(losses[-100:]) / math.log(2) if losses else float('nan')

    elapsed = time.time() - start
    print(f"\n✅ {name} DONE: {step} steps, {elapsed/60:.1f}min")
    print(f"   Train BPB: {train_bpb:.4f}")
    print(f"   Val   BPB: {val_bpb:.4f}")
    return {'name': name, 'steps': step, 'train_bpb': train_bpb, 'val_bpb': val_bpb, 'params': n_params}


# ========================
# Model factory
# ========================
MODEL_KWARGS = dict(
    vocab_size=VOCAB_SIZE,
    dim=256,
    n_layers=6,
    n_heads=8,
    n_kv_heads=4,
    mlp_mult=4,
    max_seq_len=SEQ_LEN + 64,  # some headroom
)

DURATION = 900  # 15 minutes per run

if __name__ == "__main__":
    results = []

    # ─── Experiment 1: Baseline SwiGLU ───
    print("\n>>> [1/3] Baseline: SwiGLU")
    baseline = GPT_Base(**MODEL_KWARGS, block_cls=Block)
    r = train_model(baseline, "baseline_swiglu", DURATION)
    results.append(r)
    del baseline

    # ─── Experiment 2: LeakyReLU² ───
    print("\n>>> [2/3] LeakyReLU²: leaky_relu(x, 0.5).square()")
    leaky = GPT_Base(**MODEL_KWARGS, block_cls=Block_LeakyReLU2)
    r = train_model(leaky, "leaky_relu2", DURATION)
    results.append(r)
    del leaky

    # ─── Verify sliding window bug ───
    print("\n>>> Verifying RoPE overflow bug...")
    from models.standard_gpt import StandardGPT
    buggy = StandardGPT(
        vocab_size=VOCAB_SIZE, dim=128, n_layers=2, n_heads=4, n_kv_heads=2,
        max_seq_len=64  # intentionally small
    ).to(DEVICE)
    x_overflow = torch.randint(0, VOCAB_SIZE, (2, 80)).to(DEVICE)  # 80 > 64
    try:
        buggy(x_overflow)
        print("   (no crash - unexpected)")
    except Exception as e:
        print(f"   ✅ Bug confirmed: {type(e).__name__}: {e}")
        print("   Fix: use RotaryEmbeddingDynamic that extends past max_seq_len")

    # ─── Experiment 3: Sliding Window (fixed) ───
    print("\n>>> [3/3] Sliding Window (fixed RoPE)")
    sw_kwargs = dict(
        vocab_size=VOCAB_SIZE, dim=256, n_layers=6, n_heads=8, n_kv_heads=4,
        mlp_mult=4,
        max_seq_len=SEQ_LEN + 64,  # set proper headroom
        window_size=128
    )
    sliding = GPT_SlidingWindow(**sw_kwargs)
    r = train_model(sliding, "sliding_window_fixed", DURATION)
    results.append(r)
    del sliding

    # ─── Final Summary ───
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    print(f"{'Model':<30} {'Params':>10} {'Train BPB':>12} {'Val BPB':>12}")
    print("-"*60)
    for r in results:
        print(f"{r['name']:<30} {r['params']/1e6:>9.2f}M {r['train_bpb']:>12.4f} {r['val_bpb']:>12.4f}")
    print("="*60)

    # Save results
    import json
    out_path = "/tmp/parameter-golf-solution/experiment_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

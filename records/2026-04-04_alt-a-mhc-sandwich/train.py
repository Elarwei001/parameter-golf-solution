"""
Sandwich MLP Experiment
=======================

## Purpose
Compare Uniform MLP vs Sandwich MLP on Alternating Attention + mHC architecture.
Sandwich MLP assigns larger hidden dim to important layers (shallow & deep) and
smaller hidden dim to middle layers, based on mHC β_mlp observations.

## Design
- Baseline: Alt-A mHC-scratch (all layers MLP hidden = dim*3), BPB 1.4777
- Experiment: Sandwich MLP (layers 0-3: 3x, 4-16: 1.2x, 17-19: 3x)
- mHC params: learned from scratch (init to 1.0)
- Only variable: MLP hidden dim distribution

## Run
```bash
# Uniform (baseline, should reproduce ~1.4777)
modal run --detach scripts/modal/modal_sandwich_mlp.py --style uniform

# Sandwich
modal run --detach scripts/modal/modal_sandwich_mlp.py --style sandwich
```

## Related
- mHC learning script: scripts/modal/modal_alt_mhc_learn.py
- Alternating Attention: scripts/modal/modal_alternating_attn.py
"""
import modal
import os
import math

app = modal.App("sandwich-mlp")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch==2.5.1",
        "numpy",
    ])
)

data_volume = modal.Volume.from_name("parameter-golf-data", create_if_missing=True)

MLP_SCALES_UNIFORM = [3.0] * 20
MLP_SCALES_SANDWICH = [3.0]*4 + [1.2]*13 + [3.0]*3  # 0-3: high, 4-16: low, 17-19: high
MLP_SCALES_FRONT = [3.0]*4 + [1.2]*16  # 0-3: high, 4-19: low (no deep recovery)


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={"/data": data_volume},
    timeout=7200,
)
def train_sandwich(
    style: str = "uniform",  # "uniform" or "sandwich"
    seed: int = 42,
    dim: int = 384,
    n_layers: int = 20,
    n_heads: int = 8,
    n_kv_heads: int = 4,
    local_window: int = 128,
    lr: float = 1e-3,
    batch_size: int = 64,
    seq_len: int = 256,
    steps: int = 5000,
):
    """Train Alternating Attention + mHC with Uniform or Sandwich MLP."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import time
    import json

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    VOCAB_SIZE = 8192
    DATA_DIR = "/data/datasets/fineweb10B_sp8192"
    HEADER_SIZE = 256 * 4
    BYTES_PER_TOKEN = 3.67

    if style == "sandwich":
        mlp_scales = MLP_SCALES_SANDWICH
    elif style == "front":
        mlp_scales = MLP_SCALES_FRONT
    else:
        mlp_scales = MLP_SCALES_UNIFORM

    print("=" * 70)
    print(f"Sandwich MLP Experiment: {style.upper()}")
    print(f"  dim={dim}, layers={n_layers}")
    print(f"  MLP scales: {mlp_scales}")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    train_files = sorted([f for f in os.listdir(DATA_DIR) if 'train' in f])
    val_files = sorted([f for f in os.listdir(DATA_DIR) if 'val' in f])

    train_data = []
    for f in train_files[:5]:
        with open(os.path.join(DATA_DIR, f), 'rb') as fp:
            fp.seek(HEADER_SIZE)
            data = np.frombuffer(fp.read(), dtype=np.uint16)
        train_data.append(data)
    train_data = np.concatenate(train_data)
    train_tokens = torch.from_numpy(train_data.astype(np.int64))

    val_data = []
    for f in val_files:
        with open(os.path.join(DATA_DIR, f), 'rb') as fp:
            fp.seek(HEADER_SIZE)
            data = np.frombuffer(fp.read(), dtype=np.uint16)
        val_data.append(data)
    val_data = np.concatenate(val_data)
    val_tokens = torch.from_numpy(val_data.astype(np.int64))

    print(f"  Train: {len(train_tokens)/1e6:.1f}M tokens")
    print(f"  Val: {len(val_tokens)/1e6:.1f}M tokens")

    # Model definition
    class RMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

    class RotaryEmbedding(nn.Module):
        def __init__(self, dim, max_seq_len=4096):
            super().__init__()
            inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
            t = torch.arange(max_seq_len)
            freqs = torch.outer(t, inv_freq)
            self.register_buffer('cos', freqs.cos())
            self.register_buffer('sin', freqs.sin())

        def forward(self, x, offset=0):
            seq_len = x.shape[1]
            cos = self.cos[offset:offset+seq_len].unsqueeze(0).unsqueeze(2)
            sin = self.sin[offset:offset+seq_len].unsqueeze(0).unsqueeze(2)
            x1, x2 = x[..., ::2], x[..., 1::2]
            return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    class AlternatingAttention(nn.Module):
        def __init__(self, dim, n_heads, n_kv_heads=None, local_window=128, is_global=True):
            super().__init__()
            self.n_heads = n_heads
            self.n_kv_heads = n_kv_heads or n_heads
            self.head_dim = dim // n_heads
            self.local_window = local_window
            self.is_global = is_global

            self.wq = nn.Linear(dim, dim, bias=False)
            self.wk = nn.Linear(dim, self.head_dim * self.n_kv_heads, bias=False)
            self.wv = nn.Linear(dim, self.head_dim * self.n_kv_heads, bias=False)
            self.wo = nn.Linear(dim, dim, bias=False)
            self.rope = RotaryEmbedding(self.head_dim)

        def forward(self, x):
            B, T, C = x.shape

            q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
            k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
            v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

            q = self.rope(q)
            k = self.rope(k)

            if self.n_kv_heads < self.n_heads:
                k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=2)
                v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=2)

            q, k, v = [t.transpose(1, 2) for t in (q, k, v)]

            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale

            diag_mask = torch.eye(T, device=x.device, dtype=torch.bool)
            scores = scores.masked_fill(diag_mask, 0.0)

            rows = torch.arange(T, device=x.device).view(-1, 1)
            cols = torch.arange(T, device=x.device).view(1, -1)
            causal_mask = cols > rows

            if self.is_global:
                mask = causal_mask
            else:
                window_mask = (rows - cols) > self.local_window
                mask = causal_mask | window_mask

            scores = scores.masked_fill(mask, float('-inf'))
            attn = F.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)
            out = out.transpose(1, 2).contiguous().view(B, T, C)
            return self.wo(out)

    class MLP(nn.Module):
        def __init__(self, dim, hidden_dim):
            super().__init__()
            self.w1 = nn.Linear(dim, hidden_dim, bias=False)
            self.w2 = nn.Linear(hidden_dim, dim, bias=False)

        def forward(self, x):
            h = self.w1(x)
            h = F.leaky_relu(h, 0.5) ** 2
            return self.w2(h)

    class SandwichBlock(nn.Module):
        def __init__(self, dim, n_heads, n_kv_heads, local_window, layer_idx, n_layers, mlp_scale):
            super().__init__()
            self.layer_idx = layer_idx

            is_last = (layer_idx == n_layers - 1)
            is_global = (layer_idx % 2 == 0) or is_last
            self.is_global = is_global
            self.attn_type = "Global" if is_global else f"Local(w={local_window})"

            self.attn = AlternatingAttention(dim, n_heads, n_kv_heads, local_window, is_global)
            mlp_hidden = int(dim * mlp_scale)
            self.mlp = MLP(dim, mlp_hidden)
            self.mlp_scale = mlp_scale
            self.mlp_hidden = mlp_hidden
            self.ln1 = RMSNorm(dim)
            self.ln2 = RMSNorm(dim)

            # mHC parameters - learned from scratch (init to 1.0)
            self.alpha_attn = nn.Parameter(torch.ones(1))
            self.beta_attn = nn.Parameter(torch.ones(1))
            self.alpha_mlp = nn.Parameter(torch.ones(1))
            self.beta_mlp = nn.Parameter(torch.ones(1))

        def forward(self, x):
            attn_out = self.attn(self.ln1(x))
            x = self.alpha_attn * x + self.beta_attn * attn_out

            mlp_out = self.mlp(self.ln2(x))
            x = self.alpha_mlp * x + self.beta_mlp * mlp_out
            return x

        def get_mhc_params(self):
            return {
                'alpha_attn': self.alpha_attn.item(),
                'beta_attn': self.beta_attn.item(),
                'alpha_mlp': self.alpha_mlp.item(),
                'beta_mlp': self.beta_mlp.item(),
                'layer': self.layer_idx,
                'attn_type': self.attn_type,
                'mlp_scale': self.mlp_scale,
                'mlp_hidden': self.mlp_hidden,
            }

    class SandwichTransformer(nn.Module):
        def __init__(self, vocab_size, dim, n_layers, n_heads, n_kv_heads, local_window, mlp_scales):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, dim)
            self.blocks = nn.ModuleList([
                SandwichBlock(dim, n_heads, n_kv_heads, local_window, i, n_layers, mlp_scales[i])
                for i in range(n_layers)
            ])
            self.ln_f = RMSNorm(dim)
            self.lm_head = nn.Linear(dim, vocab_size, bias=False)
            self.embed.weight = self.lm_head.weight

        def forward(self, x, targets=None):
            x = self.embed(x)
            for block in self.blocks:
                x = block(x)
            x = self.ln_f(x)
            logits = self.lm_head(x)

            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss

    # Create model
    model = SandwichTransformer(
        vocab_size=VOCAB_SIZE,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        local_window=local_window,
        mlp_scales=mlp_scales,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    global_layers = sum(1 for b in model.blocks if b.is_global)
    local_layers = n_layers - global_layers

    print(f"\nModel: {total_params/1e6:.2f}M params")
    print(f"  Global layers: {global_layers}, Local layers: {local_layers}")

    print(f"\nLayer configuration ({style}):")
    print(f"  {'Layer':>5} {'Type':<12} {'MLP scale':>10} {'MLP hidden':>11}")
    print(f"  {'-'*42}")
    for block in model.blocks:
        print(f"  {block.layer_idx:>5} {block.attn_type:<12} {block.mlp_scale:>10.1f}x {block.mlp_hidden:>11}")

    # Count params breakdown
    attn_params = sum(p.numel() for b in model.blocks for p in b.attn.parameters())
    mlp_params = sum(p.numel() for b in model.blocks for p in b.mlp.parameters())
    embed_params = sum(p.numel() for p in model.embed.parameters())
    print(f"\nParameter breakdown:")
    print(f"  Embedding/LM Head (shared): {embed_params/1e6:.2f}M")
    print(f"  Attention (all layers):     {attn_params/1e6:.2f}M")
    print(f"  MLP (all layers):           {mlp_params/1e6:.2f}M")
    print(f"  Total:                      {total_params/1e6:.2f}M")

    # Training
    min_lr_ratio = 0.1
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)

    def get_batch(split):
        data = train_tokens if split == 'train' else val_tokens
        max_start = len(data) - seq_len - 1
        ix = torch.randint(0, max_start, (batch_size,))
        x = torch.stack([data[i:i+seq_len] for i in ix]).long().to(device)
        y = torch.stack([data[i+1:i+seq_len+1] for i in ix]).long().to(device)
        return x, y

    def cosine_lr(step):
        if step < 200:
            return step / 200
        progress = (step - 200) / (steps - 200)
        return min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + math.cos(math.pi * progress))

    print(f"\n[TRAIN] Starting training ({steps} steps, style={style})...\n")
    start_time = time.time()

    for step in range(1, steps + 1):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * cosine_lr(step)

        x, y = get_batch('train')
        _, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 500 == 0:
            elapsed = time.time() - start_time
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Step {step}/{steps} | Loss {loss.item():.4f} | LR {current_lr:.2e} | Time {elapsed:.0f}s")

        # Print mHC params at intervals
        if step % 2500 == 0:
            print(f"\n  --- mHC Parameters at step {step} ---")
            print(f"  {'Layer':>5} {'Type':<12} {'MLP':>5} {'a_attn':>7} {'b_attn':>7} {'a_mlp':>7} {'b_mlp':>7}")
            for block in model.blocks:
                p = block.get_mhc_params()
                print(f"  {p['layer']:>5} {p['attn_type']:<12} {p['mlp_scale']:>4.1f}x {p['alpha_attn']:>7.3f} {p['beta_attn']:>7.3f} {p['alpha_mlp']:>7.3f} {p['beta_mlp']:>7.3f}")
            print()

    print(f"\nTraining completed: {time.time() - start_time:.0f}s")

    # Validation
    model.eval()
    val_losses = []
    with torch.no_grad():
        for _ in range(100):
            x, y = get_batch('val')
            _, loss = model(x, y)
            val_losses.append(loss.item())

    val_loss = sum(val_losses) / len(val_losses)
    val_bpb = (val_loss / math.log(2)) * (1.0 / BYTES_PER_TOKEN)

    print("\n" + "=" * 70)
    print(f"[RESULTS] style={style}")
    print("=" * 70)
    print(f"  Val Loss: {val_loss:.4f}")
    print(f"  Val BPB:  {val_bpb:.4f}")
    print(f"  Params:   {total_params/1e6:.2f}M")
    print(f"  vs Baseline (1.5025): {(val_bpb - 1.5025) / 1.5025 * 100:+.2f}%")
    print(f"  vs Alt-A mHC-scratch Uniform (1.4777): {(val_bpb - 1.4777) / 1.4777 * 100:+.2f}%")
    print("=" * 70)

    # Print final mHC params
    print(f"\n[FINAL mHC PARAMETERS ({style})]")
    print(f"  {'Layer':>5} {'Type':<12} {'MLP':>5} {'a_attn':>7} {'b_attn':>7} {'a_mlp':>7} {'b_mlp':>7}")
    print("  " + "-" * 60)
    final_params = []
    for block in model.blocks:
        p = block.get_mhc_params()
        final_params.append(p)
        print(f"  {p['layer']:>5} {p['attn_type']:<12} {p['mlp_scale']:>4.1f}x {p['alpha_attn']:>7.3f} {p['beta_attn']:>7.3f} {p['alpha_mlp']:>7.3f} {p['beta_mlp']:>7.3f}")

    # Save results
    checkpoint_dir = f"/data/checkpoints/sandwich_mlp/{style}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    results = {
        'style': style,
        'config': {
            'dim': dim, 'n_layers': n_layers, 'n_heads': n_heads,
            'n_kv_heads': n_kv_heads, 'local_window': local_window,
            'steps': steps, 'lr': lr, 'batch_size': batch_size, 'seq_len': seq_len,
            'mlp_scales': mlp_scales,
        },
        'val_loss': val_loss,
        'val_bpb': val_bpb,
        'total_params': total_params,
        'final_params': final_params,
    }

    results_path = os.path.join(checkpoint_dir, f"sandwich_{style}_bpb{val_bpb:.4f}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] Results: {results_path}")

    ckpt_path = os.path.join(checkpoint_dir, f"sandwich_{style}_step{steps}.pt")
    torch.save({
        'step': steps,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_bpb': val_bpb,
        'final_params': final_params,
        'config': results['config'],
    }, ckpt_path)
    print(f"[SAVED] Checkpoint: {ckpt_path}")

    data_volume.commit()

    return {
        'style': style,
        'val_loss': val_loss,
        'val_bpb': val_bpb,
        'total_params': total_params,
        'final_params': final_params,
    }


@app.local_entrypoint()
def main(
    style: str = "uniform",
):
    """Run Sandwich MLP experiment. style='uniform' or 'sandwich'"""
    assert style in ("uniform", "sandwich", "front"), f"style must be 'uniform', 'sandwich', or 'front', got '{style}'"

    result = train_sandwich.remote(style=style)

    print(f"\n[FINAL] style={result['style']}")
    print(f"  BPB: {result['val_bpb']:.4f}")
    print(f"  Params: {result['total_params']/1e6:.2f}M")
    print(f"  vs Baseline (1.5025): {(result['val_bpb'] - 1.5025) / 1.5025 * 100:+.2f}%")
    print(f"  vs Alt-A mHC-scratch Uniform (1.4777): {(result['val_bpb'] - 1.4777) / 1.4777 * 100:+.2f}%")

"""
Sandwich MLP + QAT 30L dim=544 Scylla (vocab=998) Experiment
==============================================================

Option A: dim=544, n_heads=8, n_kv=4, head_dim=68
- 9 layers at 3.0x MLP + 21 layers at 1.2x MLP (Sandwich)
- Scylla tokenizer (vocab=998) → tiny embedding (0.90MB FP16)
- Total ~15.89MB with FP16 embedding + ternary weights
- mHC, alt-attn, adaptive QAT, 40k steps

## Run
```bash
modal run --detach scripts/modal/modal_sandwich_qat_30l_dim544_scylla.py
```
"""
import modal
import os
import math

app = modal.App("sandwich-qat-30l-dim544-scylla")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch==2.5.1",
        "numpy",
    ])
)

data_volume = modal.Volume.from_name("parameter-golf-data", create_if_missing=True)

# dim=544 config: 9 layers at 3x, 21 layers at 1.2x
MLP_SCALES_SANDWICH_30L = [3.0]*9 + [1.2]*21


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={"/data": data_volume},
    timeout=14400,
)
def train_sandwich_qat(
    dim: int = 544,
    n_layers: int = 30,
    n_heads: int = 8,
    n_kv_heads: int = 4,
    local_window: int = 128,
    vocab_size: int = 998,  # Scylla tokenizer!
    max_seq_len: int = 256,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    weight_decay: float = 0.1,
    warmup_steps: int = 1000,
    max_steps: int = 40000,
    qat_start_step: int = -1,  # -1 = auto (when loss plateaus)
    checkpoint_every: int = 10000,
    data_dir: str = "/data/datasets/fineweb10B_scylla",
    tokenizer_meta_path: str = "/data/tokenizers/scylla/candidate.meta.npz",
):
    """Train Sandwich QAT 30L dim=544 with Scylla tokenizer."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import time
    import json

    device = torch.device("cuda")
    mlp_scales = MLP_SCALES_SANDWICH_30L

    print("=" * 70)
    print("Sandwich MLP + QAT 30L dim=544 Scylla Experiment")
    print(f"  dim={dim}, layers={n_layers}, n_heads={n_heads}, n_kv={n_kv_heads}")
    print(f"  vocab_size={vocab_size} (Scylla)")
    print(f"  head_dim={dim//n_heads}, local_window={local_window}")
    print(f"  MLP scales: {mlp_scales}")
    print(f"  max_steps={max_steps}, batch_size={batch_size}")
    print("=" * 70)

    # ── Load tokenizer metadata for BPB calculation ──────────
    BYTES_PER_TOKEN = None  # Will calculate from metadata
    if os.path.exists(tokenizer_meta_path):
        meta = np.load(tokenizer_meta_path)
        base_bytes = np.asarray(meta["base_bytes"], dtype=np.float32)
        BYTES_PER_TOKEN = float(base_bytes.mean())
        print(f"  Tokenizer meta loaded: avg bytes/token = {BYTES_PER_TOKEN:.4f}")
    else:
        print(f"  ⚠️ Tokenizer meta not found at {tokenizer_meta_path}")
        print(f"  Falling back to estimate for BPB calculation")

    # ── Ternary Quantization with STE ─────────────────────────
    class TernaryQuantize(torch.autograd.Function):
        @staticmethod
        def forward(ctx, weight):
            scale = weight.abs().mean()
            w_quant = torch.clamp(torch.round(weight / scale), -1, 1)
            ctx.save_for_backward(weight)
            return w_quant * scale

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output

    def ternary_quantize(weight):
        return TernaryQuantize.apply(weight)

    # ── Model components ──────────────────────────────────────
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

    class QATLinear(nn.Module):
        def __init__(self, in_features, out_features, bias=False):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.weight = nn.Parameter(torch.empty(out_features, in_features))
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            self.register_parameter('bias', None)
            self.qat_enabled = False

        def forward(self, x):
            if self.qat_enabled:
                w = ternary_quantize(self.weight)
            else:
                w = self.weight
            return F.linear(x, w, self.bias)

        def enable_qat(self):
            self.qat_enabled = True

    class AlternatingAttention(nn.Module):
        def __init__(self, dim, n_heads, n_kv_heads=None, local_window=128, is_global=True):
            super().__init__()
            self.n_heads = n_heads
            self.n_kv_heads = n_kv_heads or n_heads
            self.head_dim = dim // n_heads
            self.local_window = local_window
            self.is_global = is_global

            self.wq = QATLinear(dim, dim)
            self.wk = QATLinear(dim, self.head_dim * self.n_kv_heads)
            self.wv = QATLinear(dim, self.head_dim * self.n_kv_heads)
            self.wo = QATLinear(dim, dim)
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
            self.w1 = QATLinear(dim, hidden_dim)
            self.w2 = QATLinear(hidden_dim, dim)

        def forward(self, x):
            h = self.w1(x)
            h = F.leaky_relu(h, 0.5) ** 2
            return self.w2(h)

    class SandwichQATBlock(nn.Module):
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

    class SandwichQATTransformer(nn.Module):
        def __init__(self, vocab_size, dim, n_layers, n_heads, n_kv_heads, local_window, mlp_scales):
            super().__init__()
            self.vocab_size = vocab_size
            self.embed = nn.Embedding(vocab_size, dim)
            self.blocks = nn.ModuleList([
                SandwichQATBlock(dim, n_heads, n_kv_heads, local_window, i, n_layers, mlp_scales[i])
                for i in range(n_layers)
            ])
            self.ln_f = RMSNorm(dim)
            self.lm_head = QATLinear(dim, vocab_size)
            self.embed.weight = self.lm_head.weight

        def forward(self, x, targets=None):
            x = self.embed(x)
            for block in self.blocks:
                x = block(x)
            x = self.ln_f(x)
            logits = self.lm_head(x)
            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
            return logits, loss

        def enable_qat(self):
            for module in self.modules():
                if isinstance(module, QATLinear):
                    module.enable_qat()

    # ── Build model ───────────────────────────────────────────
    model = SandwichQATTransformer(
        vocab_size=vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        local_window=local_window,
        mlp_scales=mlp_scales,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel params: {total_params/1e6:.2f}M")

    # Print per-layer info
    print("\nPer-layer config:")
    print(f"  {'Layer':>5} {'Type':<15} {'Scale':>5} {'Hidden':>7}")
    print(f"  {'-'*40}")
    for i in range(n_layers):
        print(f"  {i:>5} {model.blocks[i].attn_type:<15} {model.blocks[i].mlp_scale:>5.1f}x {model.blocks[i].mlp_hidden:>7}")

    # Calculate quantized size
    ternary_params = 0
    fp_params = 0
    for name, param in model.named_parameters():
        if isinstance(param, nn.Parameter):
            if 'embed' in name or 'ln' in name or 'alpha' in name or 'beta' in name:
                fp_params += param.numel()
            else:
                ternary_params += param.numel()

    ternary_mb = ternary_params * 1.58 / 8 / 1e6
    fp16_emb_mb = vocab_size * dim * 2 / 1e6
    fp32_mb = fp_params * 4 / 1e6
    total_mb = ternary_mb + fp16_emb_mb + fp32_mb
    print(f"\n  Estimated size: ternary={ternary_mb:.2f}MB + fp16_emb={fp16_emb_mb:.2f}MB + fp32={fp32_mb:.2f}MB = {total_mb:.2f}MB")
    if total_mb > 16:
        print(f"  ⚠️ OVER 16MB!")
    else:
        print(f"  ✅ Under 16MB")

    # ── Load data ─────────────────────────────────────────────
    print("\nLoading data...")
    HEADER_SIZE = 256 * 4  # 256 uint32 header
    data_files = sorted([f for f in os.listdir(data_dir) if f.startswith('fineweb_train')])
    val_files = sorted([f for f in os.listdir(data_dir) if 'val' in f])

    train_data = []
    for f in data_files:
        with open(os.path.join(data_dir, f), 'rb') as fp:
            fp.seek(HEADER_SIZE)
            data = np.frombuffer(fp.read(), dtype=np.uint16)
            train_data.append(data)
        print(f"  Train: {f} -> {len(data)/1e6:.1f}M tokens")
    train_data = np.concatenate(train_data)
    print(f"  Total train tokens: {len(train_data)/1e6:.1f}M")

    val_data = []
    for f in val_files:
        with open(os.path.join(data_dir, f), 'rb') as fp:
            fp.seek(HEADER_SIZE)
            data = np.frombuffer(fp.read(), dtype=np.uint16)
            val_data.append(data)
    val_data = np.concatenate(val_data)
    print(f"  Total val tokens: {len(val_data)/1e6:.1f}M")

    # ── Training setup ────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.95))

    def get_lr(step):
        if step < warmup_steps:
            return learning_rate * step / warmup_steps
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        return learning_rate * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

    # QAT schedule
    qat_enabled = False
    ema_loss = None
    qat_start = qat_start_step if qat_start_step > 0 else None

    # Checkpoint dir
    ckpt_dir = "/data/checkpoints/sandwich_qat_30l_dim544_scylla"
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────
    print("\n[TRAIN] Starting training...")
    start_time = time.time()

    for step in range(max_steps):
        model.train()

        # Sample batch
        starts = np.random.randint(0, len(train_data) - max_seq_len - 1, batch_size)
        batch = np.stack([train_data[i:i+max_seq_len+1] for i in starts])
        x = torch.from_numpy(batch[:, :-1].astype(np.int64)).to(device)
        y = torch.from_numpy(batch[:, 1:].astype(np.int64)).to(device)

        # Clip to vocab range
        x = torch.clamp(x, 0, vocab_size - 1)
        y = torch.clamp(y, 0, vocab_size - 1)

        logits, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # EMA loss for QAT trigger
        if ema_loss is None:
            ema_loss = loss.item()
        else:
            ema_loss = 0.99 * ema_loss + 0.01 * loss.item()

        # Auto QAT: enable when loss improvement slows
        if not qat_enabled and qat_start is None:
            if step > 5000 and ema_loss < 4.0:
                qat_start = step
                print(f"\n[QAT] Auto-enabling QAT at step {step} (ema_loss={ema_loss:.4f})")
                model.enable_qat()
                qat_enabled = True

        if not qat_enabled and qat_start is not None and step >= qat_start:
            model.enable_qat()
            qat_enabled = True

        # Logging
        if step % 500 == 0:
            lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - start_time
            tokens_per_sec = (step + 1) * batch_size * max_seq_len / elapsed if elapsed > 0 else 0
            print(f"  Step {step:>6d} | Loss {loss.item():.4f} | EMA {ema_loss:.4f} | LR {lr:.6f} | QAT {'ON' if qat_enabled else 'OFF'} | {tokens_per_sec:.0f} tok/s")

        # Validation
        if (step + 1) % checkpoint_every == 0 or step == max_steps - 1:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for _ in range(100):
                    starts = np.random.randint(0, len(val_data) - max_seq_len - 1, batch_size)
                    batch = np.stack([val_data[i:i+max_seq_len+1] for i in starts])
                    x = torch.from_numpy(batch[:, :-1].astype(np.int64)).to(device)
                    y = torch.from_numpy(batch[:, 1:].astype(np.int64)).to(device)
                    x = torch.clamp(x, 0, vocab_size - 1)
                    y = torch.clamp(y, 0, vocab_size - 1)
                    _, vloss = model(x, y)
                    val_losses.append(vloss.item())

            avg_val_loss = sum(val_losses) / len(val_losses)
            if BYTES_PER_TOKEN:
                val_bpb = (avg_val_loss / math.log(2)) / BYTES_PER_TOKEN
            else:
                val_bpb = (avg_val_loss / math.log(2)) * (1.0 / 3.67)  # fallback

            print(f"\n[VAL] Step {step+1} | Val Loss {avg_val_loss:.4f} | Val BPB {val_bpb:.4f}")

            # Save checkpoint
            ckpt_path = os.path.join(ckpt_dir, f"sandwich_qat_30l_dim544_scylla_step{step+1}.pt")
            torch.save({
                'step': step + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_bpb': val_bpb,
                'config': {
                    'dim': dim, 'n_layers': n_layers, 'n_heads': n_heads,
                    'n_kv_heads': n_kv_heads, 'vocab_size': vocab_size,
                    'mlp_scales': mlp_scales, 'local_window': local_window,
                    'tokenizer': 'scylla_998',
                }
            }, ckpt_path)
            data_volume.commit()
            print(f"[SAVED] Checkpoint: {ckpt_path}")

            # Save results JSON
            results = {
                'style': 'sandwich_qat_30l_dim544_scylla',
                'step': step + 1,
                'val_loss': avg_val_loss,
                'val_bpb': val_bpb,
                'config': {
                    'dim': dim, 'n_layers': n_layers, 'n_heads': n_heads,
                    'n_kv_heads': n_kv_heads, 'vocab_size': vocab_size,
                    'mlp_scales': mlp_scales,
                }
            }
            results_path = os.path.join(ckpt_dir, f"sandwich_qat_30l_dim544_scylla_bpb{val_bpb:.4f}.json")
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            data_volume.commit()
            print(f"[SAVED] Results: {results_path}")

            model.train()

    total_time = time.time() - start_time
    print(f"\n[DONE] Total training time: {total_time/3600:.1f} hours")


@app.local_entrypoint()
def main():
    train_sandwich_qat.remote()

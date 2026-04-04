"""
Sandwich MLP + QAT (Quantization-Aware Training) Experiment
============================================================

## Purpose
Combine Sandwich MLP (3x/1.2x/3x) with Ternary QAT (1.58-bit) quantization.
mHC parameters are trained from scratch (not quantized, kept as FP32).

## Design
- MLP: Sandwich config (layers 0-3: 3x, 4-16: 1.2x, 17-19: 3x)
- QAT: Ternary {-1, 0, +1} with Straight-Through Estimator (STE)
- mHC: 4 params per layer, FP32, init to 1.0
- Logs mHC params every 1000 steps to track evolution

## Run
```bash
modal run --detach scripts/modal/modal_sandwich_qat.py
```

## Related
- Sandwich MLP: scripts/modal/modal_sandwich_mlp.py
- QAT v2: scripts/modal/modal_qat_v2.py
"""
import modal
import os
import math

app = modal.App("sandwich-qat")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch==2.5.1",
        "numpy",
    ])
)

data_volume = modal.Volume.from_name("parameter-golf-data", create_if_missing=True)

MLP_SCALES_SANDWICH = [3.0]*4 + [1.2]*13 + [3.0]*3


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={"/data": data_volume},
    timeout=7200,
)
def train_sandwich_qat(
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
    warmup_steps: int = 500,
    qat_start_step: int = -1,  # -1 = auto (adaptive), 0 = from start, >0 = fixed step
    qat_warmup_steps: int = 2000,  # min FP32 steps before auto-switch
    loss_ema_alpha: float = 0.99,  # EMA smoothing for loss tracking
    qat_switch_threshold: float = 0.001,  # switch when loss_rate < this
):
    """Train Sandwich MLP + QAT + mHC (from scratch)."""
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
    mlp_scales = MLP_SCALES_SANDWICH

    print("=" * 70)
    print("Sandwich MLP + QAT + mHC (from scratch)")
    print(f"  dim={dim}, layers={n_layers}")
    print(f"  MLP scales: {mlp_scales}")
    print(f"  QAT: Ternary (1.58-bit), mode={'auto (adaptive)' if qat_start_step == -1 else f'from step {qat_start_step}'}")
    print(f"  mHC: FP32, init to 1.0, learned from scratch")
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

    # ── QAT Linear Layer ──────────────────────────────────────
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

            # mHC parameters - FP32, NOT quantized
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

    class SandwichQATTransformer(nn.Module):
        def __init__(self, vocab_size, dim, n_layers, n_heads, n_kv_heads, local_window, mlp_scales):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, dim)
            self.blocks = nn.ModuleList([
                SandwichQATBlock(dim, n_heads, n_kv_heads, local_window, i, n_layers, mlp_scales[i])
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

        def enable_qat(self):
            for module in self.modules():
                if isinstance(module, QATLinear):
                    module.enable_qat()

        def count_params(self):
            ternary = 0
            fp = 0
            for name, param in self.named_parameters():
                if isinstance(param, nn.Parameter):
                    # QATLinear weights are ternary, everything else is FP32
                    is_qat_weight = False
                    for module in self.modules():
                        if isinstance(module, QATLinear) and param is module.weight:
                            is_qat_weight = True
                            break
                    if is_qat_weight:
                        ternary += param.numel()
                    else:
                        fp += param.numel()
            return ternary, fp

    # Create model
    model = SandwichQATTransformer(
        vocab_size=VOCAB_SIZE,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        local_window=local_window,
        mlp_scales=mlp_scales,
    ).to(device)

    # Enable QAT from the start
    if qat_start_step == 0:
        model.enable_qat()
        print("[QAT] Enabled from step 0")

    total_params = sum(p.numel() for p in model.parameters())
    ternary_params, fp_params = model.count_params()
    global_layers = sum(1 for b in model.blocks if b.is_global)
    local_layers = n_layers - global_layers

    print(f"\nModel: {total_params/1e6:.2f}M params")
    print(f"  Ternary (1.58-bit): {ternary_params/1e6:.2f}M")
    print(f"  FP32: {fp_params/1e6:.2f}M")
    print(f"  Global layers: {global_layers}, Local layers: {local_layers}")

    # Estimated quantized size
    ternary_size_mb = (ternary_params * 1.58) / 8 / 1e6
    fp_size_mb = (fp_params * 32) / 8 / 1e6
    total_size_mb = ternary_size_mb + fp_size_mb
    print(f"\nEstimated quantized model size: {total_size_mb:.2f} MB")
    print(f"  Ternary: {ternary_size_mb:.2f} MB ({ternary_params/1e6:.2f}M params x 1.58 bit)")
    print(f"  FP32: {fp_size_mb:.2f} MB ({fp_params/1e6:.2f}M params x 32 bit)")
    marker = "<<< FITS 16MB!" if total_size_mb <= 16.0 else "<<< OVER 16MB!"
    print(f"  {marker}")

    print(f"\nLayer configuration:")
    print(f"  {'Layer':>5} {'Type':<12} {'MLP scale':>10} {'MLP hidden':>11}")
    print(f"  {'-'*42}")
    for block in model.blocks:
        print(f"  {block.layer_idx:>5} {block.attn_type:<12} {block.mlp_scale:>10.1f}x {block.mlp_hidden:>11}")

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
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (steps - warmup_steps)
        return min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + math.cos(math.pi * progress))

    def log_mhc_params(model, step):
        print(f"\n  --- mHC Parameters at step {step} ---")
        print(f"  {'Layer':>5} {'Type':<12} {'MLP':>5} {'a_attn':>7} {'b_attn':>7} {'a_mlp':>7} {'b_mlp':>7}")
        for block in model.blocks:
            p = block.get_mhc_params()
            print(f"  {p['layer']:>5} {p['attn_type']:<12} {p['mlp_scale']:>4.1f}x {p['alpha_attn']:>7.3f} {p['beta_attn']:>7.3f} {p['alpha_mlp']:>7.3f} {p['beta_mlp']:>7.3f}")
        print()

    # Track mHC evolution
    mhc_history = []

    print(f"\n[TRAIN] Starting training ({steps} steps, QAT mode={'auto (adaptive)' if qat_start_step == -1 else f'from step {qat_start_step}'})...\n")
    start_time = time.time()

    # Adaptive QAT switching state
    qat_enabled = (qat_start_step == 0)  # only True if explicitly from step 0
    ema_loss = None
    qat_actual_step = None  # record when QAT was actually enabled

    for step in range(1, steps + 1):
        # Enable QAT at specified fixed step
        if not qat_enabled and qat_start_step > 0 and step == qat_start_step:
            model.enable_qat()
            qat_enabled = True
            qat_actual_step = step
            print(f"\n[QAT] Enabled at fixed step {step}\n")

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * cosine_lr(step)

        x, y = get_batch('train')
        _, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Adaptive QAT switching: check loss convergence
        if not qat_enabled and qat_start_step == -1:
            current_loss = loss.item()
            if ema_loss is None:
                ema_loss = current_loss
            else:
                ema_loss_prev = ema_loss
                ema_loss = loss_ema_alpha * ema_loss + (1 - loss_ema_alpha) * current_loss

                # Only consider switching after minimum warmup
                if step >= qat_warmup_steps:
                    loss_rate = (ema_loss_prev - ema_loss) / ema_loss_prev
                    if loss_rate < qat_switch_threshold:
                        model.enable_qat()
                        qat_enabled = True
                        qat_actual_step = step
                        print(f"\n[QAT] Adaptive switch at step {step}!")
                        print(f"  EMA Loss: {ema_loss:.4f}")
                        print(f"  Loss rate: {loss_rate:.6f} < threshold {qat_switch_threshold}")
                        print(f"  FP32 trained for {step} steps, QAT will run for {steps - step} steps\n")

        if step % 500 == 0:
            elapsed = time.time() - start_time
            current_lr = optimizer.param_groups[0]['lr']
            qat_status = "QAT" if qat_enabled else "FP32"
            print(f"Step {step}/{steps} | Loss {loss.item():.4f} | LR {current_lr:.2e} | {qat_status} | Time {elapsed:.0f}s")

        # Log mHC params every 1000 steps
        if step % 1000 == 0:
            log_mhc_params(model, step)
            snapshot = {
                'step': step,
                'params': [block.get_mhc_params() for block in model.blocks]
            }
            mhc_history.append(snapshot)

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
    print("[RESULTS] Sandwich MLP + QAT + mHC")
    print("=" * 70)
    print(f"  Val Loss: {val_loss:.4f}")
    print(f"  Val BPB:  {val_bpb:.4f}")
    print(f"  Params:   {total_params/1e6:.2f}M (Ternary: {ternary_params/1e6:.2f}M, FP32: {fp_params/1e6:.2f}M)")
    print(f"  Quantized Size: {total_size_mb:.2f} MB")
    print(f"  vs Baseline (1.5025): {(val_bpb - 1.5025) / 1.5025 * 100:+.2f}%")
    print(f"  vs Sandwich FP32 (1.4833): {(val_bpb - 1.4833) / 1.4833 * 100:+.2f}%")
    print(f"  vs Alt-A mHC-scratch Uniform (1.4777): {(val_bpb - 1.4777) / 1.4777 * 100:+.2f}%")
    print("=" * 70)

    # Final mHC params
    print(f"\n[FINAL mHC PARAMETERS]")
    print(f"  {'Layer':>5} {'Type':<12} {'MLP':>5} {'a_attn':>7} {'b_attn':>7} {'a_mlp':>7} {'b_mlp':>7}")
    print("  " + "-" * 60)
    final_params = []
    for block in model.blocks:
        p = block.get_mhc_params()
        final_params.append(p)
        print(f"  {p['layer']:>5} {p['attn_type']:<12} {p['mlp_scale']:>4.1f}x {p['alpha_attn']:>7.3f} {p['beta_attn']:>7.3f} {p['alpha_mlp']:>7.3f} {p['beta_mlp']:>7.3f}")

    # Save results
    checkpoint_dir = "/data/checkpoints/sandwich_qat"
    os.makedirs(checkpoint_dir, exist_ok=True)

    results = {
        'style': 'sandwich_qat_adaptive',
        'config': {
            'dim': dim, 'n_layers': n_layers, 'n_heads': n_heads,
            'n_kv_heads': n_kv_heads, 'local_window': local_window,
            'steps': steps, 'lr': lr, 'batch_size': batch_size, 'seq_len': seq_len,
            'mlp_scales': mlp_scales, 'qat_start_step': qat_start_step,
            'qat_actual_step': qat_actual_step,
            'qat_switch_threshold': qat_switch_threshold,
            'loss_ema_alpha': loss_ema_alpha,
            'qat_warmup_steps': qat_warmup_steps,
            'qat_actual_step': qat_actual_step,
            'qat_switch_threshold': qat_switch_threshold,
            'loss_ema_alpha': loss_ema_alpha,
            'qat_warmup_steps': qat_warmup_steps,
        },
        'val_loss': val_loss,
        'val_bpb': val_bpb,
        'total_params': total_params,
        'ternary_params': ternary_params,
        'fp_params': fp_params,
        'quantized_size_mb': total_size_mb,
        'final_params': final_params,
        'mhc_history': mhc_history,
    }

    results_path = os.path.join(checkpoint_dir, f"sandwich_qat_bpb{val_bpb:.4f}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] Results: {results_path}")

    ckpt_path = os.path.join(checkpoint_dir, f"sandwich_qat_step{steps}.pt")
    torch.save({
        'step': steps,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_bpb': val_bpb,
        'final_params': final_params,
        'mhc_history': mhc_history,
        'config': results['config'],
    }, ckpt_path)
    print(f"[SAVED] Checkpoint: {ckpt_path}")

    data_volume.commit()

    return results


@app.local_entrypoint()
def main(
    qat_start_step: int = -1,  # -1 = auto, 0 = from start
    qat_warmup_steps: int = 500,
    qat_switch_threshold: float = 0.001,
):
    result = train_sandwich_qat.remote(
        qat_start_step=qat_start_step,
        qat_warmup_steps=qat_warmup_steps,
        qat_switch_threshold=qat_switch_threshold,
    )

    print(f"\n[FINAL] Sandwich MLP + QAT")
    print(f"  BPB: {result['val_bpb']:.4f}")
    print(f"  vs Baseline (1.5025): {(result['val_bpb'] - 1.5025) / 1.5025 * 100:+.2f}%")
    print(f"  vs Sandwich FP32 (1.4833): {(result['val_bpb'] - 1.4833) / 1.4833 * 100:+.2f}%")
    print(f"  QAT switched at step: {result.get('qat_actual_step', 'N/A')}")

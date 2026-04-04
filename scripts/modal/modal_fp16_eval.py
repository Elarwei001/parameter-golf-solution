"""
Evaluate Sandwich QAT model in FP16 (post-training conversion).

No retraining — just load checkpoint, convert to FP16, run validation.

Usage:
    modal run scripts/modal/modal_fp16_eval.py
"""
import modal
import os
import math
import json

app = modal.App("sandwich-qat-fp16-eval")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(["torch==2.5.1", "numpy"])
)

data_volume = modal.Volume.from_name("parameter-golf-data", create_if_missing=True)

MLP_SCALES_SANDWICH = [3.0]*4 + [1.2]*13 + [3.0]*3


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={"/data": data_volume},
    timeout=600,
)
def eval_fp16():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np

    device = torch.device("cuda")
    VOCAB_SIZE = 8192
    DATA_DIR = "/data/datasets/fineweb10B_sp8192"
    HEADER_SIZE = 256 * 4
    BYTES_PER_TOKEN = 3.67

    # ── Model classes (same as training) ──
    class TernaryQuantize(torch.autograd.Function):
        @staticmethod
        def forward(ctx, weight):
            scale = weight.abs().mean()
            w_quant = torch.clamp(torch.round(weight / scale), -1, 1)
            return w_quant * scale
        @staticmethod
        def backward(ctx, grad_output):
            return grad_output

    def ternary_quantize(weight):
        return TernaryQuantize.apply(weight)

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

    # ── Load checkpoint ──
    ckpt_path = "/data/checkpoints/sandwich_qat/sandwich_qat_step5000.pt"
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt['config']

    # Create model and load weights
    model = SandwichQATTransformer(
        vocab_size=VOCAB_SIZE,
        dim=config['dim'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        n_kv_heads=config['n_kv_heads'],
        local_window=config['local_window'],
        mlp_scales=MLP_SCALES_SANDWICH,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.enable_qat()  # QAT was enabled during training
    model.eval()

    print(f"\nFP32 checkpoint loaded. Original BPB: {ckpt.get('val_bpb', 'N/A')}")

    # ── FP32 validation ──
    # Load val data
    val_files = sorted([f for f in os.listdir(DATA_DIR) if 'val' in f])
    val_data = []
    for f in val_files:
        with open(os.path.join(DATA_DIR, f), 'rb') as fp:
            fp.seek(HEADER_SIZE)
            data = np.frombuffer(fp.read(), dtype=np.uint16)
        val_data.append(data)
    val_data = np.concatenate(val_data)
    val_tokens = torch.from_numpy(val_data.astype(np.int64))

    batch_size = 64
    seq_len = 256

    def evaluate(model, label=""):
        model.eval()
        val_losses = []
        with torch.no_grad():
            for _ in range(200):
                max_start = len(val_tokens) - seq_len - 1
                ix = torch.randint(0, max_start, (batch_size,))
                x = torch.stack([val_tokens[i:i+seq_len] for i in ix]).long().to(device)
                y = torch.stack([val_tokens[i+1:i+seq_len+1] for i in ix]).long().to(device)
                _, loss = model(x, y)
                val_losses.append(loss.item())
        val_loss = sum(val_losses) / len(val_losses)
        val_bpb = (val_loss / math.log(2)) * (1.0 / BYTES_PER_TOKEN)
        print(f"[{label}] Val Loss: {val_loss:.4f} | Val BPB: {val_bpb:.4f}")
        return val_loss, val_bpb

    # FP32 eval
    fp32_loss, fp32_bpb = evaluate(model, "FP32 (ternary QAT)")

    # ── Convert to FP16 ──
    print("\nConverting to FP16...")
    model_fp16 = model.half()
    fp16_loss, fp16_bpb = evaluate(model_fp16, "FP16 (ternary QAT)")

    # ── Results ──
    print("\n" + "=" * 60)
    print("FP16 Conversion Results")
    print("=" * 60)
    print(f"  FP32 Val Loss: {fp32_loss:.4f} | BPB: {fp32_bpb:.4f}")
    print(f"  FP16 Val Loss: {fp16_loss:.4f} | BPB: {fp16_bpb:.4f}")
    print(f"  Delta BPB:    {fp16_bpb - fp32_bpb:+.4f} ({(fp16_bpb - fp32_bpb)/fp32_bpb*100:+.2f}%)")

    # Model sizes
    fp32_size = sum(p.numel() * 4 for p in model.parameters()) / 1e6
    fp16_size = sum(p.numel() * 2 for p in model.parameters()) / 1e6
    print(f"\n  FP32 model size: {fp32_size:.2f} MB")
    print(f"  FP16 model size: {fp16_size:.2f} MB")
    print("=" * 60)

    # Save results
    results = {
        'experiment': 'fp16_conversion',
        'source_checkpoint': ckpt_path,
        'fp32_val_loss': fp32_loss,
        'fp32_val_bpb': fp32_bpb,
        'fp16_val_loss': fp16_loss,
        'fp16_val_bpb': fp16_bpb,
        'delta_bpb': fp16_bpb - fp32_bpb,
        'fp32_model_size_mb': fp32_size,
        'fp16_model_size_mb': fp16_size,
    }
    results_path = "/data/checkpoints/sandwich_qat/fp16_eval_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    data_volume.commit()
    print(f"\n[SAVED] Results: {results_path}")

    return results


@app.local_entrypoint()
def main():
    result = eval_fp16.remote()
    print(f"\nFP16 conversion complete!")
    print(f"  FP32 BPB: {result['fp32_val_bpb']:.4f}")
    print(f"  FP16 BPB: {result['fp16_val_bpb']:.4f}")
    print(f"  Delta: {result['delta_bpb']:+.4f}")

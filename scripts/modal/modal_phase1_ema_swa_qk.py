"""
Phase 1.1: EMA+SWA+QK-Gain Optimization
=====================================

Experiment: Add low-cost high-impact optimizations to current best config
- Base: dim=448 Sandwich QAT (1.3805 BPB baseline)
- Add: EMA (decay=0.997) + SWA (every 50 steps) + QK-Gain (init=4.0)
- Target: 1.35-1.36 BPB (-1.5% improvement)

Run:
modal run --detach scripts/modal/modal_phase1_ema_swa_qk.py::train_phase1_ema_swa_qk
"""
import modal
import os
import math

app = modal.App("phase1-ema-swa-qk")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch==2.5.1",
        "numpy",
    ])
)

data_volume = modal.Volume.from_name("parameter-golf-data", create_if_missing=True)

# Optimal configuration from dim=448 experiments
MLP_SCALES_SANDWICH = [3.0]*2 + [1.2]*28  # Simplified based on mHC analysis

@app.function(
    image=image,
    gpu="A100-40GB", 
    volumes={"/data": data_volume},
    timeout=7200,
)
def train_phase1_ema_swa_qk(
    seed: int = 42,
    dim: int = 448,
    n_layers: int = 30,
    n_heads: int = 8,
    n_kv_heads: int = 4,
    local_window: int = 128,
    lr: float = 1e-3,
    batch_size: int = 64,
    seq_len: int = 256,
    steps: int = 40000,
    warmup_steps: int = 1000,
    # QAT config (current best)
    qat_start_step: int = 2000,
    qat_ramp_steps: int = 1000,
    # NEW: EMA config
    ema_decay: float = 0.997,
    # NEW: SWA config  
    swa_start_step: int = 2000,
    swa_update_freq: int = 50,
    # NEW: QK-Gain config
    qk_gain_init: float = 4.0,
):
    """Phase 1.1: EMA + SWA + QK-Gain optimization."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import time
    import json
    from copy import deepcopy

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    VOCAB_SIZE = 8192
    DATA_DIR = "/data/datasets/fineweb10B_sp8192"
    HEADER_SIZE = 256 * 4
    BYTES_PER_TOKEN = 3.67
    mlp_scales = MLP_SCALES_SANDWICH

    print("=" * 70)
    print("PHASE 1.1: EMA + SWA + QK-Gain Optimization")
    print("=" * 70)
    print(f"Base config: dim={dim}, layers={n_layers}")
    print(f"EMA decay: {ema_decay}")
    print(f"SWA: start={swa_start_step}, freq={swa_update_freq}")
    print(f"QK-Gain init: {qk_gain_init}")
    print("=" * 70)

    class QKGain(nn.Module):
        """Learnable Q/K gain scaling per layer."""
        def __init__(self, n_layers: int, init_value: float = 4.0):
            super().__init__()
            self.gains = nn.Parameter(torch.full((n_layers,), init_value))
        
        def forward(self, layer_idx: int) -> float:
            return self.gains[layer_idx]

    class RMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))
        
        def forward(self, x):
            rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            return x / rms * self.weight

    class RotaryEmbedding(nn.Module):
        def __init__(self, dim, max_seq_len=8192, base=10000.0):
            super().__init__()
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
            self.register_buffer("inv_freq", inv_freq)
            
            t = torch.arange(max_seq_len).float()
            freqs = torch.einsum("i,j->ij", t, inv_freq)
            emb = torch.cat([freqs, freqs], dim=-1)
            self.register_buffer("cos", emb.cos())
            self.register_buffer("sin", emb.sin())
        
        def forward(self, x, seq_len):
            return self.cos[:seq_len], self.sin[:seq_len]

    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def apply_rotary_pos_emb(q, k, cos, sin):
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

    class STE(torch.autograd.Function):
        """Straight Through Estimator for ternary quantization."""
        @staticmethod
        def forward(ctx, input):
            return torch.sign(input).clamp(-1, 1)
        
        @staticmethod
        def backward(ctx, grad_output):
            return grad_output

    def ternary_quantize(x):
        """Ternary quantization: {-1, 0, +1}."""
        return STE.apply(x)

    class QuantizedLinear(nn.Module):
        def __init__(self, in_features, out_features, bias=False):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
            self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
            
            # Quantization parameters
            self.quantized = False
            
        def forward(self, x):
            if self.quantized:
                w = ternary_quantize(self.weight)
            else:
                w = self.weight
            return F.linear(x, w, self.bias)
        
        def enable_quantization(self):
            self.quantized = True
        
        def disable_quantization(self):
            self.quantized = False

    class Attention(nn.Module):
        def __init__(self, dim, n_heads, n_kv_heads, is_local=False, local_window=128, layer_idx=0, qk_gain_module=None):
            super().__init__()
            self.n_heads = n_heads
            self.n_kv_heads = n_kv_heads
            self.head_dim = dim // n_heads
            self.n_rep = n_heads // n_kv_heads
            self.is_local = is_local
            self.local_window = local_window
            self.layer_idx = layer_idx
            self.qk_gain_module = qk_gain_module
            
            self.wq = QuantizedLinear(dim, n_heads * self.head_dim, bias=False)
            self.wk = QuantizedLinear(dim, n_kv_heads * self.head_dim, bias=False)
            self.wv = QuantizedLinear(dim, n_kv_heads * self.head_dim, bias=False)
            self.wo = QuantizedLinear(n_heads * self.head_dim, dim, bias=False)
        
        def forward(self, x, cos, sin):
            B, L, _ = x.shape
            
            q = self.wq(x).view(B, L, self.n_heads, self.head_dim)
            k = self.wk(x).view(B, L, self.n_kv_heads, self.head_dim)
            v = self.wv(x).view(B, L, self.n_kv_heads, self.head_dim)
            
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            q, k = apply_rotary_pos_emb(q, k, cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0))
            
            if self.n_rep > 1:
                k = k.repeat_interleave(self.n_rep, dim=1)
                v = v.repeat_interleave(self.n_rep, dim=1)
            
            # Apply QK-Gain
            qk_gain = 1.0
            if self.qk_gain_module is not None:
                qk_gain = self.qk_gain_module(self.layer_idx)
            
            scale = (self.head_dim ** -0.5) * qk_gain
            attn = (q @ k.transpose(-2, -1)) * scale
            
            if self.is_local:
                # Local attention with sliding window
                mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
                for i in range(L):
                    start = max(0, i - self.local_window + 1)
                    mask[i, start:i+1] = False
                attn = attn.masked_fill(mask, float('-inf'))
            else:
                # Global causal attention
                mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
                attn = attn.masked_fill(mask, float('-inf'))
            
            attn = F.softmax(attn, dim=-1)
            out = (attn @ v).transpose(1, 2).reshape(B, L, -1)
            return self.wo(out)

    class MLP(nn.Module):
        def __init__(self, dim, scale=4.0):
            super().__init__()
            hidden = int(dim * scale * 2 / 3)
            self.w1 = QuantizedLinear(dim, hidden, bias=False)
            self.w2 = QuantizedLinear(hidden, dim, bias=False)
            self.w3 = QuantizedLinear(dim, hidden, bias=False)
        
        def forward(self, x):
            return self.w2(F.silu(self.w1(x)) * self.w3(x))

    class mHC(nn.Module):
        """Multi-Head Coefficient: learnable residual scaling."""
        def __init__(self):
            super().__init__()
            self.alpha_attn = nn.Parameter(torch.ones(1))
            self.beta_attn = nn.Parameter(torch.ones(1))
            self.alpha_mlp = nn.Parameter(torch.ones(1))
            self.beta_mlp = nn.Parameter(torch.ones(1))

    class TransformerBlock(nn.Module):
        def __init__(self, dim, n_heads, n_kv_heads, mlp_scale, is_local=False, local_window=128, layer_idx=0, qk_gain_module=None):
            super().__init__()
            self.ln1 = RMSNorm(dim)
            self.attn = Attention(dim, n_heads, n_kv_heads, is_local, local_window, layer_idx, qk_gain_module)
            self.ln2 = RMSNorm(dim)
            self.mlp = MLP(dim, mlp_scale)
            self.mhc = mHC()
        
        def forward(self, x, cos, sin):
            # Attention with mHC
            x_norm = self.ln1(x)
            attn_out = self.attn(x_norm, cos, sin)
            x = self.mhc.alpha_attn * x + self.mhc.beta_attn * attn_out
            
            # MLP with mHC
            x_norm = self.ln2(x)
            mlp_out = self.mlp(x_norm)
            x = self.mhc.alpha_mlp * x + self.mhc.beta_mlp * mlp_out
            
            return x

    class SandwichGPT(nn.Module):
        def __init__(self, vocab_size, dim, n_layers, n_heads, n_kv_heads, mlp_scales, local_window=128, qk_gain_init=4.0):
            super().__init__()
            self.vocab_size = vocab_size
            self.dim = dim
            self.n_layers = n_layers
            
            self.tok_emb = nn.Embedding(vocab_size, dim)
            self.rope = RotaryEmbedding(dim // n_heads)
            
            # QK-Gain module
            self.qk_gain = QKGain(n_layers, qk_gain_init)
            
            self.blocks = nn.ModuleList([
                TransformerBlock(
                    dim, n_heads, n_kv_heads, 
                    mlp_scales[i],
                    is_local=(i % 2 == 1),  # Alternating: even=global, odd=local
                    local_window=local_window,
                    layer_idx=i,
                    qk_gain_module=self.qk_gain
                )
                for i in range(n_layers)
            ])
            
            self.ln_f = RMSNorm(dim)
        
        def forward(self, x):
            B, L = x.shape
            h = self.tok_emb(x)
            cos, sin = self.rope(h, L)
            
            for block in self.blocks:
                h = block(h, cos, sin)
            
            h = self.ln_f(h)
            logits = h @ self.tok_emb.weight.T
            return logits
        
        def get_quantizable_modules(self):
            """Return all QuantizedLinear modules for QAT control."""
            modules = []
            for module in self.modules():
                if isinstance(module, QuantizedLinear):
                    modules.append(module)
            return modules
        
        def enable_quantization(self):
            for module in self.get_quantizable_modules():
                module.enable_quantization()
        
        def disable_quantization(self):
            for module in self.get_quantizable_modules():
                module.disable_quantization()
        
        def count_parameters(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        def estimate_size_mb(self, bits=16):
            n_params = self.count_parameters()
            return (n_params * bits / 8) / (1024 * 1024)

    def get_batch(data_dir, file_id, seq_len, batch_size, header_size=HEADER_SIZE):
        """Load batch from data file."""
        file_path = os.path.join(data_dir, f"fineweb_train_{file_id:06d}.bin")
        
        with open(file_path, 'rb') as f:
            f.seek(header_size)
            data = np.frombuffer(f.read(), dtype=np.uint16)
        
        data = torch.from_numpy(data).long()
        
        batch = []
        for _ in range(batch_size):
            start = torch.randint(0, len(data) - seq_len, (1,)).item()
            batch.append(data[start:start + seq_len])
        
        return torch.stack(batch)

    def update_ema(ema_model, model, decay):
        """Update EMA model weights."""
        with torch.no_grad():
            for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                ema_param.lerp_(param, 1 - decay)

    def update_swa(swa_model, model, num_updates):
        """Update SWA (Stochastic Weight Averaging) model."""
        with torch.no_grad():
            for swa_param, param in zip(swa_model.parameters(), model.parameters()):
                swa_param.data = (swa_param.data * num_updates + param.data) / (num_updates + 1)

    # Initialize model
    model = SandwichGPT(
        vocab_size=VOCAB_SIZE,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        mlp_scales=mlp_scales,
        local_window=local_window,
        qk_gain_init=qk_gain_init
    ).to(device)

    # Initialize EMA model
    ema_model = deepcopy(model)
    for param in ema_model.parameters():
        param.requires_grad = False
    
    # Initialize SWA model
    swa_model = deepcopy(model)
    for param in swa_model.parameters():
        param.requires_grad = False
    swa_updates = 0

    n_params = model.count_parameters()
    print(f"Model parameters: {n_params:,}")
    print(f"Estimated FP16 size: {model.estimate_size_mb(16):.2f} MB")
    print(f"Estimated 3-bit size: {model.estimate_size_mb(3):.2f} MB")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))

    # Training loop
    model.train()
    start_time = time.time()
    
    for step in range(1, steps + 1):
        # Learning rate schedule
        if step <= warmup_steps:
            current_lr = lr * step / warmup_steps
        else:
            current_lr = lr * 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (steps - warmup_steps)))
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        # QAT control
        if qat_start_step > 0 and step >= qat_start_step:
            if qat_ramp_steps > 0 and step < qat_start_step + qat_ramp_steps:
                # Ramp mode: partially quantize
                ramp_progress = (step - qat_start_step) / qat_ramp_steps
                if torch.rand(1).item() < ramp_progress:
                    model.enable_quantization()
                else:
                    model.disable_quantization()
            else:
                # Full QAT mode
                model.enable_quantization()
        else:
            model.disable_quantization()

        # Get batch
        file_id = step % 10
        batch = get_batch(DATA_DIR, file_id, seq_len, batch_size).to(device)
        
        # Forward pass
        logits = model(batch[:, :-1])
        targets = batch[:, 1:]
        
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Update EMA
        update_ema(ema_model, model, ema_decay)
        
        # Update SWA
        if step >= swa_start_step and step % swa_update_freq == 0:
            update_swa(swa_model, model, swa_updates)
            swa_updates += 1

        # Logging
        if step % 500 == 0:
            elapsed = time.time() - start_time
            bpb = loss.item() / math.log(2) / BYTES_PER_TOKEN
            
            print(f"Step {step:5d}/{steps} | Loss {loss.item():.4f} | BPB {bpb:.4f} | LR {current_lr:.2e} | Time {elapsed:.0f}s")
            
            if step % 2000 == 0:
                # Log QK gains
                with torch.no_grad():
                    gains = model.qk_gain.gains.cpu().numpy()
                    print(f"  QK gains: min={gains.min():.3f}, max={gains.max():.3f}, mean={gains.mean():.3f}")

    # Final evaluation on validation data
    print("\n" + "="*50)
    print("FINAL EVALUATION")
    print("="*50)
    
    models_to_eval = [
        ("Main Model", model),
        ("EMA Model", ema_model),
        ("SWA Model", swa_model if swa_updates > 0 else None)
    ]
    
    results = {}
    
    for model_name, eval_model in models_to_eval:
        if eval_model is None:
            continue
            
        eval_model.eval()
        eval_model.disable_quantization()  # Use FP32 for evaluation
        
        total_loss = 0.0
        total_tokens = 0
        n_batches = 20
        
        with torch.no_grad():
            for i in range(n_batches):
                file_id = (i + 10) % 20  # Different files for validation
                batch = get_batch(DATA_DIR, file_id, seq_len, batch_size).to(device)
                
                logits = eval_model(batch[:, :-1])
                targets = batch[:, 1:]
                
                loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
                
                batch_tokens = targets.numel()
                total_loss += loss.item() * batch_tokens
                total_tokens += batch_tokens
        
        avg_loss = total_loss / total_tokens
        bpb = avg_loss / math.log(2) / BYTES_PER_TOKEN
        
        print(f"{model_name}: Val Loss {avg_loss:.4f} | Val BPB {bpb:.4f}")
        results[model_name.lower().replace(" ", "_")] = {
            "val_loss": avg_loss,
            "val_bpb": bpb
        }

    # Get best result
    best_bpb = min(results[k]["val_bpb"] for k in results.keys())
    best_model = min(results.keys(), key=lambda k: results[k]["val_bpb"])
    
    print(f"\nBEST: {best_model} with BPB = {best_bpb:.4f}")

    # Save results
    os.makedirs("/data/checkpoints/phase1_ema_swa_qk", exist_ok=True)
    
    final_results = {
        "experiment": "phase1_ema_swa_qk",
        "config": {
            "dim": dim,
            "n_layers": n_layers,
            "steps": steps,
            "ema_decay": ema_decay,
            "swa_start_step": swa_start_step,
            "swa_update_freq": swa_update_freq,
            "qk_gain_init": qk_gain_init,
            "qat_start_step": qat_start_step,
            "qat_ramp_steps": qat_ramp_steps,
        },
        "results": results,
        "best_bpb": best_bpb,
        "best_model": best_model,
        "parameters": n_params,
        "training_time_seconds": time.time() - start_time
    }
    
    with open(f"/data/checkpoints/phase1_ema_swa_qk/results_bpb{best_bpb:.4f}.json", "w") as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nResults saved: BPB {best_bpb:.4f}")
    return final_results

if __name__ == "__main__":
    # Local test
    print("Phase 1.1: EMA + SWA + QK-Gain")
    print("Run with: modal run --detach scripts/modal/modal_phase1_ema_swa_qk.py::train_phase1_ema_swa_qk")
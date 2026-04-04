"""
Alternating Attention Experiment
================================

## Technique
- Even layers (0,2,4,...) + last layer: Global full attention
- Odd layers (1,3,5,...): Local sliding-window attention (window=128)
- Inspired by: Gemma 3n / gamma4's suggestion

## Hypothesis
Different layers learn different patterns: Local layers focus on local context (n-gram level),
Global layers handle global semantics. This layer specialization may be more efficient than
uniform Global attention across all layers.

## Experiment Design
- Alt-A: dim=384 (same as baseline), verify Alternating doesn't hurt
- Alt-A + mHC: dim=384 + mHC parameter initialization
- Alt-B: dim=448 (larger), verify larger dim improves performance with Alternating
- Alt-B + mHC: dim=448 + mHC parameter initialization (requires Alternating-specific mHC)

## Run
```bash
# Alt-A: same as baseline config
modal run --detach scripts/modal/modal_alternating_attn.py

# Alt-A + mHC:
modal run --detach scripts/modal/modal_alternating_attn.py -- --mhc

# Alt-B: larger dim
modal run --detach scripts/modal/modal_alternating_attn.py -- --dim 448

# Alt-B + mHC:
modal run --detach scripts/modal/modal_alternating_attn.py -- --dim 448 --mhc

# Resume from checkpoint (continue training to 10000 steps):
modal run --detach scripts/modal/modal_alternating_attn.py -- --steps 10000 --resume /data/checkpoints/alternating_attn/alt_Vanilla_dim384_L20_step5000.pt
```

## Baseline Comparison
- mHC v2, 20 layers, dim=384: BPB = 1.5025
- Script: https://github.com/Elarwei001/parameter-golf-solution/blob/master/scripts/modal/modal_mhc_v2_deep.py
"""
import modal
import os
import math

app = modal.App("alternating-attn-experiment")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch==2.5.1",
        "numpy",
    ])
)

data_volume = modal.Volume.from_name("parameter-golf-data", create_if_missing=True)


# mHC parameters learned from 20-layer experiment
MHC_PARAMS_20L = [
    {"alpha_attn": 1.120, "beta_attn": 0.278, "alpha_mlp": 1.096, "beta_mlp": 0.559},  # Layer 0
    {"alpha_attn": 1.066, "beta_attn": 0.383, "alpha_mlp": 1.051, "beta_mlp": 0.694},  # Layer 1
    {"alpha_attn": 1.031, "beta_attn": 0.376, "alpha_mlp": 1.014, "beta_mlp": 0.734},  # Layer 2
    {"alpha_attn": 0.995, "beta_attn": 0.873, "alpha_mlp": 1.019, "beta_mlp": 0.843},  # Layer 3
    {"alpha_attn": 1.014, "beta_attn": 0.616, "alpha_mlp": 1.006, "beta_mlp": 0.864},  # Layer 4
    {"alpha_attn": 1.003, "beta_attn": 0.528, "alpha_mlp": 0.990, "beta_mlp": 0.851},  # Layer 5
    {"alpha_attn": 0.988, "beta_attn": 0.519, "alpha_mlp": 0.976, "beta_mlp": 0.822},  # Layer 6
    {"alpha_attn": 0.973, "beta_attn": 0.533, "alpha_mlp": 0.963, "beta_mlp": 0.788},  # Layer 7
    {"alpha_attn": 0.959, "beta_attn": 0.662, "alpha_mlp": 0.952, "beta_mlp": 0.746},  # Layer 8
    {"alpha_attn": 0.944, "beta_attn": 0.930, "alpha_mlp": 0.956, "beta_mlp": 0.784},  # Layer 9
    {"alpha_attn": 0.952, "beta_attn": 0.690, "alpha_mlp": 0.948, "beta_mlp": 0.743},  # Layer 10
    {"alpha_attn": 0.946, "beta_attn": 0.795, "alpha_mlp": 0.950, "beta_mlp": 0.702},  # Layer 11
    {"alpha_attn": 0.943, "beta_attn": 0.757, "alpha_mlp": 0.945, "beta_mlp": 0.678},  # Layer 12
    {"alpha_attn": 0.940, "beta_attn": 0.793, "alpha_mlp": 0.942, "beta_mlp": 0.680},  # Layer 13
    {"alpha_attn": 0.935, "beta_attn": 0.760, "alpha_mlp": 0.933, "beta_mlp": 0.697},  # Layer 14
    {"alpha_attn": 0.927, "beta_attn": 0.733, "alpha_mlp": 0.921, "beta_mlp": 0.678},  # Layer 15
    {"alpha_attn": 0.914, "beta_attn": 0.761, "alpha_mlp": 0.902, "beta_mlp": 0.677},  # Layer 16
    {"alpha_attn": 0.886, "beta_attn": 0.854, "alpha_mlp": 0.891, "beta_mlp": 0.615},  # Layer 17
    {"alpha_attn": 0.867, "beta_attn": 0.818, "alpha_mlp": 0.874, "beta_mlp": 0.583},  # Layer 18
    {"alpha_attn": 0.825, "beta_attn": 1.049, "alpha_mlp": 0.922, "beta_mlp": 0.627},  # Layer 19
]


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={"/data": data_volume},
    timeout=7200,
)
def train_alternating(
    seed: int = 42,
    # Architecture params
    dim: int = 384,
    n_layers: int = 20,
    n_heads: int = 8,
    n_kv_heads: int = 4,
    local_window: int = 128,  # Local attention window size
    # Mode selection
    use_mhc: bool = False,  # Whether to use learned mHC parameters
    # Training params
    lr: float = 1e-3,
    batch_size: int = 64,
    seq_len: int = 256,
    steps: int = 5000,
    # Resume from checkpoint
    resume_from: str = None,  # Path to checkpoint file to resume from
):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import time
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    VOCAB_SIZE = 8192
    DATA_DIR = "/data/datasets/fineweb10B_sp8192"
    HEADER_SIZE = 256 * 4
    
    mode_str = "mHC" if use_mhc else "Vanilla"
    print("="*70)
    print(f"Alternating Attention Experiment")
    print(f"  Layers: {n_layers}, dim: {dim}, mode: {mode_str}")
    print(f"  Odd layers: Local (window={local_window}), Even layers + last: Global")
    print("="*70)
    
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
        """Attention with Local/Global switching"""
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
            
            # XSA: mask diagonal (same as baseline)
            diag_mask = torch.eye(T, device=x.device, dtype=torch.bool)
            scores = scores.masked_fill(diag_mask, 0.0)
            
            # Causal mask (always)
            rows = torch.arange(T, device=x.device).view(-1, 1)
            cols = torch.arange(T, device=x.device).view(1, -1)
            causal_mask = cols > rows
            
            if self.is_global:
                # Global: causal mask only
                mask = causal_mask
            else:
                # Local: causal + window mask
                window_mask = (rows - cols) > self.local_window
                mask = causal_mask | window_mask
            
            # NOTE: We compute full scores then mask, rather than sparse computation.
            # For seq_len=256, this is efficient enough. For longer sequences,
            # consider FlashAttention or xformers for true sparse computation.
            scores = scores.masked_fill(mask, float('-inf'))
            attn = F.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)
            out = out.transpose(1, 2).contiguous().view(B, T, C)
            return self.wo(out)
    
    class MLP(nn.Module):
        def __init__(self, dim, hidden_dim=None):
            super().__init__()
            hidden_dim = hidden_dim or dim * 3
            self.w1 = nn.Linear(dim, hidden_dim, bias=False)
            self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        
        def forward(self, x):
            h = self.w1(x)
            h = F.leaky_relu(h, 0.5) ** 2
            return self.w2(h)
    
    class AlternatingBlock(nn.Module):
        def __init__(self, dim, n_heads, n_kv_heads, local_window, layer_idx, n_layers, use_mhc=False):
            super().__init__()
            self.layer_idx = layer_idx
            
            # Decide Global or Local
            # Even layers (0, 2, 4, ...) and last layer: Global
            # Odd layers (1, 3, 5, ...): Local
            is_last = (layer_idx == n_layers - 1)
            is_global = (layer_idx % 2 == 0) or is_last
            self.is_global = is_global
            
            self.attn = AlternatingAttention(dim, n_heads, n_kv_heads, local_window, is_global)
            self.mlp = MLP(dim)
            self.ln1 = RMSNorm(dim)
            self.ln2 = RMSNorm(dim)
            
            # mHC parameters
            self.use_mhc = use_mhc
            if use_mhc and layer_idx < len(MHC_PARAMS_20L):
                params = MHC_PARAMS_20L[layer_idx]
                self.alpha_attn = nn.Parameter(torch.tensor([params["alpha_attn"]]))
                self.beta_attn = nn.Parameter(torch.tensor([params["beta_attn"]]))
                self.alpha_mlp = nn.Parameter(torch.tensor([params["alpha_mlp"]]))
                self.beta_mlp = nn.Parameter(torch.tensor([params["beta_mlp"]]))
            else:
                self.alpha_attn = nn.Parameter(torch.ones(1))
                self.beta_attn = nn.Parameter(torch.ones(1))
                self.alpha_mlp = nn.Parameter(torch.ones(1))
                self.beta_mlp = nn.Parameter(torch.ones(1))
        
        def forward(self, x):
            if self.use_mhc:
                attn_out = self.attn(self.ln1(x))
                x = self.alpha_attn * x + self.beta_attn * attn_out
                
                mlp_out = self.mlp(self.ln2(x))
                x = self.alpha_mlp * x + self.beta_mlp * mlp_out
            else:
                # Vanilla: standard residual
                x = x + self.attn(self.ln1(x))
                x = x + self.mlp(self.ln2(x))
            
            return x
    
    class AlternatingTransformer(nn.Module):
        def __init__(self, vocab_size, dim, n_layers, n_heads, n_kv_heads, local_window, use_mhc):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, dim)
            self.blocks = nn.ModuleList([
                AlternatingBlock(dim, n_heads, n_kv_heads, local_window, i, n_layers, use_mhc)
                for i in range(n_layers)
            ])
            self.ln_f = RMSNorm(dim)
            self.lm_head = nn.Linear(dim, vocab_size, bias=False)
            self.embed.weight = self.lm_head.weight  # Weight tying
        
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
    model = AlternatingTransformer(
        vocab_size=VOCAB_SIZE,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        local_window=local_window,
        use_mhc=use_mhc,
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    
    # Count Global/Local layers
    global_layers = sum(1 for b in model.blocks if b.is_global)
    local_layers = n_layers - global_layers
    
    print(f"\nModel: {total_params/1e6:.2f}M params")
    print(f"  Global layers: {global_layers}, Local layers: {local_layers}")
    print(f"  Using mHC: {use_mhc}")
    
    # Print layer configuration
    print("\nLayer configuration:")
    for i, block in enumerate(model.blocks):
        attn_type = "Global" if block.is_global else f"Local(w={local_window})"
        print(f"  Layer {i:2d}: {attn_type}")
    
    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    
    # Resume from checkpoint if specified
    start_step = 0
    if resume_from and os.path.exists(resume_from):
        print(f"\n[RESUME] Loading checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint['step']
        print(f"   Resuming from step {start_step}")
        print(f"   Previous loss: {checkpoint['loss']:.4f}")
    
    def get_batch(split):
        data = train_tokens if split == 'train' else val_tokens
        max_start = len(data) - seq_len - 1
        ix = torch.randint(0, max_start, (batch_size,))
        x = torch.stack([data[i:i+seq_len] for i in ix]).long().to(device)
        y = torch.stack([data[i+1:i+seq_len+1] for i in ix]).long().to(device)
        return x, y
    
    min_lr_ratio = 0.1  # Minimum LR = 10% of initial LR
    
    def cosine_lr(step):
        if step < 200:
            return step / 200  # Warmup
        progress = (step - 200) / (steps - 200)
        # Cosine decay from 1.0 to min_lr_ratio
        return min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + math.cos(math.pi * progress))
    
    # Checkpoint directory
    checkpoint_dir = "/data/checkpoints/alternating_attn"
    os.makedirs(checkpoint_dir, exist_ok=True)
    run_name = f"alt_{mode_str}_dim{dim}_L{n_layers}"
    
    print(f"\n[TRAIN] Starting training ({steps} steps)...")
    print(f"   Checkpoints will be saved to: {checkpoint_dir}/{run_name}_*.pt\n")
    start_time = time.time()
    
    for step in range(start_step + 1, steps + 1):
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
        
        # Save checkpoint every 5000 steps
        if step % 5000 == 0:
            checkpoint_path = f"{checkpoint_dir}/{run_name}_step{step}.pt"
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
                'config': {
                    'dim': dim,
                    'n_layers': n_layers,
                    'n_heads': n_heads,
                    'n_kv_heads': n_kv_heads,
                    'local_window': local_window,
                    'use_mhc': use_mhc,
                }
            }, checkpoint_path)
            print(f"   [CHECKPOINT] Saved: {checkpoint_path}")
            data_volume.commit()  # Persist to volume
    
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
    val_bpb = val_loss / math.log(2)
    
    print("\n" + "="*70)
    print("[RESULTS]")
    print("="*70)
    print(f"  Mode: {mode_str}")
    print(f"  Layers: {n_layers} (Global: {global_layers}, Local: {local_layers})")
    print(f"  Params: {total_params/1e6:.2f}M")
    print(f"  Val Loss: {val_loss:.4f}")
    print(f"  Val BPB:  {val_bpb:.4f}")
    print("="*70)
    
    return {
        "mode": mode_str,
        "n_layers": n_layers,
        "global_layers": global_layers,
        "local_layers": local_layers,
        "params_m": total_params / 1e6,
        "val_loss": val_loss,
        "val_bpb": val_bpb,
    }


@app.local_entrypoint()
def main(
    mhc: bool = False,
    dim: int = 384,
    n_layers: int = 20,
    local_window: int = 128,
    steps: int = 5000,
    resume: str = None,
):
    """Run Alternating Attention experiment.
    
    Args:
        mhc: Use learned mHC parameters
        dim: Model dimension (default: 384)
        n_layers: Number of layers (default: 20)
        local_window: Local attention window (default: 128)
        steps: Training steps (default: 5000)
        resume: Path to checkpoint to resume from
    """
    mode = "mHC" if mhc else "Vanilla"
    print(f"\n[START] Alternating Attention Experiment")
    print(f"   Mode: {mode}, dim={dim}, layers={n_layers}, window={local_window}")
    
    result = train_alternating.remote(
        n_layers=n_layers,
        dim=dim,
        n_heads=8,
        n_kv_heads=4,
        local_window=local_window,
        use_mhc=mhc,
        seed=42,
        steps=steps,
        resume_from=resume,
    )
    
    print(f"\n[FINAL] BPB: {result['val_bpb']:.4f}")
    print(f"[FINAL] vs Baseline (1.5025): {(result['val_bpb'] - 1.5025) / 1.5025 * 100:+.2f}%")
    print(f"[FINAL] Global layers: {result['global_layers']}, Local layers: {result['local_layers']}")

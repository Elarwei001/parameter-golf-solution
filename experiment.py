"""
AutoResearch-style experiment runner for Parameter Golf.
Runs experiments on Modal and tracks results.
"""
import modal
import os
import json
import time
from datetime import datetime

app = modal.App("parameter-golf-experiments")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["git"])
    .pip_install([
        "torch>=2.0",
        "numpy",
        "sentencepiece",
        "tqdm",
    ])
)

data_volume = modal.Volume.from_name("parameter-golf-data", create_if_missing=True)
output_volume = modal.Volume.from_name("parameter-golf-output", create_if_missing=True)

CODE_REPO = "https://github.com/Elarwei001/parameter-golf-solution.git"


def setup_code():
    """Clone our code repo"""
    import subprocess
    if not os.path.exists("/root/project"):
        subprocess.run(["git", "clone", CODE_REPO, "/root/project"], check=True)
    import sys
    sys.path.insert(0, "/root/project")


@app.function(
    image=image,
    gpu="A100",
    volumes={
        "/data": data_volume,
        "/output": output_volume,
    },
    timeout=7200,  # 2 hour max
)
def run_experiment(
    # Model config
    model_type: str = "latent",  # "latent", "standard", or "shared"
    latent_dim: int = 64,
    n_layers: int = 8,
    n_heads: int = 8,
    n_kv_heads: int = 4,
    mlp_ratio: float = 4.0,
    embed_dim: int = 256,
    vocab_size: int = 1024,
    # Weight-sharing params (model_type="shared")
    shared_layers: int = 3,   # unique layers
    n_passes: int = 3,        # cycles per forward pass
    # Training config
    seq_length: int = 1024,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    sigreg_weight: float = 0.1,
    max_steps: int = 1000000,
    max_seconds: int = 120,
    # Optimizer
    optimizer_type: str = "adamw",  # "adamw" or "muon"
    muon_momentum: float = 0.95,
    # Checkpoint
    resume_from: str = "",  # checkpoint name to resume from
    save_checkpoint: bool = True,  # save checkpoint at end
    # Eval
    sliding_window_eval: bool = True,   # use sliding window evaluation
    eval_stride: int = 64,              # sliding window stride
    eval_batch_seqs: int = 128,         # windows per forward pass
    # Experiment
    experiment_name: str = "default",
):
    """Run a single experiment with given hyperparameters"""
    import torch
    import torch.nn.functional as F
    import numpy as np
    import math
    
    setup_code()
    
    from configs.base import Config, ModelConfig, TrainingConfig
    from models.latent_lm import LatentLM
    from models.standard_gpt import StandardGPT
    from models.mamba_lm import MambaLM, MambaConfig
    
    print(f"🚀 {experiment_name}: {model_type}, dim={embed_dim}, layers={n_layers}")
    
    # Create model based on type
    device = torch.device("cuda")
    
    if model_type == "mamba":
        mamba_config = MambaConfig(
            vocab_size=vocab_size,
            d_model=embed_dim,
            n_layer=n_layers,
            d_state=64,
            expand=2,
            headdim=32,
            max_seq_len=seq_length,
        )
        model = MambaLM(mamba_config).to(device)
    elif model_type == "standard":
        model = StandardGPT(
            vocab_size=vocab_size,
            dim=embed_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            mlp_mult=int(mlp_ratio),
            max_seq_len=seq_length,
            tie_embeddings=True,
        ).to(device)
    elif model_type == "shared":
        # Universal Transformer weight-sharing
        model = StandardGPT(
            vocab_size=vocab_size,
            dim=embed_dim,
            n_layers=n_layers,   # ignored when shared_layers > 0
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            mlp_mult=int(mlp_ratio),
            max_seq_len=seq_length,
            tie_embeddings=True,
            shared_layers=shared_layers,
            n_passes=n_passes,
        ).to(device)
    else:
        config = Config(
            model=ModelConfig(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                latent_dim=latent_dim,
                n_layers=n_layers,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
            ),
        )
        model = LatentLM(config.model).to(device)
    
    n_params = model.count_parameters()
    size_3bit = model.estimate_size(3)
    
    print(f"📊 {n_params/1e6:.1f}M params, {size_3bit:.1f}MB @ 3-bit")
    
    if size_3bit > 16:
        print(f"⚠️  WARNING: Model too large for 16MB limit!")
        return {
            "status": "error",
            "error": f"Model size {size_3bit:.2f}MB exceeds 16MB limit",
            "params": n_params,
            "size_3bit_mb": size_3bit,
        }
    
    # Load data (silent)
    data_path = "/data/datasets/fineweb10B_sp1024"
    train_files = sorted([f for f in os.listdir(data_path) if "train" in f])
    
    if not train_files:
        return {"status": "error", "error": "No training files found"}
    
    # Load ALL training shards (skip 512-uint16 header in each shard)
    HEADER_TOKENS = 512  # 256 int32 header = 512 uint16 positions
    all_data = []
    total_tokens = 0
    for f in train_files:
        train_path = os.path.join(data_path, f)
        shard_raw = np.memmap(train_path, dtype=np.uint16, mode='r')
        shard_data = shard_raw[HEADER_TOKENS:]  # skip header
        all_data.append(shard_data)
        total_tokens += len(shard_data)
    
    # Concatenate (use first shard for simplicity if memory constrained)
    if len(all_data) == 1:
        train_data = all_data[0]
    else:
        # Randomly sample from shards during training
        train_data = all_data  # Keep as list
    
    print(f"📚 {total_tokens/1e6:.0f}M tokens loaded")
    
    # Load validation data for sliding window eval
    val_files = sorted([f for f in os.listdir(data_path) if "val" in f])
    if val_files:
        val_path = os.path.join(data_path, val_files[0])
        val_raw = np.memmap(val_path, dtype=np.uint16, mode='r')
        val_data = val_raw[HEADER_TOKENS:]  # skip header
        print(f"📖 Val: {len(val_data)/1e6:.1f}M tokens (skipped {HEADER_TOKENS} header tokens)")
    else:
        val_data = None
        print(f"⚠️  No val file found, will use train data for eval")
    
    # Optimizer (silent setup)
    if optimizer_type == "muon":
        try:
            muon_params = [p for p in model.parameters() if p.dim() == 2]
            adamw_params = [p for p in model.parameters() if p.dim() != 2]
            from torch.optim import Muon, AdamW
            optimizer = Muon(muon_params, lr=learning_rate, momentum=muon_momentum)
            if adamw_params:
                adamw_opt = AdamW(adamw_params, lr=learning_rate * 0.1, betas=(0.9, 0.95), weight_decay=0.01)
                optimizer = (optimizer, adamw_opt)
            print(f"⚡ Muon optimizer")
        except (ImportError, AttributeError):
            optimizer_type = "adamw"
    
    if optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.01)
        print(f"⚡ AdamW optimizer")
    
    # Resume from checkpoint if specified
    start_step = 0
    total_time_before = 0
    
    if resume_from:
        checkpoint_path = f"/output/checkpoints/{resume_from}.pt"
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model'])
            if isinstance(optimizer, tuple):
                optimizer[0].load_state_dict(checkpoint['optimizer'])
                if 'optimizer2' in checkpoint:
                    optimizer[1].load_state_dict(checkpoint['optimizer2'])
            else:
                optimizer.load_state_dict(checkpoint['optimizer'])
            start_step = checkpoint['step']
            total_time_before = checkpoint.get('total_time', 0)
            prev_bpb = checkpoint.get('bpb', 0)
            print(f"📂 Resumed: step {start_step}, {total_time_before/60:.0f}min, BPB {prev_bpb:.2f}")
        else:
            print(f"⚠️ Checkpoint not found: {checkpoint_path}")
    
    # Training loop
    print(f"🏃 Training for {max_seconds/60:.0f}min...")
    start_time = time.time()
    last_progress_time = 0  # Track last progress log time
    progress_interval = 900  # Log every 15 minutes (to file, not stdout)
    
    # Progress file (local, not printed to LLM)
    progress_file = f"/output/progress/{experiment_name}.log"
    os.makedirs("/output/progress", exist_ok=True)
    
    model.train()
    losses = []
    
    for step in range(start_step, max_steps):
        # Check time limit
        elapsed = time.time() - start_time
        if elapsed >= max_seconds:
            break
        
        # Sample random batch
        if isinstance(train_data, list):
            # Multiple shards - sample from random shard
            shard_idx = np.random.randint(0, len(train_data))
            shard = train_data[shard_idx]
            batch_starts = np.random.randint(0, len(shard) - seq_length - 1, batch_size)
            batch = np.stack([shard[i:i+seq_length+1] for i in batch_starts])
        else:
            batch_starts = np.random.randint(0, len(train_data) - seq_length - 1, batch_size)
            batch = np.stack([train_data[i:i+seq_length+1] for i in batch_starts])
        batch = torch.from_numpy(batch.astype(np.int64)).to(device)
        
        # Forward + backward
        if isinstance(optimizer, tuple):
            # Dual optimizer (Muon + AdamW)
            optimizer[0].zero_grad()
            optimizer[1].zero_grad()
        else:
            optimizer.zero_grad()
            
        if model_type == "latent":
            loss_dict = model.compute_loss(batch, sigreg_weight=sigreg_weight)
        elif model_type in ("standard", "mamba", "shared"):
            loss_dict = model.compute_loss(batch)
        else:
            loss_dict = model.compute_loss(batch)
        loss_dict['loss'].backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        if isinstance(optimizer, tuple):
            optimizer[0].step()
            optimizer[1].step()
        else:
            optimizer.step()
        
        losses.append(loss_dict['ce_loss'].item())
        
        # Progress to file every 15 minutes (not printed, saves tokens)
        if elapsed - last_progress_time >= progress_interval:
            avg_loss = np.mean(losses[-50:]) if losses else 0
            bpb = avg_loss / math.log(2)
            total_mins = (total_time_before + elapsed) / 60
            with open(progress_file, "a") as f:
                f.write(f"[{total_mins:.0f}min] step {step}, BPB {bpb:.2f}\n")
            last_progress_time = elapsed
        
        # Only print every 500 steps (minimal stdout)
        if step % 500 == 0 and step > start_step:
            avg_loss = np.mean(losses[-50:]) if losses else 0
            bpb = avg_loss / math.log(2)
            mins = elapsed / 60
            print(f"[{mins:.0f}min] step {step}, BPB {bpb:.2f}")
    
    # Final evaluation
    final_steps = step + 1
    elapsed = time.time() - start_time
    
    # Calculate final BPB
    model.eval()
    
    # Choose eval data source
    eval_data_src = val_data if val_data is not None else (train_data if not isinstance(train_data, list) else train_data[0])
    
    if sliding_window_eval and val_data is not None:
        # Sliding window evaluation (scores every token exactly once with maximum context)
        print(f"📏 Sliding window eval (stride={eval_stride}, batch_seqs={eval_batch_seqs})...")
        total_tokens = len(eval_data_src) - 1
        window_starts = list(range(0, total_tokens, eval_stride))
        
        loss_sum = 0.0
        token_count = 0
        
        with torch.no_grad():
            for bi in range(0, len(window_starts), eval_batch_seqs):
                batch_ws = window_starts[bi:bi + eval_batch_seqs]
                bsz = len(batch_ws)
                
                x_batch = torch.zeros(bsz, seq_length, dtype=torch.int64, device=device)
                y_batch = torch.zeros(bsz, seq_length, dtype=torch.int64, device=device)
                wlens = []
                
                for i, ws in enumerate(batch_ws):
                    end = min(ws + seq_length, total_tokens)
                    wlen = end - ws
                    wlens.append(wlen)
                    chunk = torch.from_numpy(eval_data_src[ws:end + 1].astype(np.int64))
                    x_batch[i, :wlen] = chunk[:-1]
                    y_batch[i, :wlen] = chunk[1:]
                
                logits = model(x_batch)
                
                nll = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y_batch.reshape(-1),
                    reduction="none",
                ).reshape(bsz, seq_length)
                
                for i, ws in enumerate(batch_ws):
                    wlen = wlens[i]
                    s = 0 if ws == 0 else max(wlen - eval_stride, 0)
                    scored_nll = nll[i, s:wlen]
                    loss_sum += scored_nll.sum().item()
                    token_count += (wlen - s)
        
        final_loss = loss_sum / token_count if token_count > 0 else float('nan')
        final_bpb = final_loss / math.log(2)
        eval_mode = f"sliding(stride={eval_stride})"
    else:
        # Standard evaluation: random batches from val/train
        val_losses = []
        with torch.no_grad():
            for _ in range(10):
                batch_starts = np.random.randint(0, len(eval_data_src) - seq_length - 1, batch_size)
                batch = np.stack([eval_data_src[i:i+seq_length+1] for i in batch_starts])
                batch = torch.from_numpy(batch.astype(np.int64)).to(device)
                
                if model_type == "latent":
                    loss_dict = model.compute_loss(batch, sigreg_weight=0)
                else:
                    loss_dict = model.compute_loss(batch)
                val_losses.append(loss_dict['ce_loss'].item())
        
        final_loss = float(np.mean(val_losses))
        final_bpb = final_loss / math.log(2)
        eval_mode = "standard"
    
    # Compact summary (reduces token usage in chat)
    mins = elapsed / 60
    total_mins = total_time_before / 60 + mins
    print(f"\n✅ {experiment_name}: {final_steps} steps, {mins:.0f}min (total {total_mins:.0f}min), BPB {final_bpb:.2f} [{eval_mode}], size {size_3bit:.1f}MB")
    
    # Save checkpoint for resuming
    total_time = total_time_before + elapsed
    
    if save_checkpoint:
        os.makedirs("/output/checkpoints", exist_ok=True)
        checkpoint_path = f"/output/checkpoints/{experiment_name}.pt"
        checkpoint = {
            'model': model.state_dict(),
            'step': int(final_steps),
            'total_time': float(total_time),
            'bpb': float(final_bpb),
            'loss': float(final_loss),
            'config': {
                'model_type': model_type,
                'embed_dim': embed_dim,
                'n_layers': n_layers,
                'n_heads': n_heads,
                'n_kv_heads': n_kv_heads,
                'vocab_size': vocab_size,
            }
        }
        if isinstance(optimizer, tuple):
            checkpoint['optimizer'] = optimizer[0].state_dict()
            checkpoint['optimizer2'] = optimizer[1].state_dict()
        else:
            checkpoint['optimizer'] = optimizer.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        print(f"💾 Checkpoint saved")
    
    output_volume.commit()
    
    # Save results
    result = {
        "status": "ok",
        "experiment_name": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "latent_dim": latent_dim,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "mlp_ratio": mlp_ratio,
            "embed_dim": embed_dim,
            "vocab_size": vocab_size,
            "seq_length": seq_length,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "sigreg_weight": sigreg_weight,
        },
        "results": {
            "params": n_params,
            "size_3bit_mb": size_3bit,
            "steps": final_steps,
            "time_seconds": elapsed,
            "total_time_seconds": total_time,
            "final_loss": final_loss,
            "final_bpb": final_bpb,
            "eval_mode": eval_mode,
        }
    }
    
    # Save to volume
    os.makedirs("/output/experiments", exist_ok=True)
    result_path = f"/output/experiments/{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    output_volume.commit()
    
    return result


@app.function(
    image=image,
    gpu="A100",
    volumes={"/output": output_volume},
    timeout=60,
)
def list_experiments():
    """List all experiment results"""
    exp_dir = "/output/experiments"
    if not os.path.exists(exp_dir):
        return {"experiments": []}
    
    results = []
    for f in sorted(os.listdir(exp_dir)):
        if f.endswith('.json'):
            with open(os.path.join(exp_dir, f)) as fp:
                data = json.load(fp)
                results.append({
                    "file": f,
                    "name": data.get("experiment_name"),
                    "bpb": data.get("results", {}).get("final_bpb"),
                    "params": data.get("results", {}).get("params"),
                })
    
    return {"experiments": results}


@app.local_entrypoint()
def main():
    print("Parameter Golf Experiment Runner")
    print("=" * 40)
    print("Commands:")
    print("  modal run experiment.py::run_experiment")
    print("  modal run experiment.py::run_experiment --latent-dim 128 --n-layers 12")
    print("  modal run experiment.py::list_experiments")

"""
Modal Runner - Entry point for running on Modal

Usage:
    # Download data first
    modal run modal_runner.py::download_data
    
    # Run training
    modal run modal_runner.py::train
    
    # Run with custom config
    modal run modal_runner.py::train --latent-dim 128 --n-layers 12
"""
import modal
import os

# Create Modal app
app = modal.App("parameter-golf")

# Create image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.0",
        "numpy",
        "sentencepiece",
        "huggingface-hub",
        "datasets",
        "tqdm",
        "zstandard",
    ])
)

# Volumes for data and outputs
data_volume = modal.Volume.from_name("parameter-golf-data", create_if_missing=True)
output_volume = modal.Volume.from_name("parameter-golf-output", create_if_missing=True)


@app.function(
    image=image,
    volumes={"/data": data_volume},
    timeout=600,
)
def download_data(variant: str = "sp1024", train_shards: int = 10):
    """Download FineWeb dataset to Modal volume"""
    import subprocess
    
    # Clone repo
    subprocess.run([
        "git", "clone", "https://github.com/openai/parameter-golf.git",
        "/tmp/parameter-golf"
    ], check=True)
    
    # Download data
    subprocess.run([
        "python", "/tmp/parameter-golf/data/cached_challenge_fineweb.py",
        "--variant", variant,
        "--train-shards", str(train_shards),
        "--output-dir", "/data"
    ], check=True)
    
    # Commit volume
    data_volume.commit()
    
    print("✅ Data downloaded successfully!")
    print(f"Files in /data: {os.listdir('/data')}")


@app.function(
    image=image,
    gpu="A100",  # or "H100" for faster training
    volumes={
        "/data": data_volume,
        "/output": output_volume,
    },
    timeout=900,  # 15 min max
)
def train(
    latent_dim: int = 64,
    n_layers: int = 8,
    seq_length: int = 1024,
    batch_size: int = 64,
    max_seconds: int = 300,
    sigreg_weight: float = 0.1,
    debug: bool = False,
):
    """
    Run training on Modal.
    
    Args:
        latent_dim: Latent space dimension
        n_layers: Number of predictor layers
        seq_length: Sequence length
        batch_size: Batch size
        max_seconds: Max training time in seconds
        sigreg_weight: Weight for SIGReg loss
        debug: Use small config for debugging
    """
    import sys
    sys.path.insert(0, "/tmp/solution")
    
    # Copy solution code to /tmp (Modal mounts are read-only)
    import shutil
    # In practice, we'd copy from the local mount
    # For now, we'll import directly
    
    from configs.base import Config, ModelConfig, TrainingConfig, get_debug_config
    from adapters.modal_adapter import ModalAdapter
    from train_core import Trainer
    
    # Build config
    if debug:
        config = get_debug_config()
    else:
        config = Config(
            model=ModelConfig(
                latent_dim=latent_dim,
                n_layers=n_layers,
            ),
            training=TrainingConfig(
                seq_length=seq_length,
                batch_size=batch_size,
                max_wallclock_seconds=max_seconds,
                sigreg_weight=sigreg_weight,
            ),
        )
    
    # Create adapter
    adapter = ModalAdapter(config)
    
    # TODO: Create dataloaders
    # For now, just test model creation
    trainer = Trainer(config, adapter)
    
    print("✅ Model created successfully!")
    print(f"Parameters: {trainer.model.count_parameters():,}")
    print(f"Estimated 3-bit size: {trainer.model.estimate_size(3):.2f} MB")
    
    # TODO: Add actual training with dataloaders
    # result = trainer.train(train_loader, val_loader)
    
    return {"status": "ok", "params": trainer.model.count_parameters()}


@app.function(
    image=image,
    gpu="A100",
    volumes={"/data": data_volume},
    timeout=120,
)
def test_model():
    """Quick test to verify model works"""
    import torch
    
    # Import local modules
    import sys
    sys.path.insert(0, ".")
    
    from configs.base import get_debug_config
    from models.latent_lm import LatentLM
    
    config = get_debug_config()
    model = LatentLM(config.model).cuda()
    
    # Test forward pass
    batch = torch.randint(0, config.model.vocab_size, (2, 64)).cuda()
    logits, z = model(batch)
    
    print(f"✅ Forward pass successful!")
    print(f"Input shape: {batch.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Latent shape: {z.shape}")
    
    # Test loss
    loss_dict = model.compute_loss(batch)
    print(f"Loss: {loss_dict['loss'].item():.4f}")
    print(f"CE Loss: {loss_dict['ce_loss'].item():.4f}")
    print(f"SIGReg Loss: {loss_dict['sigreg_loss'].item():.4f}")
    
    return {"status": "ok"}


@app.local_entrypoint()
def main():
    """Local entry point"""
    print("Parameter Golf - Modal Runner")
    print("=" * 40)
    print("Commands:")
    print("  modal run modal_runner.py::download_data")
    print("  modal run modal_runner.py::train")
    print("  modal run modal_runner.py::test_model")

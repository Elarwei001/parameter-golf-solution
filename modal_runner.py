"""
Modal Runner - Entry point for running on Modal

Usage:
    # Download data first
    modal run modal_runner.py::download_data
    
    # Run training
    modal run modal_runner.py::train
    
    # Test model on GPU
    modal run modal_runner.py::test_model
"""
import modal
import os

# Create Modal app
app = modal.App("parameter-golf")

# Create image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["git"])
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

# GitHub repo for our code
CODE_REPO = "https://github.com/Elarwei001/parameter-golf-solution.git"


def setup_code():
    """Clone our code repo to /root/project"""
    import subprocess
    if not os.path.exists("/root/project"):
        subprocess.run([
            "git", "clone", CODE_REPO, "/root/project"
        ], check=True)
    import sys
    sys.path.insert(0, "/root/project")


@app.function(
    image=image,
    volumes={"/data": data_volume},
    timeout=600,
)
def download_data(variant: str = "sp1024", train_shards: int = 10):
    """Download FineWeb dataset to Modal volume"""
    import subprocess
    
    # Clone official repo
    subprocess.run([
        "git", "clone", "https://github.com/openai/parameter-golf.git",
        "/tmp/parameter-golf"
    ], check=True)
    
    # Download data (positional: train_shards, then --variant)
    os.chdir("/tmp/parameter-golf/data")
    subprocess.run([
        "python", "cached_challenge_fineweb.py",
        str(train_shards),
        "--variant", variant,
    ], check=True)
    
    # Copy data to volume
    import shutil
    shutil.copytree("/tmp/parameter-golf/data/datasets", "/data/datasets", dirs_exist_ok=True)
    shutil.copytree("/tmp/parameter-golf/data/tokenizers", "/data/tokenizers", dirs_exist_ok=True)
    
    # Commit volume
    data_volume.commit()
    
    print("✅ Data downloaded successfully!")
    print(f"Files in /data: {os.listdir('/data')}")


@app.function(
    image=image,
    gpu="A100",
    volumes={
        "/data": data_volume,
        "/output": output_volume,
    },
    timeout=900,
)
def train(
    latent_dim: int = 64,
    n_layers: int = 8,
    max_seconds: int = 300,
    debug: bool = False,
):
    """Run training on Modal"""
    setup_code()
    
    from configs.base import Config, ModelConfig, TrainingConfig, get_debug_config
    from models.latent_lm import LatentLM
    
    import torch
    
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
                max_wallclock_seconds=max_seconds,
            ),
        )
    
    # Create model
    model = LatentLM(config.model).cuda()
    
    n_params = model.count_parameters()
    print(f"✅ Model created on GPU!")
    print(f"Parameters: {n_params:,}")
    print(f"Estimated 3-bit size: {model.estimate_size(3):.2f} MB")
    print(f"Device: {next(model.parameters()).device}")
    
    # Quick forward pass test
    batch = torch.randint(0, config.model.vocab_size, (2, 256)).cuda()
    logits, z = model(batch)
    print(f"Forward pass: input {batch.shape} → logits {logits.shape}")
    
    # Test loss computation
    loss_dict = model.compute_loss(batch)
    print(f"Loss: {loss_dict['loss'].item():.4f}")
    
    return {
        "status": "ok",
        "params": n_params,
        "size_3bit_mb": model.estimate_size(3),
    }


@app.function(
    image=image,
    gpu="A100",
    volumes={"/data": data_volume},
    timeout=120,
)
def test_model():
    """Quick test to verify model works on GPU"""
    setup_code()
    
    import torch
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
    
    # Model size info
    n_params = model.count_parameters()
    print(f"\nModel parameters: {n_params:,}")
    print(f"FP16 size: {model.estimate_size(16):.2f} MB")
    print(f"3-bit size: {model.estimate_size(3):.2f} MB")
    
    return {"status": "ok", "params": n_params}


@app.local_entrypoint()
def main():
    """Local entry point"""
    print("Parameter Golf - Modal Runner")
    print("=" * 40)
    print("Commands:")
    print("  modal run modal_runner.py::download_data")
    print("  modal run modal_runner.py::train")
    print("  modal run modal_runner.py::test_model")

"""
Test TurboQuant on a trained model
"""
import modal
import os

app = modal.App("parameter-golf-quant-test")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["git"])
    .pip_install([
        "torch>=2.0",
        "numpy",
    ])
)

data_volume = modal.Volume.from_name("parameter-golf-data", create_if_missing=True)
output_volume = modal.Volume.from_name("parameter-golf-output", create_if_missing=True)

CODE_REPO = "https://github.com/Elarwei001/parameter-golf-solution.git"


@app.function(
    image=image,
    gpu="A100",
    volumes={
        "/data": data_volume,
        "/output": output_volume,
    },
    timeout=300,
)
def test_quantization():
    """Test TurboQuant on StandardGPT model"""
    import subprocess
    import torch
    import numpy as np
    import sys
    
    # Clone code
    subprocess.run(["git", "clone", CODE_REPO, "/root/project"], check=True)
    sys.path.insert(0, "/root/project")
    
    from models.standard_gpt import StandardGPT
    from quant.turbo_quant import TurboQuant
    
    print("=" * 60)
    print("TurboQuant Quantization Test")
    print("=" * 60)
    
    # Create model (same config as our best experiment)
    model = StandardGPT(
        vocab_size=1024,
        dim=512,
        n_layers=9,
        n_heads=8,
        n_kv_heads=4,
        mlp_mult=4,
    ).cuda()
    
    n_params = sum(p.numel() for p in model.parameters())
    fp16_size = n_params * 2 / (1024**2)
    
    print(f"\nModel: StandardGPT")
    print(f"Parameters: {n_params:,}")
    print(f"FP16 size: {fp16_size:.2f} MB")
    
    # Test forward pass before quantization
    test_input = torch.randint(0, 1024, (2, 256)).cuda()
    with torch.no_grad():
        output_before = model(test_input)
        logits_before = output_before[:, -1, :].clone()
    
    # Quantize
    print("\nQuantizing with TurboQuant (3-bit + QJL)...")
    turbo = TurboQuant(bits=3, use_qjl=True, qjl_projections=32)
    quantized = turbo.quantize_model(model)
    
    # Estimate size
    _, quant_size = turbo.estimate_size(model)
    print(f"Estimated quantized size: {quant_size:.2f} MB")
    print(f"Compression ratio: {fp16_size/quant_size:.1f}x")
    
    # Save and get actual size
    os.makedirs("/output/quant_test", exist_ok=True)
    actual_size = turbo.save_compressed(quantized, "/output/quant_test/model.bin")
    actual_size_mb = actual_size / (1024**2)
    print(f"Actual compressed size: {actual_size_mb:.2f} MB")
    
    # Check if fits in 16MB
    if actual_size_mb <= 16:
        print(f"✅ Fits in 16MB limit!")
    else:
        print(f"❌ Exceeds 16MB limit by {actual_size_mb - 16:.2f} MB")
    
    # Dequantize and compare
    print("\nDequantizing...")
    model2 = StandardGPT(
        vocab_size=1024,
        dim=512,
        n_layers=9,
        n_heads=8,
        n_kv_heads=4,
        mlp_mult=4,
    ).cuda()
    turbo.dequantize_model(quantized, model2)
    
    # Test forward pass after quantization
    with torch.no_grad():
        output_after = model2(test_input)
        logits_after = output_after[:, -1, :]
    
    # Compare
    logit_error = (logits_before - logits_after).abs().mean()
    logit_max_error = (logits_before - logits_after).abs().max()
    
    print(f"\nOutput comparison:")
    print(f"  Mean logit error: {logit_error:.6f}")
    print(f"  Max logit error: {logit_max_error:.6f}")
    
    # Check if predictions match
    pred_before = logits_before.argmax(dim=-1)
    pred_after = logits_after.argmax(dim=-1)
    pred_match = (pred_before == pred_after).float().mean()
    print(f"  Prediction match rate: {pred_match*100:.1f}%")
    
    # Test actual loss difference
    print("\nTesting on real data...")
    data_path = "/data/datasets/fineweb10B_sp1024"
    train_files = sorted([f for f in os.listdir(data_path) if "train" in f])
    if train_files:
        train_path = os.path.join(data_path, train_files[0])
        train_data = np.memmap(train_path, dtype=np.uint16, mode='r')
        
        # Sample batch
        batch_starts = np.random.randint(0, len(train_data) - 1025, 16)
        batch = np.stack([train_data[i:i+1025] for i in batch_starts])
        batch = torch.from_numpy(batch.astype(np.int64)).cuda()
        
        with torch.no_grad():
            loss_before = model.compute_loss(batch)['ce_loss'].item()
            loss_after = model2.compute_loss(batch)['ce_loss'].item()
        
        bpb_before = loss_before / 0.6931
        bpb_after = loss_after / 0.6931
        
        print(f"  Original BPB: {bpb_before:.4f}")
        print(f"  Quantized BPB: {bpb_after:.4f}")
        print(f"  BPB degradation: {bpb_after - bpb_before:.4f}")
    
    output_volume.commit()
    
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  FP16 size: {fp16_size:.2f} MB")
    print(f"  Quantized size: {actual_size_mb:.2f} MB")
    print(f"  Compression: {fp16_size/actual_size_mb:.1f}x")
    print(f"  Fits 16MB: {'✅' if actual_size_mb <= 16 else '❌'}")
    print("=" * 60)
    
    return {
        "fp16_size_mb": fp16_size,
        "quantized_size_mb": actual_size_mb,
        "compression_ratio": fp16_size / actual_size_mb,
        "fits_16mb": actual_size_mb <= 16,
        "logit_error": float(logit_error),
    }


@app.local_entrypoint()
def main():
    print("Run: modal run test_quant.py::test_quantization")

#!/usr/bin/env python3
"""
Local test script - Verify model works without GPU

Usage:
    python test_local.py
"""
import torch
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.base import get_debug_config, ModelConfig
from models.latent_lm import LatentLM, sigreg_loss


def test_sigreg():
    """Test SIGReg loss"""
    print("Testing SIGReg loss...")
    
    # Random latent (should have some loss)
    z_random = torch.randn(4, 32, 64)
    loss_random = sigreg_loss(z_random)
    print(f"  Random latent loss: {loss_random.item():.4f}")
    
    # Perfect Gaussian (should have ~0 loss)
    z_perfect = torch.randn(1000, 32, 64)
    z_perfect = (z_perfect - z_perfect.mean()) / z_perfect.std()
    loss_perfect = sigreg_loss(z_perfect)
    print(f"  Normalized latent loss: {loss_perfect.item():.4f}")
    
    # Collapsed (should have high loss)
    z_collapsed = torch.ones(4, 32, 64) * 5
    loss_collapsed = sigreg_loss(z_collapsed)
    print(f"  Collapsed latent loss: {loss_collapsed.item():.4f}")
    
    assert loss_perfect < loss_random < loss_collapsed, "SIGReg ordering wrong"
    print("  ✅ SIGReg loss working correctly!")


def test_model_shapes():
    """Test model forward pass shapes"""
    print("\nTesting model shapes...")
    
    config = get_debug_config()
    model = LatentLM(config.model)
    
    # Test input
    batch_size = 2
    seq_len = 64
    batch = torch.randint(0, config.model.vocab_size, (batch_size, seq_len))
    
    # Forward pass
    logits, z = model(batch)
    
    print(f"  Input: {batch.shape}")
    print(f"  Logits: {logits.shape}")
    print(f"  Latent: {z.shape}")
    
    assert logits.shape == (batch_size, seq_len, config.model.vocab_size)
    assert z.shape == (batch_size, seq_len, config.model.latent_dim)
    print("  ✅ Shapes correct!")


def test_model_loss():
    """Test loss computation"""
    print("\nTesting loss computation...")
    
    config = get_debug_config()
    model = LatentLM(config.model)
    
    batch = torch.randint(0, config.model.vocab_size, (2, 64))
    loss_dict = model.compute_loss(batch, sigreg_weight=0.1)
    
    print(f"  Total loss: {loss_dict['loss'].item():.4f}")
    print(f"  CE loss: {loss_dict['ce_loss'].item():.4f}")
    print(f"  SIGReg loss: {loss_dict['sigreg_loss'].item():.4f}")
    print(f"  Perplexity: {loss_dict['ppl'].item():.2f}")
    
    assert loss_dict['loss'].requires_grad, "Loss should require grad"
    print("  ✅ Loss computation working!")


def test_model_backward():
    """Test backward pass"""
    print("\nTesting backward pass...")
    
    config = get_debug_config()
    model = LatentLM(config.model)
    
    batch = torch.randint(0, config.model.vocab_size, (2, 64))
    loss_dict = model.compute_loss(batch)
    
    loss_dict['loss'].backward()
    
    # Check gradients exist
    grad_count = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for p in model.parameters())
    
    print(f"  Params with gradients: {grad_count}/{total_params}")
    assert grad_count == total_params, "Not all params have gradients"
    print("  ✅ Backward pass working!")


def test_model_size():
    """Test model size estimation"""
    print("\nTesting model size...")
    
    # Default config (what we'll use for competition)
    config = ModelConfig(
        vocab_size=1024,
        embed_dim=256,
        latent_dim=64,
        n_layers=8,
        n_heads=8,
    )
    model = LatentLM(config)
    
    n_params = model.count_parameters()
    size_fp16 = model.estimate_size(16)
    size_3bit = model.estimate_size(3)
    
    print(f"  Parameters: {n_params:,}")
    print(f"  FP16 size: {size_fp16:.2f} MB")
    print(f"  3-bit size: {size_3bit:.2f} MB")
    
    # Check 3-bit fits in 16MB
    if size_3bit < 16:
        print(f"  ✅ Fits in 16MB limit!")
    else:
        print(f"  ⚠️  Exceeds 16MB limit, need to reduce params")
    
    # Estimate how many params we can fit
    max_params_3bit = int(16 * 1024 * 1024 * 8 / 3)
    print(f"  Max params at 3-bit for 16MB: {max_params_3bit:,}")


def test_generation():
    """Test text generation"""
    print("\nTesting generation...")
    
    config = get_debug_config()
    model = LatentLM(config.model)
    model.eval()
    
    prompt = torch.randint(0, config.model.vocab_size, (1, 10))
    
    with torch.no_grad():
        output = model.generate(prompt, max_new_tokens=20)
    
    print(f"  Prompt length: {prompt.shape[1]}")
    print(f"  Output length: {output.shape[1]}")
    print(f"  Generated {output.shape[1] - prompt.shape[1]} new tokens")
    print("  ✅ Generation working!")


def main():
    print("=" * 50)
    print("Parameter Golf - Local Model Tests")
    print("=" * 50)
    
    test_sigreg()
    test_model_shapes()
    test_model_loss()
    test_model_backward()
    test_model_size()
    test_generation()
    
    print("\n" + "=" * 50)
    print("All tests passed! ✅")
    print("=" * 50)


if __name__ == "__main__":
    main()

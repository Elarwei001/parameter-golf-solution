"""
Test weight-sharing implementation.
Compares:
  - Standard GPT: 9 layers, dim=512
  - Shared GPT: 3 unique layers × 3 passes, dim=640
"""
import torch
from models.standard_gpt import StandardGPT


def test_standard():
    model = StandardGPT(
        vocab_size=1024,
        dim=512,
        n_layers=9,
        n_heads=8,
        n_kv_heads=4,
        mlp_mult=4,
        max_seq_len=1024,
        tie_embeddings=True,
    )
    n_params = model.count_parameters()
    size_3bit = model.estimate_size(3)
    print(f"[Standard 9L dim=512]  params={n_params/1e6:.2f}M  size@3bit={size_3bit:.2f}MB")

    # Forward pass
    x = torch.randint(0, 1024, (2, 16))
    logits = model(x)
    assert logits.shape == (2, 16, 1024), f"Unexpected shape: {logits.shape}"
    print(f"  forward OK: {logits.shape}")

    # Loss
    tokens = torch.randint(0, 1024, (2, 17))
    loss_dict = model.compute_loss(tokens)
    assert 'loss' in loss_dict
    print(f"  loss OK: {loss_dict['loss'].item():.4f}")
    return n_params


def test_shared():
    model = StandardGPT(
        vocab_size=1024,
        dim=640,
        n_layers=9,       # ignored when shared_layers > 0
        n_heads=8,
        n_kv_heads=4,
        mlp_mult=4,
        max_seq_len=1024,
        tie_embeddings=True,
        shared_layers=3,
        n_passes=3,
    )
    n_params = model.count_parameters()
    size_3bit = model.estimate_size(3)
    print(f"[Shared 3L×3 dim=640]  params={n_params/1e6:.2f}M  size@3bit={size_3bit:.2f}MB")
    print(f"  unique blocks: {len(model.blocks)}, effective depth: {model._effective_depth}")

    # Forward pass
    x = torch.randint(0, 1024, (2, 16))
    logits = model(x)
    assert logits.shape == (2, 16, 1024), f"Unexpected shape: {logits.shape}"
    print(f"  forward OK: {logits.shape}")

    # Loss
    tokens = torch.randint(0, 1024, (2, 17))
    loss_dict = model.compute_loss(tokens)
    assert 'loss' in loss_dict
    print(f"  loss OK: {loss_dict['loss'].item():.4f}")
    return n_params


def test_backward():
    """Ensure gradients flow properly through shared layers."""
    model = StandardGPT(
        vocab_size=1024,
        dim=256,
        n_layers=9,
        n_heads=4,
        n_kv_heads=2,
        mlp_mult=4,
        max_seq_len=64,
        shared_layers=3,
        n_passes=3,
    )
    tokens = torch.randint(0, 1024, (2, 17))
    loss_dict = model.compute_loss(tokens)
    loss_dict['loss'].backward()

    # Check all params have gradients
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No grad for {name}"
    print("[Backward] all gradients present ✓")


if __name__ == "__main__":
    print("=" * 60)
    print("Weight-Sharing Test")
    print("=" * 60)

    p_std = test_standard()
    print()
    p_shared = test_shared()
    print()
    test_backward()
    print()

    ratio = p_std / p_shared
    print(f"Parameter ratio (standard / shared): {ratio:.2f}x")
    print("=" * 60)
    print("All tests passed! ✓")

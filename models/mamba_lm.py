"""
Mamba-3 Language Model for Parameter Golf

Adapted from mamba3-minimal (https://github.com/VikramKarLex/mamba3-minimal)
Pure PyTorch implementation, no custom CUDA kernels needed.

Key difference from Transformer:
- O(n) complexity instead of O(n²)
- Fixed-size state instead of growing KV cache
- Potentially faster training within 10-minute limit
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class MambaConfig:
    """Mamba model configuration for Parameter Golf (16MB limit)."""
    vocab_size: int = 1024
    d_model: int = 384        # Model dimension
    n_layer: int = 8          # Number of layers
    d_state: int = 64         # SSM state dimension (must be even)
    expand: int = 2           # Expansion factor
    headdim: int = 32         # Head dimension
    chunk_size: int = 64      # Chunk size for SSD
    max_seq_len: int = 1024
    
    def __post_init__(self):
        self.d_inner = self.expand * self.d_model
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        assert self.d_state % 2 == 0
        # SwiGLU inner dim
        self.d_mlp = int(2.5 * self.d_model)


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def silu(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network."""
    def __init__(self, d_model: int, d_inner: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_inner, bias=False)
        self.w_up = nn.Linear(d_model, d_inner, bias=False)
        self.w_down = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w_down(silu(self.w_gate(x)) * self.w_up(x))


def apply_rope(x: Tensor, angles: Tensor) -> Tensor:
    """Apply rotary position embedding."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)
    x_rot_even = cos_a * x1 - sin_a * x2
    x_rot_odd = sin_a * x1 + cos_a * x2
    return torch.stack([x_rot_even, x_rot_odd], dim=-1).flatten(-2)


def simple_ssm_scan(x: Tensor, A: Tensor, B: Tensor, C: Tensor) -> Tensor:
    """Simple sequential SSM scan (no chunking, pure PyTorch).
    
    For debugging and correctness. Slower but simpler.
    
    Args:
        x: (batch, seqlen, nheads, headdim)
        A: (batch, seqlen, nheads) - log decay rates (negative)
        B: (batch, seqlen, nheads, d_state)
        C: (batch, seqlen, nheads, d_state)
    
    Returns:
        y: (batch, seqlen, nheads, headdim)
    """
    batch, seqlen, nheads, headdim = x.shape
    d_state = B.shape[-1]
    device = x.device
    dtype = x.dtype
    
    # Initialize state
    h = torch.zeros(batch, nheads, headdim, d_state, device=device, dtype=dtype)
    
    outputs = []
    for t in range(seqlen):
        # Get current inputs
        x_t = x[:, t]  # (batch, nheads, headdim)
        A_t = A[:, t]  # (batch, nheads)
        B_t = B[:, t]  # (batch, nheads, d_state)
        C_t = C[:, t]  # (batch, nheads, d_state)
        
        # State update: h = exp(A) * h + B * x
        decay = torch.exp(A_t).unsqueeze(-1).unsqueeze(-1)  # (batch, nheads, 1, 1)
        Bx = torch.einsum("bhn, bhp -> bhpn", B_t, x_t)  # (batch, nheads, headdim, d_state)
        h = h * decay + Bx
        
        # Output: y = C @ h
        y_t = torch.einsum("bhpn, bhn -> bhp", h, C_t)  # (batch, nheads, headdim)
        outputs.append(y_t)
    
    y = torch.stack(outputs, dim=1)  # (batch, seqlen, nheads, headdim)
    return y


def ssd(x: Tensor, A: Tensor, B: Tensor, C: Tensor, chunk_size: int, 
        initial_states: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
    """Structured State Space Duality (SSD) algorithm.
    
    Uses simple sequential scan for now (can optimize later with chunking).
    """
    batch, seqlen, nheads, headdim = x.shape
    
    y = simple_ssm_scan(x, A, B, C)
    y = y.reshape(batch, seqlen, nheads * headdim)
    
    # Return dummy final state for now
    final_state = torch.zeros(batch, nheads, headdim, B.shape[-1], device=x.device)
    
    return y, final_state


class Mamba3Block(nn.Module):
    """Single Mamba-3 block: SSM mixer + SwiGLU MLP."""
    
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        d = config.d_model
        d_inner = config.d_inner
        nheads = config.nheads
        headdim = config.headdim
        d_state = config.d_state
        
        self.norm1 = RMSNorm(d)
        self.norm2 = RMSNorm(d)
        
        # SSM projections
        # in_proj: x -> (z, x_ssm, B, C, dt, theta)
        self.in_proj = nn.Linear(d, d_inner + d_inner + nheads * d_state + nheads * d_state + nheads + nheads * (d_state // 2), bias=False)
        self.out_proj = nn.Linear(d_inner, d, bias=False)
        
        # SSM parameters
        self.A_log = nn.Parameter(torch.zeros(nheads))
        self.D = nn.Parameter(torch.ones(nheads))
        self.dt_bias = nn.Parameter(torch.zeros(nheads))
        
        # B, C biases (Mamba-3 innovation)
        self.B_bias = nn.Parameter(torch.ones(1, nheads, d_state))
        self.C_bias = nn.Parameter(torch.ones(1, nheads, d_state))
        
        # QK norm
        self.B_norm = RMSNorm(d_state)
        self.C_norm = RMSNorm(d_state)
        
        # MLP
        self.mlp = SwiGLU(d, config.d_mlp)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.uniform_(self.A_log, -4, -1)
        nn.init.uniform_(self.dt_bias, 0.001, 0.1)
    
    def forward(self, x: Tensor) -> Tensor:
        batch, seqlen, _ = x.shape
        config = self.config
        nheads = config.nheads
        headdim = config.headdim
        d_state = config.d_state
        d_inner = config.d_inner
        
        # === SSM Block ===
        h = self.norm1(x)
        
        # Project
        proj = self.in_proj(h)
        
        # Split projections
        idx = 0
        z = proj[..., idx:idx+d_inner]
        idx += d_inner
        x_ssm = proj[..., idx:idx+d_inner]
        idx += d_inner
        B = proj[..., idx:idx+nheads*d_state].view(batch, seqlen, nheads, d_state)
        idx += nheads * d_state
        C = proj[..., idx:idx+nheads*d_state].view(batch, seqlen, nheads, d_state)
        idx += nheads * d_state
        dt = proj[..., idx:idx+nheads]
        idx += nheads
        theta = proj[..., idx:].view(batch, seqlen, nheads, d_state // 2)
        
        # Process dt
        dt = F.softplus(dt + self.dt_bias)  # (batch, seqlen, nheads)
        
        # Compute A
        A = -torch.exp(self.A_log) * dt  # (batch, seqlen, nheads)
        
        # Trapezoidal coefficients (Mamba-3)
        alpha = torch.exp(A)
        beta = (alpha - 1) / (A + 1e-6) * 0.5
        gamma = (alpha - 1) / (A + 1e-6) * 0.5 + 1
        
        # QK-Norm + bias
        B = self.B_norm(B) + self.B_bias
        C = self.C_norm(C) + self.C_bias
        
        # RoPE (cumulative angles)
        cum_theta = torch.cumsum(dt.unsqueeze(-1) * theta, dim=1)
        B = apply_rope(B, cum_theta)
        C = apply_rope(C, cum_theta)
        
        # Reshape for SSD
        x_ssm = x_ssm.view(batch, seqlen, nheads, headdim)
        
        # Apply SSD with trapezoidal (simplified: use gamma term only for now)
        x_scaled = x_ssm * gamma.unsqueeze(-1)
        y, _ = ssd(x_scaled, A, B, C, config.chunk_size)
        
        # Skip connection
        y = y + (self.D.view(1, 1, nheads) * x_ssm.sum(-1)).unsqueeze(-1).expand_as(x_ssm).reshape(batch, seqlen, -1)
        
        # Gate and project
        y = y * silu(z)
        y = self.out_proj(y)
        
        x = x + y
        
        # === MLP Block ===
        x = x + self.mlp(self.norm2(x))
        
        return x


class MambaLM(nn.Module):
    """Mamba-3 Language Model for Parameter Golf."""
    
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([Mamba3Block(config) for _ in range(config.n_layer)])
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Tie embeddings
        self.lm_head.weight = self.embedding.weight
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def forward(self, input_ids: Tensor) -> Tensor:
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits
    
    def compute_loss(self, batch: Tensor) -> dict:
        """Compute cross-entropy loss."""
        input_ids = batch[:, :-1]
        targets = batch[:, 1:]
        
        logits = self.forward(input_ids)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        
        return {"loss": loss, "ce_loss": loss}
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def estimate_size(self, bits: int = 3) -> float:
        """Estimate model size in MB at given bit precision."""
        n_params = self.count_parameters()
        return n_params * bits / 8 / 1024 / 1024


def create_mamba_for_golf(
    vocab_size: int = 1024,
    target_size_mb: float = 12.0,
    bits: int = 3,
) -> MambaLM:
    """Create a Mamba model that fits within Parameter Golf limits."""
    
    # Target params for given size
    target_params = int(target_size_mb * 1024 * 1024 * 8 / bits)
    
    # Search for good config
    # Start with reasonable defaults and adjust
    configs_to_try = [
        MambaConfig(vocab_size=vocab_size, d_model=384, n_layer=8, d_state=64),
        MambaConfig(vocab_size=vocab_size, d_model=448, n_layer=8, d_state=64),
        MambaConfig(vocab_size=vocab_size, d_model=512, n_layer=6, d_state=64),
        MambaConfig(vocab_size=vocab_size, d_model=384, n_layer=10, d_state=64),
    ]
    
    best_config = configs_to_try[0]
    best_diff = float('inf')
    
    for config in configs_to_try:
        model = MambaLM(config)
        n_params = model.count_parameters()
        size_mb = model.estimate_size(bits)
        
        if size_mb <= target_size_mb:
            diff = target_size_mb - size_mb
            if diff < best_diff:
                best_diff = diff
                best_config = config
    
    return MambaLM(best_config)


if __name__ == "__main__":
    # Quick test
    print("Testing MambaLM...")
    
    config = MambaConfig(
        vocab_size=1024,
        d_model=256,
        n_layer=4,
        d_state=32,
    )
    model = MambaLM(config)
    
    n_params = model.count_parameters()
    size_3bit = model.estimate_size(3)
    
    print(f"Parameters: {n_params:,}")
    print(f"Size @ 3-bit: {size_3bit:.2f} MB")
    
    # Test forward
    x = torch.randint(0, config.vocab_size, (2, 64))
    logits = model(x)
    print(f"Input: {x.shape}, Output: {logits.shape}")
    
    # Test loss
    batch = torch.randint(0, config.vocab_size, (2, 65))
    loss_dict = model.compute_loss(batch)
    print(f"Loss: {loss_dict['loss'].item():.4f}")
    
    print("✅ MambaLM test passed!")

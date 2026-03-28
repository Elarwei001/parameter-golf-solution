"""
Latent Language Model (LeWM-inspired)

Architecture:
    Token → Encoder → Latent → Predictor → Decoder → Logits

Key features:
    - Compact latent space (64-128 dim vs 512+ in standard transformers)
    - SIGReg regularization (prevents latent collapse)
    - Tied encoder/decoder weights (parameter efficient)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from configs.base import ModelConfig


# =============================================================================
# SIGReg Loss (from LeWM/LeJEPA)
# =============================================================================

def sigreg_loss(z: torch.Tensor) -> torch.Tensor:
    """
    Gaussian regularizer to prevent latent collapse.
    Forces latent space to approximately follow N(0,1) distribution.
    
    From: LeWorldModel (LeCun et al., 2026)
    
    Args:
        z: [batch, seq_len, latent_dim] latent representations
    
    Returns:
        Scalar loss value
    """
    z_flat = z.reshape(-1, z.size(-1))  # [B*L, D]
    
    # Mean of each dimension should be ~0
    mean_loss = (z_flat.mean(dim=0) ** 2).mean()
    
    # Std of each dimension should be ~1
    std_loss = ((z_flat.std(dim=0) - 1) ** 2).mean()
    
    return mean_loss + std_loss


# =============================================================================
# Encoder
# =============================================================================

class Encoder(nn.Module):
    """
    Encodes tokens to compact latent representations.
    
    Token IDs → Embedding → Projection → Latent
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Token embedding
        self.embed = nn.Embedding(config.vocab_size, config.embed_dim)
        
        # Project to latent space
        self.proj = nn.Linear(config.embed_dim, config.latent_dim)
        
        # Positional encoding (learned)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 8192, config.latent_dim)  # Max seq len
        )
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)
        nn.init.normal_(self.pos_embed, std=0.02)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: [batch, seq_len] int tensor
        Returns:
            latent: [batch, seq_len, latent_dim] float tensor
        """
        B, L = token_ids.shape
        
        # Embed tokens
        x = self.embed(token_ids)  # [B, L, embed_dim]
        
        # Project to latent
        z = self.proj(x)  # [B, L, latent_dim]
        
        # Add positional encoding
        z = z + self.pos_embed[:, :L, :]
        
        return z


# =============================================================================
# Latent Predictor (Transformer in latent space)
# =============================================================================

class LatentAttention(nn.Module):
    """Causal self-attention in latent space."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.latent_dim // config.n_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(config.latent_dim, 3 * config.latent_dim, bias=False)
        self.proj = nn.Linear(config.latent_dim, config.latent_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, L, D = z.shape
        
        # QKV projection
        qkv = self.qkv(z).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, L, d]
        
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Causal mask
        mask = torch.triu(torch.ones(L, L, device=z.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask, float('-inf'))
        
        # Softmax and apply
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Combine heads
        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.proj(out)


class LatentMLP(nn.Module):
    """MLP block in latent space."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden_dim = int(config.latent_dim * config.mlp_ratio)
        
        self.fc1 = nn.Linear(config.latent_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, config.latent_dim, bias=False)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.fc1(z)
        z = self.act(z)
        z = self.dropout(z)
        z = self.fc2(z)
        return z


class LatentBlock(nn.Module):
    """Transformer block operating in latent space (Pre-LN)."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.latent_dim)
        self.attn = LatentAttention(config)
        self.ln2 = nn.LayerNorm(config.latent_dim)
        self.mlp = LatentMLP(config)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = z + self.attn(self.ln1(z))
        z = z + self.mlp(self.ln2(z))
        return z


class Predictor(nn.Module):
    """
    Predicts next latent from current latent sequence.
    Stack of transformer blocks operating in compact latent space.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.blocks = nn.ModuleList([
            LatentBlock(config) for _ in range(config.n_layers)
        ])
        self.ln_out = nn.LayerNorm(config.latent_dim)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            z = block(z)
        return self.ln_out(z)


# =============================================================================
# Decoder
# =============================================================================

class Decoder(nn.Module):
    """
    Decodes latent to token logits.
    Optionally ties weights with encoder for parameter efficiency.
    """
    
    def __init__(self, config: ModelConfig, encoder: Optional[Encoder] = None):
        super().__init__()
        self.config = config
        self.tied = config.tie_weights and encoder is not None
        
        if self.tied:
            # Use encoder's weights (transposed)
            self.encoder = encoder
            self.proj_inv = nn.Linear(config.latent_dim, config.embed_dim, bias=False)
        else:
            # Separate output projection
            self.proj = nn.Linear(config.latent_dim, config.vocab_size)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [batch, seq_len, latent_dim]
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        if self.tied:
            # Project back to embed space, then use embedding weights
            x = self.proj_inv(z)  # [B, L, embed_dim]
            logits = x @ self.encoder.embed.weight.T  # [B, L, vocab_size]
        else:
            logits = self.proj(z)
        
        return logits


# =============================================================================
# Full Model
# =============================================================================

class LatentLM(nn.Module):
    """
    Latent Language Model combining encoder, predictor, and decoder.
    
    Forward pass:
        tokens → encoder → latent → predictor → decoder → logits
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.encoder = Encoder(config)
        self.predictor = Predictor(config)
        self.decoder = Decoder(config, encoder=self.encoder if config.tie_weights else None)
    
    def forward(self, token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            token_ids: [batch, seq_len] input tokens
        
        Returns:
            logits: [batch, seq_len, vocab_size]
            latent: [batch, seq_len, latent_dim] (for SIGReg loss)
        """
        z = self.encoder(token_ids)
        z_pred = self.predictor(z)
        logits = self.decoder(z_pred)
        return logits, z
    
    def compute_loss(
        self, 
        token_ids: torch.Tensor, 
        sigreg_weight: float = 0.1
    ) -> dict:
        """
        Compute total loss with SIGReg regularization.
        
        Args:
            token_ids: [batch, seq_len] input tokens
            sigreg_weight: weight for SIGReg loss
        
        Returns:
            dict with 'loss', 'ce_loss', 'sigreg_loss', 'ppl'
        """
        # Forward pass
        logits, z = self.forward(token_ids[:, :-1])
        targets = token_ids[:, 1:]
        
        # Cross-entropy loss
        ce_loss = F.cross_entropy(
            logits.reshape(-1, self.config.vocab_size),
            targets.reshape(-1)
        )
        
        # SIGReg loss
        sig_loss = sigreg_loss(z)
        
        # Total loss
        total_loss = ce_loss + sigreg_weight * sig_loss
        
        # Perplexity
        ppl = torch.exp(ce_loss)
        
        return {
            'loss': total_loss,
            'ce_loss': ce_loss,
            'sigreg_loss': sig_loss,
            'ppl': ppl,
        }
    
    def count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def estimate_size(self, bits: int = 16) -> float:
        """Estimate model size in MB at given precision"""
        n_params = self.count_parameters()
        bytes_per_param = bits / 8
        return (n_params * bytes_per_param) / (1024 * 1024)
    
    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Generate tokens autoregressively."""
        self.eval()
        
        for _ in range(max_new_tokens):
            # Forward
            logits, _ = self.forward(prompt_ids)
            logits = logits[:, -1, :] / temperature
            
            # Top-k sampling
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            prompt_ids = torch.cat([prompt_ids, next_token], dim=1)
        
        return prompt_ids

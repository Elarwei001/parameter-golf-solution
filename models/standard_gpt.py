"""
Standard GPT model for comparison.
Based on official baseline architecture but simplified.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding"""
    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos", emb.cos())
        self.register_buffer("sin", emb.sin())
    
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cos[:seq_len], self.sin[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Attention(nn.Module):
    """Multi-head attention with GQA support"""
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int = None):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.head_dim = dim // n_heads
        self.n_rep = n_heads // self.n_kv_heads
        
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)
    
    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        
        q = self.wq(x).view(B, L, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, L, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, L, self.n_kv_heads, self.head_dim)
        
        # Apply RoPE
        q = q.transpose(1, 2)  # [B, H, L, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        q, k = apply_rotary_pos_emb(q, k, cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0))
        
        # Repeat KV for GQA
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
        
        # Attention
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        
        # Causal mask
        mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(B, L, -1)
        return self.wo(out)


class MLP(nn.Module):
    """Feed-forward network with SwiGLU"""
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        hidden = int(dim * mult * 2 / 3)  # SwiGLU has 2/3 hidden dim
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    """Transformer block"""
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int = None, mlp_mult: int = 4):
        super().__init__()
        self.ln1 = RMSNorm(dim)
        self.attn = Attention(dim, n_heads, n_kv_heads)
        self.ln2 = RMSNorm(dim)
        self.mlp = MLP(dim, mlp_mult)
    
    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.mlp(self.ln2(x))
        return x


class StandardGPT(nn.Module):
    """
    Standard GPT model (not latent-based).
    For comparison with LatentLM.
    """
    def __init__(
        self,
        vocab_size: int = 1024,
        dim: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        n_kv_heads: int = 4,
        mlp_mult: int = 4,
        max_seq_len: int = 1024,
        tie_embeddings: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.tie_embeddings = tie_embeddings
        
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.rope = RotaryEmbedding(dim // n_heads, max_seq_len)
        
        self.blocks = nn.ModuleList([
            Block(dim, n_heads, n_kv_heads, mlp_mult)
            for _ in range(n_layers)
        ])
        
        self.ln_f = RMSNorm(dim)
        
        if not tie_embeddings:
            self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len] token ids
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        B, L = x.shape
        
        h = self.tok_emb(x)
        cos, sin = self.rope(h, L)
        
        for block in self.blocks:
            h = block(h, cos, sin)
        
        h = self.ln_f(h)
        
        if self.tie_embeddings:
            logits = h @ self.tok_emb.weight.T
        else:
            logits = self.lm_head(h)
        
        return logits
    
    def compute_loss(self, token_ids: torch.Tensor) -> dict:
        """Compute cross-entropy loss"""
        logits = self.forward(token_ids[:, :-1])
        targets = token_ids[:, 1:]
        
        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            targets.reshape(-1)
        )
        
        return {
            'loss': loss,
            'ce_loss': loss,
            'ppl': torch.exp(loss),
        }
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def estimate_size(self, bits: int = 16) -> float:
        n_params = self.count_parameters()
        return (n_params * bits / 8) / (1024 * 1024)

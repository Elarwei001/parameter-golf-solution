"""
Per-Layer Embeddings (PLE) 实验

思路：每层都注入 token embedding 信号，让深层也能"回看" token 的原始身份

实现方式：
1. 主 embedding: vocab -> dim (标准)
2. 小 embedding: vocab -> small_dim (每层共享或独立)
3. 每层: x = x + proj(small_embed(tokens))

预期效果：
- 对小模型 (16M) 特别有价值
- 词表小 (8192)，嵌入表开销不大
- BPB 可能下降
"""

import modal
import os

app = modal.App("ple-experiment")

# Modal 镜像配置
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "numpy",
    )
)

vol = modal.Volume.from_name("parameter-golf-data", create_if_missing=True)

@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=3600,
    volumes={"/data": vol},
)
def train_ple(
    n_layers: int = 11,
    dim: int = 416,
    ple_dim: int = 64,  # 小 embedding 的维度
    ple_mode: str = "shared",  # "shared" 或 "per_layer"
    n_steps: int = 5000,
    seed: int = 42,
):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import math
    import time

    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("=" * 70)
    print(f"PLE 实验: 每层注入 token embedding")
    print(f"模式: {ple_mode}, PLE dim: {ple_dim}")
    print("=" * 70)

    # ========== 超参数 ==========
    vocab_size = 8192
    seq_len = 1024
    batch_size = 32
    n_heads = 8  # 416 / 8 = 52, head_dim=52
    head_dim = dim // n_heads
    
    lr_max = 1e-3
    warmup_steps = 200
    
    HEADER_SIZE = 1024  # 256 int32 = 1024 bytes

    # ========== 数据加载 ==========
    print("\n加载数据...")
    
    train_path = "/data/datasets/fineweb10B_sp8192/fineweb_train_000000.bin"
    val_path = "/data/datasets/fineweb10B_sp8192/fineweb_val_000000.bin"
    
    def load_tokens(path):
        with open(path, 'rb') as f:
            f.seek(HEADER_SIZE)
            tokens = np.frombuffer(f.read(), dtype=np.uint16).astype(np.int32)
        return torch.from_numpy(tokens)
    
    train_tokens = load_tokens(train_path)
    val_tokens = load_tokens(val_path)
    
    print(f"  Train: {len(train_tokens) / 1e6:.1f}M tokens")
    print(f"  Val: {len(val_tokens) / 1e6:.1f}M tokens")

    # ========== RoPE ==========
    def precompute_rope(dim, max_seq_len, base=10000.0):
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len)
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        return cos.to(device), sin.to(device)

    def apply_rope(x, cos, sin):
        B, H, T, D = x.shape
        x1, x2 = x[..., :D//2], x[..., D//2:]
        cos_t = cos[:T].unsqueeze(0).unsqueeze(0)
        sin_t = sin[:T].unsqueeze(0).unsqueeze(0)
        return torch.cat([x1 * cos_t - x2 * sin_t, x1 * sin_t + x2 * cos_t], dim=-1)

    head_dim = dim // n_heads
    rope_cos, rope_sin = precompute_rope(head_dim, seq_len + 1)

    # ========== 模型定义 ==========
    class Attention(nn.Module):
        def __init__(self, dim, n_heads):
            super().__init__()
            self.n_heads = n_heads
            self.head_dim = dim // n_heads
            self.qkv = nn.Linear(dim, 3 * dim, bias=False)
            self.proj = nn.Linear(dim, dim, bias=False)

        def forward(self, x):
            B, T, C = x.shape
            qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
            q, k, v = qkv.permute(2, 0, 3, 1, 4)
            
            q = apply_rope(q, rope_cos, rope_sin)
            k = apply_rope(k, rope_cos, rope_sin)
            
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            y = y.transpose(1, 2).reshape(B, T, C)
            return self.proj(y)

    class MLP(nn.Module):
        def __init__(self, dim):
            super().__init__()
            hidden = int(dim * 8 / 3)
            hidden = ((hidden + 63) // 64) * 64
            self.w1 = nn.Linear(dim, hidden, bias=False)
            self.w2 = nn.Linear(hidden, dim, bias=False)
            self.w3 = nn.Linear(dim, hidden, bias=False)

        def forward(self, x):
            return self.w2(F.silu(self.w1(x)) * self.w3(x))

    class PLEBlock(nn.Module):
        """带 PLE 的 Transformer Block"""
        def __init__(self, dim, n_heads, ple_proj):
            super().__init__()
            self.ln1 = nn.LayerNorm(dim)
            self.attn = Attention(dim, n_heads)
            self.ln2 = nn.LayerNorm(dim)
            self.mlp = MLP(dim)
            self.ple_proj = ple_proj  # 共享或独立的 projection

        def forward(self, x, ple_embed):
            # 注入 PLE
            x = x + self.ple_proj(ple_embed)
            # 标准 transformer
            x = x + self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
            return x

    class PLETransformer(nn.Module):
        def __init__(self, vocab_size, dim, n_layers, n_heads, ple_dim, ple_mode):
            super().__init__()
            self.dim = dim
            self.ple_mode = ple_mode
            
            # 主 embedding
            self.embed = nn.Embedding(vocab_size, dim)
            
            # PLE embedding (小维度)
            self.ple_embed = nn.Embedding(vocab_size, ple_dim)
            
            # PLE projection
            if ple_mode == "shared":
                # 所有层共享一个 projection
                ple_proj = nn.Linear(ple_dim, dim, bias=False)
                self.blocks = nn.ModuleList([
                    PLEBlock(dim, n_heads, ple_proj) for _ in range(n_layers)
                ])
            else:  # per_layer
                # 每层独立的 projection
                self.blocks = nn.ModuleList([
                    PLEBlock(dim, n_heads, nn.Linear(ple_dim, dim, bias=False))
                    for _ in range(n_layers)
                ])
            
            self.ln_f = nn.LayerNorm(dim)
            self.lm_head = nn.Linear(dim, vocab_size, bias=False)
            
            # Weight tying
            self.lm_head.weight = self.embed.weight
            
            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)
                elif isinstance(m, nn.Embedding):
                    nn.init.normal_(m.weight, std=0.02)

        def forward(self, idx, targets=None):
            B, T = idx.shape
            x = self.embed(idx)
            ple = self.ple_embed(idx)  # B, T, ple_dim
            
            for block in self.blocks:
                x = block(x, ple)
            
            x = self.ln_f(x)
            logits = self.lm_head(x)
            
            if targets is None:
                return logits, None
            
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss

        def count_parameters(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ========== 创建模型 ==========
    model = PLETransformer(vocab_size, dim, n_layers, n_heads, ple_dim, ple_mode).to(device)
    
    n_params = model.count_parameters()
    ple_params = model.ple_embed.weight.numel()
    if ple_mode == "shared":
        ple_proj_params = ple_dim * dim
    else:
        ple_proj_params = ple_dim * dim * n_layers
    
    print(f"\n模型: {n_params / 1e6:.2f}M params")
    print(f"  主 embedding: {model.embed.weight.numel() / 1e6:.2f}M")
    print(f"  PLE embedding: {ple_params / 1e6:.2f}M")
    print(f"  PLE projection: {ple_proj_params / 1e6:.2f}M")
    print(f"  PLE 总开销: {(ple_params + ple_proj_params) / 1e6:.2f}M ({100 * (ple_params + ple_proj_params) / n_params:.1f}%)")

    # ========== 优化器 ==========
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_max, betas=(0.9, 0.95), weight_decay=0.1)

    def get_lr(step):
        if step < warmup_steps:
            return lr_max * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / (n_steps - warmup_steps)
        return lr_max * 0.5 * (1 + math.cos(math.pi * progress))

    # ========== 训练循环 ==========
    def get_batch(split):
        data = train_tokens if split == 'train' else val_tokens
        max_start = len(data) - seq_len - 1
        ix = torch.randint(0, max_start, (batch_size,))
        x = torch.stack([data[i:i+seq_len] for i in ix]).long().to(device)
        y = torch.stack([data[i+1:i+seq_len+1] for i in ix]).long().to(device)
        return x, y

    @torch.no_grad()
    def eval_loss():
        model.eval()
        losses = []
        for _ in range(50):
            x, y = get_batch('val')
            _, loss = model(x, y)
            losses.append(loss.item())
        model.train()
        return sum(losses) / len(losses)

    print(f"\n🚀 开始训练 ({n_steps} steps)...\n")
    start_time = time.time()

    for step in range(n_steps):
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        x, y = get_batch('train')
        _, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (step + 1) % 500 == 0:
            elapsed = time.time() - start_time
            print(f"Step {step+1}/{n_steps} | Loss {loss.item():.4f} | LR {lr:.2e} | Time {elapsed:.0f}s")

    elapsed = time.time() - start_time
    print(f"\n训练完成: {elapsed:.0f}s")

    # ========== 最终评估 ==========
    val_loss = eval_loss()
    val_bpb = val_loss / math.log(2)

    print("\n" + "=" * 70)
    print(f"🏆 结果 (PLE)")
    print("=" * 70)
    print(f"  模式: {ple_mode}")
    print(f"  PLE dim: {ple_dim}")
    print(f"  模型: {n_layers} layers, dim={dim}")
    print(f"  参数: {n_params / 1e6:.2f}M")
    print(f"")
    print(f"  Val Loss: {val_loss:.4f}")
    print(f"  Val BPB:  {val_bpb:.4f}")
    print("=" * 70)

    return {
        "ple_mode": ple_mode,
        "ple_dim": ple_dim,
        "n_layers": n_layers,
        "dim": dim,
        "n_params": n_params,
        "val_loss": val_loss,
        "val_bpb": val_bpb,
        "training_time": elapsed,
    }


@app.local_entrypoint()
def main():
    # 先跑 shared 模式
    result = train_ple.remote(
        n_layers=11,
        dim=416,
        ple_dim=64,
        ple_mode="shared",
        n_steps=5000,
        seed=42,
    )
    
    print(f"\n🏁 BPB: {result['val_bpb']:.4f}")
    print(f"📊 PLE mode: {result['ple_mode']}, dim: {result['ple_dim']}")

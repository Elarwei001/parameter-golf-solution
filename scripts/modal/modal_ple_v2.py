"""
PLE (Per-Layer Embedding) 实验
基于 BASELINE_CONFIG.md 配置

核心思路: 每层都注入 token embedding 信号，让深层也能"回看"原始 token 身份
"""
import modal
import os
import math

app = modal.App("ple-experiment-v2")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch==2.5.1",
        "numpy",
    ])
)

data_volume = modal.Volume.from_name("parameter-golf-data", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={"/data": data_volume},
    timeout=7200,
)
def train_ple(
    seed: int = 42,
    # 架构参数 - 与 baseline 一致
    dim: int = 384,
    n_layers: int = 20,
    n_heads: int = 8,
    n_kv_heads: int = 4,
    window_size: int = 192,
    # PLE 参数
    ple_dim: int = 64,  # PLE embedding 维度
    ple_mode: str = "shared",  # shared 或 per_layer
    # 训练参数 - 与 baseline 一致
    lr: float = 1e-3,
    batch_size: int = 64,
    seq_len: int = 256,
    steps: int = 5000,
):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import time
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    VOCAB_SIZE = 8192
    DATA_DIR = "/data/datasets/fineweb10B_sp8192"
    HEADER_SIZE = 256 * 4
    
    print("="*70)
    print(f"PLE 实验: 每层注入 token embedding")
    print(f"配置: {n_layers} 层, dim={dim}, PLE dim={ple_dim}, mode={ple_mode}")
    print("="*70)
    
    # ========== 数据加载 (与 baseline 完全一致) ==========
    print("\n加载数据...")
    train_files = sorted([f for f in os.listdir(DATA_DIR) if 'train' in f])
    val_files = sorted([f for f in os.listdir(DATA_DIR) if 'val' in f])
    
    train_data = []
    for f in train_files[:5]:  # 前5个shard = ~500M tokens
        with open(os.path.join(DATA_DIR, f), 'rb') as fp:
            fp.seek(HEADER_SIZE)
            data = np.frombuffer(fp.read(), dtype=np.uint16)
        train_data.append(data)
    train_data = np.concatenate(train_data)
    
    val_data = []
    for f in val_files:
        with open(os.path.join(DATA_DIR, f), 'rb') as fp:
            fp.seek(HEADER_SIZE)
            data = np.frombuffer(fp.read(), dtype=np.uint16)
        val_data.append(data)
    val_data = np.concatenate(val_data)
    
    print(f"  Train: {len(train_data)/1e6:.1f}M tokens")
    print(f"  Val: {len(val_data)/1e6:.1f}M tokens")
    
    # ========== 模型定义 ==========
    class RMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))
        
        def forward(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
    
    class RotaryEmbedding(nn.Module):
        def __init__(self, dim, max_seq_len=4096):
            super().__init__()
            inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
            t = torch.arange(max_seq_len)
            freqs = torch.outer(t, inv_freq)
            self.register_buffer('cos', freqs.cos())
            self.register_buffer('sin', freqs.sin())
        
        def forward(self, x, offset=0):
            seq_len = x.shape[1]
            cos = self.cos[offset:offset+seq_len].unsqueeze(0).unsqueeze(2)
            sin = self.sin[offset:offset+seq_len].unsqueeze(0).unsqueeze(2)
            x1, x2 = x[..., ::2], x[..., 1::2]
            return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    
    class AttentionXSA(nn.Module):
        """与 baseline 一致的 Attention"""
        def __init__(self, dim, n_heads, n_kv_heads=None, window_size=128):
            super().__init__()
            self.n_heads = n_heads
            self.n_kv_heads = n_kv_heads or n_heads
            self.head_dim = dim // n_heads
            self.window_size = window_size
            
            self.wq = nn.Linear(dim, dim, bias=False)
            self.wk = nn.Linear(dim, self.head_dim * self.n_kv_heads, bias=False)
            self.wv = nn.Linear(dim, self.head_dim * self.n_kv_heads, bias=False)
            self.wo = nn.Linear(dim, dim, bias=False)
            self.rope = RotaryEmbedding(self.head_dim)
        
        def forward(self, x):
            B, T, C = x.shape
            
            q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
            k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
            v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)
            
            q = self.rope(q)
            k = self.rope(k)
            
            if self.n_kv_heads < self.n_heads:
                k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=2)
                v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=2)
            
            q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
            
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            
            # XSA: 去掉对角线
            diag_mask = torch.eye(T, device=x.device, dtype=torch.bool)
            scores = scores.masked_fill(diag_mask, 0.0)
            
            # Causal + window mask
            rows = torch.arange(T, device=x.device).view(-1, 1)
            cols = torch.arange(T, device=x.device).view(1, -1)
            causal_mask = cols > rows
            window_mask = (rows - cols) > self.window_size
            mask = causal_mask | window_mask
            scores = scores.masked_fill(mask, float('-inf'))
            
            attn = F.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)
            out = out.transpose(1, 2).contiguous().view(B, T, C)
            return self.wo(out)
    
    class MLP(nn.Module):
        """与 baseline 一致的 MLP"""
        def __init__(self, dim, hidden_dim=None):
            super().__init__()
            hidden_dim = hidden_dim or dim * 3
            self.w1 = nn.Linear(dim, hidden_dim, bias=False)
            self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        
        def forward(self, x):
            h = self.w1(x)
            h = F.leaky_relu(h, 0.5) ** 2
            return self.w2(h)
    
    class PLEBlock(nn.Module):
        """带 PLE 的 Transformer Block"""
        def __init__(self, dim, n_heads, n_kv_heads, window_size, ple_proj):
            super().__init__()
            self.attn = AttentionXSA(dim, n_heads, n_kv_heads, window_size)
            self.mlp = MLP(dim)
            self.ln1 = RMSNorm(dim)
            self.ln2 = RMSNorm(dim)
            self.ple_proj = ple_proj  # 共享或独立的 PLE projection
        
        def forward(self, x, ple_embed):
            # 注入 PLE
            x = x + self.ple_proj(ple_embed)
            # 标准 Transformer 残差
            x = x + self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
            return x
    
    class PLETransformer(nn.Module):
        def __init__(self, vocab_size, dim, n_layers, n_heads, n_kv_heads, window_size, ple_dim, ple_mode):
            super().__init__()
            # 主 embedding
            self.embed = nn.Embedding(vocab_size, dim)
            # PLE embedding (较小)
            self.ple_embed = nn.Embedding(vocab_size, ple_dim)
            
            # PLE projection
            if ple_mode == "shared":
                # 所有层共享一个 projection
                shared_proj = nn.Linear(ple_dim, dim, bias=False)
                self.blocks = nn.ModuleList([
                    PLEBlock(dim, n_heads, n_kv_heads, window_size, shared_proj)
                    for _ in range(n_layers)
                ])
            else:
                # 每层独立 projection
                self.blocks = nn.ModuleList([
                    PLEBlock(dim, n_heads, n_kv_heads, window_size, 
                             nn.Linear(ple_dim, dim, bias=False))
                    for _ in range(n_layers)
                ])
            
            self.ln_f = RMSNorm(dim)
            self.lm_head = nn.Linear(dim, vocab_size, bias=False)
            self.embed.weight = self.lm_head.weight  # Weight tying
        
        def forward(self, idx, targets=None):
            x = self.embed(idx)
            ple = self.ple_embed(idx)  # PLE embedding
            
            for block in self.blocks:
                x = block(x, ple)
            
            x = self.ln_f(x)
            logits = self.lm_head(x)
            
            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            return logits, loss
    
    # 创建模型
    model = PLETransformer(
        vocab_size=VOCAB_SIZE,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        window_size=window_size,
        ple_dim=ple_dim,
        ple_mode=ple_mode,
    ).to(device)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    embed_params = model.embed.weight.numel()
    ple_embed_params = model.ple_embed.weight.numel()
    
    # PLE projection 参数
    if ple_mode == "shared":
        ple_proj_params = ple_dim * dim
    else:
        ple_proj_params = n_layers * ple_dim * dim
    
    print(f"\n模型: {total_params/1e6:.2f}M params")
    print(f"  主 embedding: {embed_params/1e6:.2f}M")
    print(f"  PLE embedding: {ple_embed_params/1e6:.2f}M")
    print(f"  PLE projection: {ple_proj_params/1e6:.2f}M ({ple_mode})")
    print(f"  PLE 总开销: {(ple_embed_params + ple_proj_params)/1e6:.2f}M ({(ple_embed_params + ple_proj_params)/total_params*100:.1f}%)")
    
    # ========== 训练 (与 baseline 完全一致) ==========
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    
    def get_batch(split):
        data = train_data if split == 'train' else val_data
        max_start = len(data) - seq_len - 1
        ix = np.random.randint(0, max_start, (batch_size,))
        x = torch.from_numpy(np.stack([data[i:i+seq_len] for i in ix]).astype(np.int64)).to(device)
        y = torch.from_numpy(np.stack([data[i+1:i+seq_len+1] for i in ix]).astype(np.int64)).to(device)
        return x, y
    
    def cosine_lr(step):
        warmup = 200
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / (steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    print(f"\n🚀 开始训练 ({steps} steps)...\n")
    start_time = time.time()
    
    for step in range(1, steps + 1):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * cosine_lr(step)
        
        x, y = get_batch('train')
        _, loss = model(x, y)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % 500 == 0:
            elapsed = time.time() - start_time
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Step {step}/{steps} | Loss {loss.item():.4f} | LR {current_lr:.2e} | Time {elapsed:.0f}s")
    
    print(f"\n训练完成: {time.time() - start_time:.0f}s")
    
    # ========== 验证 ==========
    model.eval()
    val_losses = []
    with torch.no_grad():
        for _ in range(100):
            x, y = get_batch('val')
            _, loss = model(x, y)
            val_losses.append(loss.item())
    
    val_loss = sum(val_losses) / len(val_losses)
    val_bpb = val_loss / math.log(2)
    
    print("\n" + "="*70)
    print("🏆 结果 (PLE)")
    print("="*70)
    print(f"  PLE mode: {ple_mode}")
    print(f"  PLE dim: {ple_dim}")
    print(f"  模型: {n_layers} layers, dim={dim}")
    print(f"  参数: {total_params/1e6:.2f}M")
    print(f"")
    print(f"  Val Loss: {val_loss:.4f}")
    print(f"  Val BPB:  {val_bpb:.4f}")
    print(f"")
    print(f"  Baseline BPB: 1.5025")
    print(f"  差异: {(val_bpb - 1.5025) / 1.5025 * 100:+.2f}%")
    print("="*70)
    
    return {
        "ple_mode": ple_mode,
        "ple_dim": ple_dim,
        "params_m": total_params / 1e6,
        "val_loss": val_loss,
        "val_bpb": val_bpb,
    }


@app.local_entrypoint()
def main():
    print("\n🚀 启动 PLE 实验 (使用 BASELINE_CONFIG)...")
    
    result = train_ple.remote(
        n_layers=20,
        dim=384,
        n_heads=8,
        n_kv_heads=4,
        window_size=192,
        ple_dim=64,
        ple_mode="shared",
        seed=42,
        steps=5000,
    )
    
    print(f"\n🏁 BPB: {result['val_bpb']:.4f}")
    print(f"🏁 vs Baseline: {(result['val_bpb'] - 1.5025) / 1.5025 * 100:+.2f}%")

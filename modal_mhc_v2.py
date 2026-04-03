"""
mHC v2: 双参数可学习残差
y = α * x + β * layer_out
观察 α (残差权重) 和 β (层输出权重) 随层数的变化
"""
import modal
import os
import math

app = modal.App("mhc-v2-study")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.0",
        "numpy",
    ])
)

data_volume = modal.Volume.from_name("parameter-golf-data", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={"/data": data_volume},
    timeout=3600,
)
def train_mhc_v2(
    seed: int = 42,
    # 架构参数
    dim: int = 416,
    n_layers: int = 11,
    n_heads: int = 8,
    n_kv_heads: int = 4,
    window_size: int = 192,
    # 训练参数
    lr: float = 1e-3,
    batch_size: int = 64,
    seq_len: int = 256,
    steps: int = 5000,
    # 记录间隔
    log_alpha_every: int = 500,
):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import time
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device("cuda")
    VOCAB_SIZE = 8192
    DATA_DIR = "/data/datasets/fineweb10B_sp8192"
    HEADER_SIZE = 256 * 4
    
    print("="*70)
    print("mHC v2 双参数实验: y = α*x + β*layer_out")
    print("="*70)
    
    # 加载数据
    print("\n加载数据...")
    train_files = sorted([f for f in os.listdir(DATA_DIR) if 'train' in f])
    val_files = sorted([f for f in os.listdir(DATA_DIR) if 'val' in f])
    
    train_data = []
    for f in train_files[:5]:
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
    
    # 模型定义
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
            
            # XSA: 排除自己
            diag_mask = torch.eye(T, device=x.device, dtype=torch.bool)
            scores = scores.masked_fill(diag_mask, 0.0)
            
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
        def __init__(self, dim, hidden_dim=None):
            super().__init__()
            hidden_dim = hidden_dim or dim * 3
            self.w1 = nn.Linear(dim, hidden_dim, bias=False)
            self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        
        def forward(self, x):
            h = self.w1(x)
            h = F.leaky_relu(h, 0.5) ** 2
            return self.w2(h)
    
    class mHCv2Block(nn.Module):
        """mHC v2: y = α*x + β*layer_out (双参数)"""
        def __init__(self, dim, n_heads, n_kv_heads, window_size, layer_idx):
            super().__init__()
            self.layer_idx = layer_idx
            
            self.attn = AttentionXSA(dim, n_heads, n_kv_heads, window_size)
            self.mlp = MLP(dim)
            self.ln1 = RMSNorm(dim)
            self.ln2 = RMSNorm(dim)
            
            # 双参数 mHC：α (残差权重) 和 β (层输出权重)
            # 初始化为标准残差: α=1, β=1
            self.alpha_attn = nn.Parameter(torch.ones(1))
            self.beta_attn = nn.Parameter(torch.ones(1))
            self.alpha_mlp = nn.Parameter(torch.ones(1))
            self.beta_mlp = nn.Parameter(torch.ones(1))
        
        def forward(self, x):
            # Attention: y = α*x + β*attn(x)
            attn_out = self.attn(self.ln1(x))
            x = self.alpha_attn * x + self.beta_attn * attn_out
            
            # MLP: y = α*x + β*mlp(x)
            mlp_out = self.mlp(self.ln2(x))
            x = self.alpha_mlp * x + self.beta_mlp * mlp_out
            
            return x
        
        def get_mhc_params(self):
            return {
                'alpha_attn': self.alpha_attn.item(),
                'beta_attn': self.beta_attn.item(),
                'alpha_mlp': self.alpha_mlp.item(),
                'beta_mlp': self.beta_mlp.item(),
            }
    
    class mHCv2GPT(nn.Module):
        def __init__(self, vocab_size, dim, n_layers, n_heads, n_kv_heads, window_size):
            super().__init__()
            self.vocab_size = vocab_size
            
            self.tok_emb = nn.Embedding(vocab_size, dim)
            self.layers = nn.ModuleList([
                mHCv2Block(dim, n_heads, n_kv_heads, window_size, i)
                for i in range(n_layers)
            ])
            self.ln_f = RMSNorm(dim)
            self.head = nn.Linear(dim, vocab_size, bias=False)
        
        def forward(self, idx):
            x = self.tok_emb(idx)
            for layer in self.layers:
                x = layer(x)
            x = self.ln_f(x)
            return self.head(x)
        
        def loss(self, idx):
            logits = self(idx[:, :-1])
            targets = idx[:, 1:]
            return F.cross_entropy(logits.reshape(-1, self.vocab_size),
                                   targets.reshape(-1))
        
        def get_all_mhc_params(self):
            return [layer.get_mhc_params() for layer in self.layers]
    
    # 创建模型
    model = mHCv2GPT(
        vocab_size=VOCAB_SIZE,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        window_size=window_size,
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    mhc_params = n_layers * 4  # 每层 4 个参数
    
    print(f"\n模型: {n_params/1e6:.2f}M params")
    print(f"mHC 参数: {mhc_params} (每层 α_attn, β_attn, α_mlp, β_mlp)")
    
    def get_batch(data, batch_size=batch_size, seq_len=seq_len):
        starts = np.random.randint(0, len(data) - seq_len - 1, batch_size)
        batch = np.stack([data[i:i+seq_len+1] for i in starts])
        return torch.from_numpy(batch.astype(np.int64)).to(device)
    
    # 记录 α/β 变化
    mhc_history = []
    
    # 训练
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    
    start_time = time.time()
    LOG_EVERY = 500
    
    print(f"\n🚀 开始训练 ({steps} steps)...")
    print(f"   每 {log_alpha_every} 步记录 α/β 值\n")
    
    for step in range(1, steps + 1):
        model.train()
        
        # LR schedule
        if step < 100:
            lr_mult = step / 100
        else:
            progress = (step - 100) / (steps - 100)
            lr_mult = 0.5 * (1 + math.cos(math.pi * progress))
        
        for pg in opt.param_groups:
            pg['lr'] = lr * lr_mult
        
        batch = get_batch(train_data)
        loss = model.loss(batch)
        
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        
        if step % log_alpha_every == 0:
            mhc_params_current = model.get_all_mhc_params()
            mhc_history.append({
                'step': step,
                'params': mhc_params_current
            })
        
        if step % LOG_EVERY == 0:
            elapsed = time.time() - start_time
            print(f"Step {step}/{steps} | Loss {loss.item():.4f} | "
                  f"LR {lr * lr_mult:.2e} | Time {elapsed:.0f}s")
    
    train_time = time.time() - start_time
    print(f"\n训练完成: {train_time:.0f}s")
    
    # 评估
    BYTES_PER_TOKEN = 3.67
    
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for _ in range(100):
            batch = get_batch(val_data)
            loss = model.loss(batch)
            total_loss += loss.item() * (batch.shape[0] * batch.shape[1])
            total_tokens += batch.shape[0] * batch.shape[1]
    
    avg_loss = total_loss / total_tokens
    bpb = (avg_loss / math.log(2)) * (1.0 / BYTES_PER_TOKEN)
    
    # 打印最终 α/β 值
    print(f"\n{'='*70}")
    print("最终学习到的 α/β 值 (y = α*x + β*layer_out)")
    print("="*70)
    print(f"{'Layer':<8} {'α_attn':>8} {'β_attn':>8} {'α_mlp':>8} {'β_mlp':>8} | {'α+β (attn)':>10} {'α+β (mlp)':>10}")
    print("-"*70)
    
    final_params = model.get_all_mhc_params()
    for i, p in enumerate(final_params):
        sum_attn = p['alpha_attn'] + p['beta_attn']
        sum_mlp = p['alpha_mlp'] + p['beta_mlp']
        print(f"Layer {i:<2} {p['alpha_attn']:>8.3f} {p['beta_attn']:>8.3f} "
              f"{p['alpha_mlp']:>8.3f} {p['beta_mlp']:>8.3f} | "
              f"{sum_attn:>10.3f} {sum_mlp:>10.3f}")
    
    # 分析趋势
    print(f"\n{'='*70}")
    print("层级趋势分析")
    print("="*70)
    
    # 计算浅层 (0-3) vs 深层 (7-10) 的平均值
    shallow_alpha_attn = np.mean([final_params[i]['alpha_attn'] for i in range(4)])
    shallow_beta_attn = np.mean([final_params[i]['beta_attn'] for i in range(4)])
    deep_alpha_attn = np.mean([final_params[i]['alpha_attn'] for i in range(7, 11)])
    deep_beta_attn = np.mean([final_params[i]['beta_attn'] for i in range(7, 11)])
    
    shallow_alpha_mlp = np.mean([final_params[i]['alpha_mlp'] for i in range(4)])
    shallow_beta_mlp = np.mean([final_params[i]['beta_mlp'] for i in range(4)])
    deep_alpha_mlp = np.mean([final_params[i]['alpha_mlp'] for i in range(7, 11)])
    deep_beta_mlp = np.mean([final_params[i]['beta_mlp'] for i in range(7, 11)])
    
    print(f"\nAttention:")
    print(f"  浅层 (0-3):  α={shallow_alpha_attn:.3f}, β={shallow_beta_attn:.3f}")
    print(f"  深层 (7-10): α={deep_alpha_attn:.3f}, β={deep_beta_attn:.3f}")
    
    print(f"\nMLP:")
    print(f"  浅层 (0-3):  α={shallow_alpha_mlp:.3f}, β={shallow_beta_mlp:.3f}")
    print(f"  深层 (7-10): α={deep_alpha_mlp:.3f}, β={deep_beta_mlp:.3f}")
    
    print(f"\n{'='*70}")
    print(f"🏆 结果 (mHC v2)")
    print(f"{'='*70}")
    print(f"  模型: {n_layers} layers, dim={dim}")
    print(f"  参数: {n_params/1e6:.2f}M")
    print(f"")
    print(f"  Val Loss: {avg_loss:.4f}")
    print(f"  Val BPB:  {bpb:.4f}")
    print(f"{'='*70}")
    
    # 保存
    checkpoint_dir = "/data/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(checkpoint_dir, f"mhc_v2_bpb{bpb:.4f}.pt")
    torch.save({
        'state_dict': model.state_dict(),
        'config': {'dim': dim, 'n_layers': n_layers},
        'mhc_params': final_params,
        'mhc_history': mhc_history,
        'bpb': bpb,
    }, checkpoint_path)
    
    data_volume.commit()
    
    print(f"\n💾 保存: {checkpoint_path}")
    
    return {
        'bpb': bpb,
        'loss': avg_loss,
        'n_params': n_params,
        'final_mhc_params': final_params,
        'mhc_history': mhc_history,
    }


@app.local_entrypoint()
def main(seed: int = 42):
    print("="*70)
    print("mHC v2 双参数实验: y = α*x + β*layer_out")
    print("="*70)
    result = train_mhc_v2.remote(seed=seed)
    if result:
        print(f"\n🏁 BPB: {result['bpb']:.4f}")

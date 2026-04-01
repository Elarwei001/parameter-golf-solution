"""
Train model and analyze embedding space efficiency

Usage:
    modal run modal_analyze_embedding.py::train_and_analyze --steps 2000
"""
import modal
import os

app = modal.App("parameter-golf-embedding-analysis")

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
    gpu="H100",
    volumes={"/data": data_volume},
    timeout=1200,
)
def train_and_analyze(steps: int = 2000):
    """Train GPT and analyze embedding space"""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import math
    
    DEVICE = torch.device("cuda")
    VOCAB_SIZE = 8192
    DIM = 512
    N_LAYERS = 9
    N_HEADS = 8
    N_KV_HEADS = 4
    SEQ_LEN = 256
    BATCH_SIZE = 64
    DATA_DIR = "/data/datasets/fineweb10B_sp8192"
    
    print("="*60)
    print("Training model to analyze embedding space")
    print(f"Config: vocab={VOCAB_SIZE}, dim={DIM}, layers={N_LAYERS}")
    print("="*60)
    
    # ── Load Data ─────────────────────────────────────────────
    train_files = sorted([f for f in os.listdir(DATA_DIR) if 'train' in f])
    train_data = []
    for f in train_files[:5]:
        path = os.path.join(DATA_DIR, f)
        data = np.fromfile(path, dtype=np.uint16)
        train_data.append(data)
    train_data = np.concatenate(train_data)
    print(f"Loaded {len(train_data)/1e6:.1f}M tokens")
    
    def get_batch():
        starts = np.random.randint(0, len(train_data) - SEQ_LEN - 1, BATCH_SIZE)
        batch = np.stack([train_data[i:i+SEQ_LEN+1] for i in starts])
        return torch.from_numpy(batch.astype(np.int64)).to(DEVICE)
    
    # ── Simple GPT Model ──────────────────────────────────────
    class RMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))
        def forward(self, x):
            rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
            return x / rms * self.weight
    
    class SimpleAttention(nn.Module):
        def __init__(self, dim, n_heads):
            super().__init__()
            self.n_heads = n_heads
            self.head_dim = dim // n_heads
            self.wqkv = nn.Linear(dim, 3 * dim, bias=False)
            self.wo = nn.Linear(dim, dim, bias=False)
        
        def forward(self, x):
            B, L, _ = x.shape
            qkv = self.wqkv(x).chunk(3, dim=-1)
            q, k, v = [t.view(B, L, self.n_heads, self.head_dim).transpose(1,2) for t in qkv]
            
            scale = self.head_dim ** -0.5
            attn = (q @ k.transpose(-2,-1)) * scale
            mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
            attn = attn.masked_fill(mask, float('-inf'))
            attn = F.softmax(attn, dim=-1)
            out = (attn @ v).transpose(1,2).reshape(B, L, -1)
            return self.wo(out)
    
    class MLP(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.w1 = nn.Linear(dim, 4 * dim, bias=False)
            self.w2 = nn.Linear(4 * dim, dim, bias=False)
        def forward(self, x):
            return self.w2(F.gelu(self.w1(x)))
    
    class Block(nn.Module):
        def __init__(self, dim, n_heads):
            super().__init__()
            self.attn = SimpleAttention(dim, n_heads)
            self.mlp = MLP(dim)
            self.norm1 = RMSNorm(dim)
            self.norm2 = RMSNorm(dim)
        def forward(self, x):
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
            return x
    
    class GPT(nn.Module):
        def __init__(self, vocab_size, dim, n_layers, n_heads):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, dim)
            self.layers = nn.ModuleList([Block(dim, n_heads) for _ in range(n_layers)])
            self.norm = RMSNorm(dim)
            self.head = nn.Linear(dim, vocab_size, bias=False)
            self.tok_emb.weight = self.head.weight
        
        def forward(self, idx):
            x = self.tok_emb(idx)
            for layer in self.layers:
                x = layer(x)
            return self.head(self.norm(x))
    
    model = GPT(VOCAB_SIZE, DIM, N_LAYERS, N_HEADS).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params/1e6:.1f}M")
    
    # ── Analyze BEFORE training (random init) ─────────────────
    def analyze_embeddings(emb_matrix, name):
        """Analyze embedding space efficiency"""
        emb = emb_matrix.detach().cpu().numpy()
        
        # Center
        centered = emb - emb.mean(axis=0)
        
        # SVD
        U, S, Vh = np.linalg.svd(centered, full_matrices=False)
        
        # Participation Ratio
        s2 = S ** 2
        s4 = S ** 4
        pr = (s2.sum() ** 2) / (s4.sum() + 1e-10)
        
        # Explained variance
        cumvar = np.cumsum(s2) / s2.sum()
        dims_95 = np.searchsorted(cumvar, 0.95) + 1
        
        # Anisotropy
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        normalized = emb / (norms + 1e-10)
        n = min(1000, len(emb))
        idx1 = np.random.choice(len(emb), n, replace=False)
        idx2 = np.random.choice(len(emb), n, replace=False)
        cos_sims = (normalized[idx1] * normalized[idx2]).sum(axis=1)
        anisotropy = cos_sims.mean()
        
        print(f"\n{'='*50}")
        print(f"📊 {name}")
        print(f"{'='*50}")
        print(f"Shape: {emb.shape}")
        print(f"Participation Ratio: {pr:.1f} / {emb.shape[1]} ({pr/emb.shape[1]*100:.1f}% efficiency)")
        print(f"Dims for 95% variance: {dims_95}")
        print(f"Anisotropy: {anisotropy:.4f}")
        print(f"Top 5 singular values: {S[:5].round(2)}")
        
        return {
            'pr': pr,
            'dims_95': dims_95,
            'anisotropy': anisotropy,
            'singular_values': S[:50].tolist(),
        }
    
    print("\n" + "="*60)
    print("BEFORE TRAINING (Random Initialization)")
    print("="*60)
    before = analyze_embeddings(model.tok_emb.weight, "Random Init Embeddings")
    
    # ── Train ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"Training for {steps} steps...")
    print("="*60)
    
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    
    for step in range(1, steps + 1):
        batch = get_batch()
        logits = model(batch[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), batch[:, 1:].reshape(-1))
        
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()
        
        if step % 500 == 0:
            print(f"Step {step}/{steps} | Loss: {loss.item():.4f}")
    
    # ── Analyze AFTER training ────────────────────────────────
    print("\n" + "="*60)
    print("AFTER TRAINING")
    print("="*60)
    after = analyze_embeddings(model.tok_emb.weight, "Trained Embeddings")
    
    # ── Summary ───────────────────────────────────────────────
    print("\n" + "="*60)
    print("📈 SUMMARY: How Training Changed the Embedding Space")
    print("="*60)
    print(f"""
                        Before      After       Change
Participation Ratio:    {before['pr']:.1f}        {after['pr']:.1f}        {after['pr'] - before['pr']:+.1f}
Dims for 95% var:       {before['dims_95']}          {after['dims_95']}          {after['dims_95'] - before['dims_95']:+d}
Anisotropy:             {before['anisotropy']:.4f}      {after['anisotropy']:.4f}      {after['anisotropy'] - before['anisotropy']:+.4f}
    """)
    
    if after['pr'] < before['pr'] * 0.5:
        print("⚠️  WARNING: Training collapsed the embedding space significantly!")
        print(f"   Only using {after['pr']:.0f} of {DIM} dimensions effectively.")
        print(f"   Potential optimization: Try dim={int(after['pr']*1.2)} with whitening.")
    elif after['anisotropy'] > 0.3:
        print("⚠️  WARNING: High anisotropy detected!")
        print("   Embeddings are clustered, not using full space.")
        print("   Potential optimization: Apply whitening transformation.")
    else:
        print("✅ Embedding space looks reasonably efficient!")
    
    return {
        'before': before,
        'after': after,
    }


@app.local_entrypoint()
def main():
    result = train_and_analyze.remote(steps=2000)
    print("\nDone!")

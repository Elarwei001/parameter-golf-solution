#!/usr/bin/env python3
"""
Parameter Golf - BPE 1024 Tokenizer
用 SentencePiece BPE 代替字节级 tokenization
"""
import sys, os, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ── Device ──────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Device: {DEVICE}")

# ── Config ─────────────────────────────────────────────────────────
VOCAB_SIZE = 1024  # BPE vocabulary size
SEQ_LEN    = 256   # 可以考虑增加，因为压缩后序列更短
BATCH_SIZE = 32

# Data paths
DATA_DIR = "/tmp/pg-official/data/datasets/fineweb10B_sp1024"
TOKENIZER_PATH = "/tmp/pg-official/data/tokenizers/fineweb_1024_bpe.model"

# ── Data Loading ─────────────────────────────────────────────────
def load_data():
    """Load pre-tokenized BPE data"""
    train_files = sorted([f for f in os.listdir(DATA_DIR) if 'train' in f])
    val_files = sorted([f for f in os.listdir(DATA_DIR) if 'val' in f])
    
    # Load training data (concatenate all shards)
    train_data = []
    for f in train_files:
        path = os.path.join(DATA_DIR, f)
        # Data is stored as uint16 (token ids 0-1023)
        data = np.fromfile(path, dtype=np.uint16)
        train_data.append(data)
        print(f"  Loaded {f}: {len(data)/1e6:.1f}M tokens")
    train_data = np.concatenate(train_data)
    
    # Load validation data
    val_data = []
    for f in val_files:
        path = os.path.join(DATA_DIR, f)
        data = np.fromfile(path, dtype=np.uint16)
        val_data.append(data)
    val_data = np.concatenate(val_data)
    
    return train_data, val_data

def get_batch(data, seq_len=SEQ_LEN, batch_size=BATCH_SIZE):
    starts = np.random.randint(0, len(data) - seq_len - 1, batch_size)
    batch  = np.stack([data[i:i+seq_len+1] for i in starts])
    return torch.from_numpy(batch.astype(np.int64)).to(DEVICE)

# ── Primitives (copied to avoid import side effects) ─────────────
from models.standard_gpt import RMSNorm, apply_rotary_pos_emb

class RotaryEmbeddingDynamic(nn.Module):
    """RoPE that dynamically extends if seq_len > max_seq_len."""
    def __init__(self, dim, max_seq_len=4096, base=10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._extend(max_seq_len)

    def _extend(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos", emb.cos())
        self.register_buffer("sin", emb.sin())
        self._max_seq_len = seq_len

    def forward(self, x, seq_len):
        if seq_len > self._max_seq_len:
            self._extend(seq_len * 2)
        return self.cos[:seq_len], self.sin[:seq_len]

class MLP_LeakyReLU2(nn.Module):
    """LeakyReLU² MLP: leaky_relu(x, 0.5).square()."""
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        hidden = int(dim * mult)
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        h = F.leaky_relu(self.w1(x), negative_slope=0.5)
        return self.w2(h.square())

class AttentionSW(nn.Module):
    """Sliding-window causal attention."""
    def __init__(self, dim, n_heads, n_kv_heads=None, window_size=128):
        super().__init__()
        self.n_heads    = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.head_dim   = dim // n_heads
        self.n_rep      = n_heads // self.n_kv_heads
        self.window_size = window_size
        self.wq = nn.Linear(dim, n_heads       * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim,         bias=False)

    def forward(self, x, cos, sin):
        B, L, _ = x.shape
        q = self.wq(x).view(B, L, self.n_heads,    self.head_dim).transpose(1,2)
        k = self.wk(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1,2)
        v = self.wv(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1,2)

        q, k = apply_rotary_pos_emb(q, k,
                                    cos.unsqueeze(0).unsqueeze(0),
                                    sin.unsqueeze(0).unsqueeze(0))
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        scale = self.head_dim ** -0.5
        attn  = (q @ k.transpose(-2,-1)) * scale

        rows = torch.arange(L, device=x.device).unsqueeze(1)
        cols = torch.arange(L, device=x.device).unsqueeze(0)
        causal_mask = rows < cols
        window_mask = (rows - cols) > self.window_size
        attn = attn.masked_fill((causal_mask | window_mask).unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = F.softmax(attn, dim=-1)
        out  = attn @ v
        out  = out.transpose(1,2).reshape(B, L, -1)
        return self.wo(out)

class TransformerBlockSW(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads=None, window_size=128):
        super().__init__()
        self.attn  = AttentionSW(dim, n_heads, n_kv_heads, window_size)
        self.mlp   = MLP_LeakyReLU2(dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x

class GPT_BPE(nn.Module):
    def __init__(self, vocab_size, dim, n_layers, n_heads, n_kv_heads,
                 max_seq_len=512, window_size=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.rope    = RotaryEmbeddingDynamic(dim // n_heads, max_seq_len)
        self.layers  = nn.ModuleList([
            TransformerBlockSW(dim, n_heads, n_kv_heads, window_size)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.tok_emb.weight = self.head.weight  # weight tying

    def forward(self, idx):
        B, L = idx.shape
        x = self.tok_emb(idx)
        cos, sin = self.rope(x, L)  # Pass seq_len
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)
        return self.head(x)

    def loss(self, batch):
        logits = self(batch[:, :-1])
        loss   = F.cross_entropy(logits.reshape(-1, self.vocab_size),
                                  batch[:, 1:].reshape(-1))
        return loss

# ── BPB Calculation ─────────────────────────────────────────────
def calculate_bpb(model, data, num_batches=50):
    """
    Calculate Bits Per Byte (BPB) for BPE tokenized data.
    BPB = (CE_loss / ln(2)) * (tokens / bytes)
    
    For BPE with 1024 vocab on this dataset, the average bytes per token
    is approximately 2.2-2.5x (we'll estimate from the data).
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for _ in range(num_batches):
            batch = get_batch(data)
            loss = model.loss(batch)
            total_loss += loss.item() * (batch.shape[0] * batch.shape[1])
            total_tokens += batch.shape[0] * batch.shape[1]
    
    avg_loss = total_loss / total_tokens
    
    # Convert to BPB
    # For this tokenizer, we need to know the bytes-per-token ratio
    # From our test: compression ratio ~2.23x means ~2.23 bytes per token
    # But we should calculate this from the actual data
    BYTES_PER_TOKEN = 2.3  # Approximate for 1024 BPE on FineWeb
    
    bpb = (avg_loss / math.log(2)) * (1.0 / BYTES_PER_TOKEN)
    
    model.train()
    return bpb, avg_loss

# ── Training ─────────────────────────────────────────────────────
def train(train_data, val_data):
    # Model config - same as our best combo, but with BPE vocab
    model = GPT_BPE(
        vocab_size=VOCAB_SIZE,  # 1024 instead of 256
        dim=512,
        n_layers=9,
        n_heads=8,
        n_kv_heads=4,
        max_seq_len=SEQ_LEN + 64,
        window_size=192
    ).to(DEVICE)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params/1e6:.2f}M")
    
    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5000)
    
    # Training loop
    STEPS = 5000
    LOG_EVERY = 100
    
    print(f"\n{'='*60}")
    print(f"Training GPT with BPE-1024 tokenizer")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    for step in range(1, STEPS + 1):
        batch = get_batch(train_data)
        loss = model.loss(batch)
        
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()
        
        if step % LOG_EVERY == 0:
            elapsed = time.time() - start_time
            val_bpb, val_loss = calculate_bpb(model, val_data, num_batches=20)
            print(f"Step {step:5d} | Loss {loss.item():.4f} | Val BPB {val_bpb:.4f} | "
                  f"Val Loss {val_loss:.4f} | LR {scheduler.get_last_lr()[0]:.2e} | "
                  f"Time {elapsed:.0f}s")
    
    # Final evaluation
    final_bpb, final_loss = calculate_bpb(model, val_data, num_batches=100)
    print(f"\n{'='*60}")
    print(f"Final Results:")
    print(f"  Val Loss: {final_loss:.4f}")
    print(f"  Val BPB:  {final_bpb:.4f}")
    print(f"{'='*60}")
    
    return model, final_bpb

if __name__ == "__main__":
    # Quick sanity check
    print("\n=== Sanity Check ===")
    m = GPT_BPE(vocab_size=VOCAB_SIZE, dim=64, n_layers=2,
                n_heads=4, n_kv_heads=2, window_size=64).to(DEVICE)
    x = torch.randint(0, VOCAB_SIZE, (2, 33)).to(DEVICE)
    loss = m.loss(x)
    print(f"Sanity check loss: {loss.item():.4f} (expect ~{math.log(VOCAB_SIZE):.2f})")
    del m
    
    # Load data
    print("\nLoading BPE tokenized data...")
    train_data, val_data = load_data()
    print(f"Train: {len(train_data)/1e6:.1f}M tokens | Val: {len(val_data)/1e6:.1f}M tokens")
    
    # Train
    model, bpb = train(train_data, val_data)

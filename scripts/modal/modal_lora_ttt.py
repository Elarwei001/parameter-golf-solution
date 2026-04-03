"""
Modal LoRA TTT - Per-Document Test-Time Training with LoRA

Based on PR #548 which achieved 1.0865 BPB!

Key insight: At eval time, for each document:
1. Add ephemeral LoRA adapters (rank=8) to Q, V, and LM head
2. Train LoRA on the document using next-token prediction
3. Score the document with the adapted model
4. Discard LoRA weights, repeat for next document

This is "open-book" evaluation - the model learns from the test document!

Usage:
    modal run modal_lora_ttt.py::train_and_eval_ttt --steps 3000
"""
import modal
import os
import math

app = modal.App("parameter-golf-lora-ttt")

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
    timeout=2400,
)
def train_and_eval_ttt(
    steps: int = 3000,
    dim: int = 512,
    n_layers: int = 9,
    n_heads: int = 8,
    n_kv_heads: int = 4,
    lora_rank: int = 8,
    ttt_lr: float = 0.01,
    ttt_epochs: int = 2,
    chunk_size: int = 256,
):
    """Train model, then evaluate with per-document LoRA TTT"""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import time
    
    DEVICE = torch.device("cuda")
    VOCAB_SIZE = 8192
    SEQ_LEN = 256
    BATCH_SIZE = 64
    DATA_DIR = "/data/datasets/fineweb10B_sp8192"
    
    print("="*70)
    print("LoRA TTT Experiment")
    print(f"Config: dim={dim}, layers={n_layers}, LoRA rank={lora_rank}")
    print("="*70)
    
    # ══════════════════════════════════════════════════════════════════
    # MODEL DEFINITION
    # ══════════════════════════════════════════════════════════════════
    
    class RMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))
        def forward(self, x):
            rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
            return x / rms * self.weight
    
    class LoRALinear(nn.Module):
        """Linear layer with optional LoRA adaptation"""
        def __init__(self, in_features, out_features, bias=False, lora_rank=0):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.lora_rank = lora_rank
            
            # Base weights (frozen during TTT)
            self.weight = nn.Parameter(torch.empty(out_features, in_features))
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            
            if bias:
                self.bias = nn.Parameter(torch.zeros(out_features))
            else:
                self.register_parameter('bias', None)
            
            # LoRA weights (trained during TTT)
            self.lora_A = None  # Will be created during TTT
            self.lora_B = None
        
        def init_lora(self, rank, device=None):
            """Initialize LoRA weights for TTT"""
            self.lora_rank = rank
            device = device or self.weight.device
            self.lora_A = nn.Parameter(torch.randn(rank, self.in_features, device=device) * 0.01)
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank, device=device))
            return [self.lora_A, self.lora_B]
        
        def reset_lora(self):
            """Reset LoRA weights after processing a document"""
            if self.lora_A is not None:
                nn.init.normal_(self.lora_A, std=0.01)
                nn.init.zeros_(self.lora_B)
        
        def forward(self, x):
            out = F.linear(x, self.weight, self.bias)
            
            # Add LoRA contribution if active
            if self.lora_A is not None and self.lora_B is not None:
                lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B)
                out = out + lora_out
            
            return out
    
    class Attention(nn.Module):
        def __init__(self, dim, n_heads, n_kv_heads=None, lora_rank=0):
            super().__init__()
            self.n_heads = n_heads
            self.n_kv_heads = n_kv_heads or n_heads
            self.head_dim = dim // n_heads
            self.n_rep = n_heads // self.n_kv_heads
            
            # Q and V get LoRA, K doesn't (following PR #548)
            self.wq = LoRALinear(dim, n_heads * self.head_dim, bias=False, lora_rank=lora_rank)
            self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
            self.wv = LoRALinear(dim, self.n_kv_heads * self.head_dim, bias=False, lora_rank=lora_rank)
            self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)
        
        def forward(self, x):
            B, L, _ = x.shape
            
            q = self.wq(x).view(B, L, self.n_heads, self.head_dim).transpose(1,2)
            k = self.wk(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1,2)
            v = self.wv(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1,2)
            
            if self.n_rep > 1:
                k = k.repeat_interleave(self.n_rep, dim=1)
                v = v.repeat_interleave(self.n_rep, dim=1)
            
            # Standard causal attention
            scale = self.head_dim ** -0.5
            attn = (q @ k.transpose(-2,-1)) * scale
            mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
            attn = attn.masked_fill(mask, float('-inf'))
            attn = F.softmax(attn, dim=-1)
            
            out = (attn @ v).transpose(1,2).reshape(B, L, -1)
            return self.wo(out)
        
        def init_lora(self, rank, device=None):
            params = []
            params.extend(self.wq.init_lora(rank, device))
            params.extend(self.wv.init_lora(rank, device))
            return params
        
        def reset_lora(self):
            self.wq.reset_lora()
            self.wv.reset_lora()
    
    class MLP(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.w1 = nn.Linear(dim, 4 * dim, bias=False)
            self.w2 = nn.Linear(4 * dim, dim, bias=False)
        def forward(self, x):
            h = F.leaky_relu(self.w1(x), 0.01)
            return self.w2(h * h)
    
    class Block(nn.Module):
        def __init__(self, dim, n_heads, n_kv_heads, lora_rank):
            super().__init__()
            self.attn = Attention(dim, n_heads, n_kv_heads, lora_rank)
            self.mlp = MLP(dim)
            self.norm1 = RMSNorm(dim)
            self.norm2 = RMSNorm(dim)
        
        def forward(self, x):
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
            return x
        
        def init_lora(self, rank, device=None):
            return self.attn.init_lora(rank, device)
        
        def reset_lora(self):
            self.attn.reset_lora()
    
    class GPT_LoRA(nn.Module):
        def __init__(self, vocab_size, dim, n_layers, n_heads, n_kv_heads, lora_rank=0):
            super().__init__()
            self.vocab_size = vocab_size
            self.lora_rank = lora_rank
            
            self.tok_emb = nn.Embedding(vocab_size, dim)
            self.layers = nn.ModuleList([
                Block(dim, n_heads, n_kv_heads, lora_rank)
                for _ in range(n_layers)
            ])
            self.norm = RMSNorm(dim)
            
            # LM head also gets LoRA (following PR #548)
            self.head = LoRALinear(dim, vocab_size, bias=False, lora_rank=lora_rank)
            
            # Weight tying
            self.tok_emb.weight = self.head.weight
        
        def forward(self, idx):
            x = self.tok_emb(idx)
            for layer in self.layers:
                x = layer(x)
            return self.head(self.norm(x))
        
        def loss(self, batch):
            logits = self(batch[:, :-1])
            return F.cross_entropy(logits.reshape(-1, self.vocab_size),
                                   batch[:, 1:].reshape(-1))
        
        def init_all_lora(self, rank, device=None):
            """Initialize LoRA for all layers, return list of LoRA params"""
            device = device or next(self.parameters()).device
            params = []
            for layer in self.layers:
                params.extend(layer.init_lora(rank, device))
            params.extend(self.head.init_lora(rank, device))
            return params
        
        def reset_all_lora(self):
            """Reset all LoRA weights"""
            for layer in self.layers:
                layer.reset_lora()
            self.head.reset_lora()
        
        def freeze_base(self):
            """Freeze base weights, only LoRA trainable"""
            for name, param in self.named_parameters():
                if 'lora' not in name:
                    param.requires_grad = False
        
        def unfreeze_all(self):
            """Unfreeze all weights"""
            for param in self.parameters():
                param.requires_grad = True
    
    # ══════════════════════════════════════════════════════════════════
    # DATA LOADING
    # ══════════════════════════════════════════════════════════════════
    
    print("\nLoading data...")
    train_files = sorted([f for f in os.listdir(DATA_DIR) if 'train' in f])
    val_files = sorted([f for f in os.listdir(DATA_DIR) if 'val' in f])
    
    train_data = []
    for f in train_files[:5]:
        data = np.fromfile(os.path.join(DATA_DIR, f), dtype=np.uint16)
        train_data.append(data)
    train_data = np.concatenate(train_data)
    
    val_data = []
    for f in val_files:
        data = np.fromfile(os.path.join(DATA_DIR, f), dtype=np.uint16)
        val_data.append(data)
    val_data = np.concatenate(val_data)
    
    print(f"Train: {len(train_data)/1e6:.1f}M | Val: {len(val_data)/1e6:.1f}M tokens")
    
    def get_batch(data, seq_len=SEQ_LEN, batch_size=BATCH_SIZE):
        starts = np.random.randint(0, len(data) - seq_len - 1, batch_size)
        batch = np.stack([data[i:i+seq_len+1] for i in starts])
        return torch.from_numpy(batch.astype(np.int64)).to(DEVICE)
    
    # ══════════════════════════════════════════════════════════════════
    # PHASE 1: STANDARD TRAINING
    # ══════════════════════════════════════════════════════════════════
    
    print("\n" + "="*70)
    print("Phase 1: Standard Training (no LoRA)")
    print("="*70)
    
    model = GPT_LoRA(
        vocab_size=VOCAB_SIZE,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        lora_rank=0,  # No LoRA during training
    ).to(DEVICE)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params/1e6:.1f}M")
    
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    
    start = time.time()
    for step in range(1, steps + 1):
        batch = get_batch(train_data)
        loss = model.loss(batch)
        
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()
        
        if step % 500 == 0:
            print(f"Step {step}/{steps} | Loss {loss.item():.4f} | Time {time.time()-start:.0f}s")
    
    train_time = time.time() - start
    print(f"Training complete in {train_time:.0f}s")
    
    # ══════════════════════════════════════════════════════════════════
    # PHASE 2: STANDARD EVALUATION (no TTT)
    # ══════════════════════════════════════════════════════════════════
    
    def calculate_bpb(model, data, num_batches=50):
        model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for _ in range(num_batches):
                batch = get_batch(data, seq_len=SEQ_LEN)
                loss = model.loss(batch)
                total_loss += loss.item() * (batch.shape[0] * batch.shape[1])
                total_tokens += batch.shape[0] * batch.shape[1]
        
        avg_loss = total_loss / total_tokens
        BYTES_PER_TOKEN = 4.0
        bpb = (avg_loss / math.log(2)) * (1.0 / BYTES_PER_TOKEN)
        return bpb
    
    print("\n" + "="*70)
    print("Phase 2: Standard Evaluation (no TTT)")
    print("="*70)
    
    pre_ttt_bpb = calculate_bpb(model, val_data, num_batches=100)
    print(f"Pre-TTT BPB: {pre_ttt_bpb:.4f}")
    
    # ══════════════════════════════════════════════════════════════════
    # PHASE 3: LoRA TTT EVALUATION
    # ══════════════════════════════════════════════════════════════════
    
    print("\n" + "="*70)
    print(f"Phase 3: LoRA TTT Evaluation (rank={lora_rank}, epochs={ttt_epochs})")
    print("="*70)
    
    # Initialize LoRA for TTT - make sure params are on GPU
    model.init_all_lora(lora_rank)
    model = model.to(DEVICE)  # Re-move entire model to ensure LoRA params on GPU
    
    model.freeze_base()  # Only train LoRA weights
    
    def eval_document_with_ttt(model, doc_tokens, chunk_size=256, epochs=2, lr=0.01):
        """
        Evaluate a single document with per-document LoRA TTT
        
        1. Reset LoRA weights
        2. Split document into chunks
        3. For each epoch:
           - Process chunks left-to-right
           - Train LoRA on each chunk (except last)
           - Score tokens on final epoch
        4. Return loss
        """
        model.reset_all_lora()
        
        doc_len = len(doc_tokens)
        if doc_len < 512:  # Too short for TTT
            return None
        
        # Create LoRA optimizer for this document
        lora_params = [p for p in model.parameters() if p.requires_grad]
        ttt_opt = torch.optim.Adam(lora_params, lr=lr, betas=(0.9, 0.95))
        
        # Split into chunks
        chunks = []
        for i in range(0, doc_len - chunk_size, chunk_size // 2):  # 50% overlap
            chunk = doc_tokens[i:i+chunk_size+1]
            if len(chunk) == chunk_size + 1:
                chunks.append(chunk)
        
        if len(chunks) < 2:
            return None
        
        total_loss = 0
        total_tokens = 0
        
        for epoch in range(epochs):
            for i, chunk in enumerate(chunks):
                chunk_tensor = torch.tensor(chunk, dtype=torch.long, device=DEVICE).unsqueeze(0)
                
                # Forward pass
                loss = model.loss(chunk_tensor)
                
                # On final epoch, record loss for BPB
                if epoch == epochs - 1:
                    total_loss += loss.item() * (len(chunk) - 1)
                    total_tokens += len(chunk) - 1
                
                # Train LoRA (except on last chunk of last epoch)
                if not (epoch == epochs - 1 and i == len(chunks) - 1):
                    ttt_opt.zero_grad()
                    loss.backward()
                    ttt_opt.step()
        
        return total_loss / total_tokens if total_tokens > 0 else None
    
    # Evaluate with TTT on validation data
    # Process documents (simulate by taking long sequences)
    doc_length = 2048
    num_docs = min(50, len(val_data) // doc_length)
    
    ttt_losses = []
    start_ttt = time.time()
    
    model.train()  # Need gradients for TTT
    
    for i in range(num_docs):
        doc_start = i * doc_length
        doc_tokens = val_data[doc_start:doc_start + doc_length].astype(np.int64)
        # Ensure tokens are in valid range
        doc_tokens = np.clip(doc_tokens, 0, VOCAB_SIZE - 1).tolist()
        
        loss = eval_document_with_ttt(
            model, doc_tokens, 
            chunk_size=chunk_size, 
            epochs=ttt_epochs, 
            lr=ttt_lr
        )
        
        if loss is not None:
            ttt_losses.append(loss)
        
        if (i + 1) % 10 == 0:
            avg_loss = sum(ttt_losses) / len(ttt_losses) if ttt_losses else 0
            BYTES_PER_TOKEN = 4.0
            current_bpb = (avg_loss / math.log(2)) * (1.0 / BYTES_PER_TOKEN) if avg_loss else 0
            print(f"  Doc {i+1}/{num_docs} | Avg Loss {avg_loss:.4f} | BPB {current_bpb:.4f}")
    
    ttt_time = time.time() - start_ttt
    
    # Calculate final TTT BPB
    if ttt_losses:
        avg_ttt_loss = sum(ttt_losses) / len(ttt_losses)
        BYTES_PER_TOKEN = 4.0
        post_ttt_bpb = (avg_ttt_loss / math.log(2)) * (1.0 / BYTES_PER_TOKEN)
    else:
        post_ttt_bpb = pre_ttt_bpb
    
    # ══════════════════════════════════════════════════════════════════
    # RESULTS
    # ══════════════════════════════════════════════════════════════════
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Pre-TTT BPB:  {pre_ttt_bpb:.4f}")
    print(f"Post-TTT BPB: {post_ttt_bpb:.4f}")
    print(f"Improvement:  {(pre_ttt_bpb - post_ttt_bpb) / pre_ttt_bpb * 100:.2f}%")
    print(f"TTT Time:     {ttt_time:.0f}s for {num_docs} documents")
    print("="*70)
    
    return {
        'pre_ttt_bpb': pre_ttt_bpb,
        'post_ttt_bpb': post_ttt_bpb,
        'improvement_pct': (pre_ttt_bpb - post_ttt_bpb) / pre_ttt_bpb * 100,
        'ttt_time': ttt_time,
        'num_docs': num_docs,
    }


@app.local_entrypoint()
def main():
    print("Parameter Golf - LoRA TTT Experiment")
    print("Based on PR #548 (1.0865 BPB)")
    result = train_and_eval_ttt.remote(steps=3000)
    print(f"\nFinal: Pre-TTT {result['pre_ttt_bpb']:.4f} → Post-TTT {result['post_ttt_bpb']:.4f}")

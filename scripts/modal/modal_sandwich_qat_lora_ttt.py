"""
Sandwich QAT 30L dim=448 + legal score-first LoRA TTT evaluation.

Load a pre-trained checkpoint, score each validation chunk before any updates,
then adapt LoRA only on already-scored tokens.
"""
import os
import math

try:
    import modal
except ModuleNotFoundError:
    modal = None


if modal is not None:
    app = modal.App("sandwich-qat-lora-ttt")

    image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install([
            "torch==2.5.1",
            "numpy",
        ])
    )

    data_volume = modal.Volume.from_name("parameter-golf-data", create_if_missing=True)
else:
    app = None
    image = None
    data_volume = None


MLP_SCALES_SANDWICH_30L = [3.0] * 2 + [1.2] * 28


def _require_modal():
    if modal is None:
        raise ModuleNotFoundError(
            "No module named 'modal'. Install modal to run this script locally or on Modal."
        )


if modal is not None:
    @app.function(
        image=image,
        gpu="A100-40GB",
        volumes={"/data": data_volume},
        timeout=7200,
    )
    def eval_with_lora_ttt(
        dim: int = 448,
        n_layers: int = 30,
        n_heads: int = 8,
        n_kv_heads: int = 4,
        local_window: int = 128,
        lora_rank: int = 8,
        ttt_lr: float = 0.01,
        ttt_epochs: int = 2,
        chunk_size: int = 256,
        checkpoint_path: str = "/data/checkpoints/sandwich_qat_30l_dim448/sandwich_qat_30l_dim448_step40000.pt",
    ):
        """Load pre-trained Sandwich QAT model, evaluate with legal score-first LoRA TTT."""
        _require_modal()
        import json
        import time
        import numpy as np
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        device = torch.device("cuda")
        vocab_size = 8192
        data_dir = "/data/datasets/fineweb10B_sp8192"
        header_size = 256 * 4
        bytes_per_token = 3.67
        mlp_scales = MLP_SCALES_SANDWICH_30L

        print("=" * 70)
        print("Sandwich QAT 30L dim=448 + Legal Score-First LoRA TTT")
        print(f"  dim={dim}, layers={n_layers}, n_heads={n_heads}, n_kv={n_kv_heads}")
        print(f"  LoRA rank={lora_rank}, TTT lr={ttt_lr}, epochs={ttt_epochs}")
        print(f"  Chunk size={chunk_size}")
        print(f"  Checkpoint: {checkpoint_path}")
        print("=" * 70)

        class TernaryQuantize(torch.autograd.Function):
            @staticmethod
            def forward(ctx, weight):
                scale = weight.abs().mean()
                w_quant = torch.clamp(torch.round(weight / scale), -1, 1)
                ctx.save_for_backward(weight)
                return w_quant * scale

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output

        def ternary_quantize(weight):
            return TernaryQuantize.apply(weight)

        class QATLinear(nn.Module):
            def __init__(self, in_features, out_features, bias=False):
                super().__init__()
                self.in_features = in_features
                self.out_features = out_features
                self.weight = nn.Parameter(torch.empty(out_features, in_features))
                nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
                self.register_parameter("bias", None)
                self.qat_enabled = False
                self.lora_A = None
                self.lora_B = None

            def forward(self, x):
                if self.qat_enabled:
                    w = ternary_quantize(self.weight)
                else:
                    w = self.weight
                out = F.linear(x, w, self.bias)
                if self.lora_A is not None and self.lora_B is not None:
                    lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B)
                    out = out + lora_out
                return out

            def enable_qat(self):
                self.qat_enabled = True

            def init_lora(self, rank):
                self.lora_A = nn.Parameter(torch.randn(rank, self.in_features, device=self.weight.device) * 0.01)
                self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank, device=self.weight.device))
                return [self.lora_A, self.lora_B]

            def reset_lora(self):
                if self.lora_A is not None:
                    nn.init.normal_(self.lora_A, std=0.01)
                    nn.init.zeros_(self.lora_B)

        class RMSNorm(nn.Module):
            def __init__(self, d, eps=1e-6):
                super().__init__()
                self.eps = eps
                self.weight = nn.Parameter(torch.ones(d))

            def forward(self, x):
                return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        class RotaryEmbedding(nn.Module):
            def __init__(self, d, max_seq_len=4096):
                super().__init__()
                inv_freq = 1.0 / (10000 ** (torch.arange(0, d, 2).float() / d))
                t = torch.arange(max_seq_len)
                freqs = torch.outer(t, inv_freq)
                self.register_buffer("cos", freqs.cos())
                self.register_buffer("sin", freqs.sin())

            def forward(self, x, offset=0):
                seq_len = x.shape[1]
                cos = self.cos[offset:offset + seq_len].unsqueeze(0).unsqueeze(2)
                sin = self.sin[offset:offset + seq_len].unsqueeze(0).unsqueeze(2)
                x1, x2 = x[..., ::2], x[..., 1::2]
                return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

        class AlternatingAttention(nn.Module):
            def __init__(self, d, heads, kv_heads=None, local_window=128, is_global=True):
                super().__init__()
                self.n_heads = heads
                self.n_kv_heads = kv_heads or heads
                self.head_dim = d // heads
                self.local_window = local_window
                self.is_global = is_global
                self.wq = QATLinear(d, d)
                self.wk = QATLinear(d, self.head_dim * self.n_kv_heads)
                self.wv = QATLinear(d, self.head_dim * self.n_kv_heads)
                self.wo = QATLinear(d, d)
                self.rope = RotaryEmbedding(self.head_dim)

            def forward(self, x):
                bsz, seqlen, width = x.shape
                q = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim)
                k = self.wk(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
                v = self.wv(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
                q = self.rope(q)
                k = self.rope(k)
                if self.n_kv_heads < self.n_heads:
                    rep = self.n_heads // self.n_kv_heads
                    k = k.repeat_interleave(rep, dim=2)
                    v = v.repeat_interleave(rep, dim=2)
                q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
                scale = 1.0 / math.sqrt(self.head_dim)
                scores = torch.matmul(q, k.transpose(-2, -1)) * scale
                diag_mask = torch.eye(seqlen, device=x.device, dtype=torch.bool)
                scores = scores.masked_fill(diag_mask, 0.0)
                rows = torch.arange(seqlen, device=x.device).view(-1, 1)
                cols = torch.arange(seqlen, device=x.device).view(1, -1)
                causal_mask = cols > rows
                if self.is_global:
                    mask = causal_mask
                else:
                    window_mask = (rows - cols) > self.local_window
                    mask = causal_mask | window_mask
                scores = scores.masked_fill(mask, float("-inf"))
                attn = F.softmax(scores, dim=-1)
                out = torch.matmul(attn, v)
                out = out.transpose(1, 2).contiguous().view(bsz, seqlen, width)
                return self.wo(out)

            def init_lora(self, rank):
                params = []
                params.extend(self.wq.init_lora(rank))
                params.extend(self.wv.init_lora(rank))
                return params

            def reset_lora(self):
                self.wq.reset_lora()
                self.wv.reset_lora()

        class MLP(nn.Module):
            def __init__(self, d, hidden_dim):
                super().__init__()
                self.w1 = QATLinear(d, hidden_dim)
                self.w2 = QATLinear(hidden_dim, d)

            def forward(self, x):
                h = self.w1(x)
                h = F.leaky_relu(h, 0.5) ** 2
                return self.w2(h)

        class SandwichQATBlock(nn.Module):
            def __init__(self, d, heads, kv_heads, local_window, layer_idx, total_layers, mlp_scale):
                super().__init__()
                is_last = layer_idx == total_layers - 1
                is_global = (layer_idx % 2 == 0) or is_last
                self.attn = AlternatingAttention(d, heads, kv_heads, local_window, is_global)
                self.mlp = MLP(d, int(d * mlp_scale))
                self.ln1 = RMSNorm(d)
                self.ln2 = RMSNorm(d)
                self.alpha_attn = nn.Parameter(torch.ones(1))
                self.beta_attn = nn.Parameter(torch.ones(1))
                self.alpha_mlp = nn.Parameter(torch.ones(1))
                self.beta_mlp = nn.Parameter(torch.ones(1))

            def forward(self, x):
                attn_out = self.attn(self.ln1(x))
                x = self.alpha_attn * x + self.beta_attn * attn_out
                mlp_out = self.mlp(self.ln2(x))
                x = self.alpha_mlp * x + self.beta_mlp * mlp_out
                return x

            def init_lora(self, rank):
                return self.attn.init_lora(rank)

            def reset_lora(self):
                self.attn.reset_lora()

        class SandwichQATTransformer(nn.Module):
            def __init__(self, vocab_size_, d, total_layers, heads, kv_heads, local_window, mlp_scales_):
                super().__init__()
                self.vocab_size = vocab_size_
                self.embed = nn.Embedding(vocab_size_, d)
                self.blocks = nn.ModuleList([
                    SandwichQATBlock(d, heads, kv_heads, local_window, i, total_layers, mlp_scales_[i])
                    for i in range(total_layers)
                ])
                self.ln_f = RMSNorm(d)
                self.lm_head = QATLinear(d, vocab_size_)

            def forward(self, x, targets=None):
                x = self.embed(x)
                for block in self.blocks:
                    x = block(x)
                x = self.ln_f(x)
                logits = self.lm_head(x)
                loss = None
                if targets is not None:
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
                return logits, loss

            def enable_qat(self):
                for module in self.modules():
                    if isinstance(module, QATLinear):
                        module.enable_qat()

            def init_all_lora(self, rank):
                params = []
                for block in self.blocks:
                    params.extend(block.init_lora(rank))
                params.extend(self.lm_head.init_lora(rank))
                return params

            def reset_all_lora(self):
                for block in self.blocks:
                    block.reset_lora()
                self.lm_head.reset_lora()

            def freeze_base(self):
                for name, param in self.named_parameters():
                    if "lora" not in name:
                        param.requires_grad = False

        print("\nLoading model from checkpoint...")
        model = SandwichQATTransformer(vocab_size, dim, n_layers, n_heads, n_kv_heads, local_window, mlp_scales).to(device)
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt["model_state_dict"]
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys: {missing}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected}")
        model.enable_qat()
        print(f"  Loaded checkpoint from step {ckpt['step']}, val_bpb={ckpt.get('val_bpb', 'N/A')}")
        print(f"  Model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

        print("\nLoading validation data...")
        val_files = sorted([f for f in os.listdir(data_dir) if "val" in f])
        val_data = []
        for name in val_files:
            with open(os.path.join(data_dir, name), "rb") as fp:
                fp.seek(header_size)
                data = np.frombuffer(fp.read(), dtype=np.uint16)
            val_data.append(data)
        val_data = np.concatenate(val_data)
        print(f"  Val tokens: {len(val_data)/1e6:.1f}M")

        def calculate_bpb_standard(model_, data_, seq_len=256, num_batches=100):
            model_.eval()
            total_loss = 0.0
            total_tokens = 0
            with torch.no_grad():
                for _ in range(num_batches):
                    starts = np.random.randint(0, len(data_) - seq_len - 1, 64)
                    batch = np.stack([data_[i:i + seq_len + 1] for i in starts])
                    batch_t = torch.from_numpy(batch.astype(np.int64)).to(device)
                    _, loss = model_(batch_t[:, :-1], batch_t[:, 1:])
                    total_loss += loss.item() * (batch_t.shape[0] * batch_t.shape[1])
                    total_tokens += batch_t.shape[0] * batch_t.shape[1]
            avg_loss = total_loss / total_tokens
            return (avg_loss / math.log(2)) * (1.0 / bytes_per_token)

        print("\n" + "=" * 70)
        print("Phase 1: Standard Evaluation (no TTT)")
        print("=" * 70)
        pre_ttt_bpb = calculate_bpb_standard(model, val_data)
        print(f"  Pre-TTT BPB: {pre_ttt_bpb:.4f}")

        print("\n" + "=" * 70)
        print(f"Phase 2: LoRA TTT Evaluation (rank={lora_rank}, epochs={ttt_epochs})")
        print("=" * 70)
        lora_params = model.init_all_lora(lora_rank)
        print(f"  LoRA params initialized: {sum(p.numel() for p in lora_params)/1e6:.2f}M")
        print(f"  Applied to: Q, V of all {n_layers} attention layers + LM head")
        print("  Eval mode: legal score-first, single-pass per chunk")
        model.freeze_base()

        def eval_document_with_ttt(model_, doc_tokens, chunk_size_, epochs_, lr_):
            model_.reset_all_lora()
            if len(doc_tokens) < 512:
                return None
            trainable = [p for p in model_.parameters() if p.requires_grad]
            ttt_opt = torch.optim.Adam(trainable, lr=lr_, betas=(0.9, 0.95))
            chunks = []
            for i in range(0, len(doc_tokens) - chunk_size_, chunk_size_ // 2):
                chunk = doc_tokens[i:i + chunk_size_ + 1]
                if len(chunk) == chunk_size_ + 1:
                    chunks.append(chunk)
            if len(chunks) < 2:
                return None
            total_loss = 0.0
            total_tokens = 0
            for chunk_idx, chunk in enumerate(chunks):
                chunk_t = torch.tensor(chunk, dtype=torch.long, device=device).unsqueeze(0)
                model_.eval()
                with torch.no_grad():
                    _, score_loss = model_(chunk_t[:, :-1], chunk_t[:, 1:])
                total_loss += score_loss.item() * (len(chunk) - 1)
                total_tokens += len(chunk) - 1
                if chunk_idx < len(chunks) - 1:
                    model_.train()
                    for _ in range(epochs_):
                        _, train_loss = model_(chunk_t[:, :-1], chunk_t[:, 1:])
                        ttt_opt.zero_grad()
                        train_loss.backward()
                        ttt_opt.step()
            return total_loss / total_tokens if total_tokens > 0 else None

        doc_length = 2048
        num_docs = min(50, len(val_data) // doc_length)
        ttt_losses = []
        start_ttt = time.time()
        for i in range(num_docs):
            doc_start = i * doc_length
            doc_tokens = val_data[doc_start:doc_start + doc_length].astype(np.int64)
            doc_tokens = np.clip(doc_tokens, 0, vocab_size - 1).tolist()
            loss = eval_document_with_ttt(model, doc_tokens, chunk_size, ttt_epochs, ttt_lr)
            if loss is not None:
                ttt_losses.append(loss)
            if (i + 1) % 10 == 0:
                avg_loss = sum(ttt_losses) / len(ttt_losses) if ttt_losses else 0
                current_bpb = (avg_loss / math.log(2)) * (1.0 / bytes_per_token) if avg_loss else 0
                print(f"  Doc {i+1}/{num_docs} | Avg Loss {avg_loss:.4f} | BPB {current_bpb:.4f} | Time {time.time()-start_ttt:.0f}s")

        ttt_time = time.time() - start_ttt
        avg_ttt_loss = sum(ttt_losses) / len(ttt_losses) if ttt_losses else 0
        post_ttt_bpb = (avg_ttt_loss / math.log(2)) * (1.0 / bytes_per_token) if ttt_losses else pre_ttt_bpb

        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"  Checkpoint:     step {ckpt['step']}")
        print(f"  Pre-TTT BPB:    {pre_ttt_bpb:.4f}")
        print(f"  Post-TTT BPB:   {post_ttt_bpb:.4f}")
        print(f"  Improvement:    {(pre_ttt_bpb - post_ttt_bpb) / pre_ttt_bpb * 100:.2f}%")
        print(f"  TTT Time:       {ttt_time:.0f}s for {num_docs} documents")
        print(f"  LoRA rank:      {lora_rank}")
        print(f"  TTT epochs:     {ttt_epochs}")
        print(f"  TTT lr:         {ttt_lr}")
        print("=" * 70)

        results = {
            "style": "sandwich_qat_30l_dim448_legal_lora_ttt",
            "checkpoint": checkpoint_path,
            "checkpoint_step": ckpt["step"],
            "config": {
                "dim": dim,
                "n_layers": n_layers,
                "n_heads": n_heads,
                "n_kv_heads": n_kv_heads,
                "local_window": local_window,
                "lora_rank": lora_rank,
                "ttt_lr": ttt_lr,
                "ttt_epochs": ttt_epochs,
                "chunk_size": chunk_size,
            },
            "pre_ttt_bpb": pre_ttt_bpb,
            "post_ttt_bpb": post_ttt_bpb,
            "improvement_pct": (pre_ttt_bpb - post_ttt_bpb) / pre_ttt_bpb * 100,
            "ttt_time": ttt_time,
            "num_docs": num_docs,
        }
        results_dir = f"/data/checkpoints/sandwich_qat_30l_dim{dim}"
        os.makedirs(results_dir, exist_ok=True)
        results_path = os.path.join(results_dir, f"lora_ttt_r{lora_rank}_pre{pre_ttt_bpb:.4f}_post{post_ttt_bpb:.4f}.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        data_volume.commit()
        print(f"\n[SAVED] Results: {results_path}")
        return results


if modal is not None:
    @app.local_entrypoint()
    def main(
        lora_rank: int = 8,
        ttt_lr: float = 0.01,
        ttt_epochs: int = 2,
        chunk_size: int = 256,
        dim: int = 448,
        checkpoint_path: str = "",
    ):
        if not checkpoint_path:
            checkpoint_path = f"/data/checkpoints/sandwich_qat_30l_dim{dim}/sandwich_qat_30l_dim{dim}_step40000.pt"
        result = eval_with_lora_ttt.remote(
            lora_rank=lora_rank,
            ttt_lr=ttt_lr,
            ttt_epochs=ttt_epochs,
            chunk_size=chunk_size,
            dim=dim,
            checkpoint_path=checkpoint_path,
        )
        print(f"\n[FINAL] Pre-TTT {result['pre_ttt_bpb']:.4f} → Post-TTT {result['post_ttt_bpb']:.4f}")
        print(f"  Improvement: {result['improvement_pct']:.2f}%")

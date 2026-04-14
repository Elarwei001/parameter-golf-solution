"""
Modal depth-utilization comparison experiment.

Goal:
- Compare three clean base-model variants under the same budget:
  1. baseline: plain 11-layer model, uniform MLP, no mHC, no recurrence
  2. recurrence: plain 11-layer model, uniform MLP, no mHC, loop middle layers
  3. sandwich: 11-layer model, sandwich MLP + mHC

Run examples:
  modal run scripts/modal/modal_depth_compare.py --mode baseline
  modal run scripts/modal/modal_depth_compare.py --mode recurrence
  modal run scripts/modal/modal_depth_compare.py --mode sandwich
"""
import os
import math

try:
    import modal
except ModuleNotFoundError:
    modal = None


def _require_modal():
    if modal is None:
        raise ModuleNotFoundError("modal is required to run this script via Modal")


APP_NAME = "depth-compare"
MLP_SCALES_BASELINE = [3.0] * 11
MLP_SCALES_SANDWICH = [3.0, 3.0, 3.0, 1.2, 1.2, 1.2, 1.2, 1.2, 3.0, 3.0, 3.0]
RECURRENCE_LAYER_RANGE = (4, 6)  # loop layers 4-5 once more

if modal is not None:
    app = modal.App(APP_NAME)

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
    def train_compare(
        mode: str = "baseline",
        seed: int = 42,
        dim: int = 448,
        n_layers: int = 11,
        n_heads: int = 8,
        n_kv_heads: int = 4,
        local_window: int = 128,
        lr: float = 1e-3,
        batch_size: int = 64,
        seq_len: int = 256,
        steps: int = 5000,
    ):
        import json
        import time
        import numpy as np
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        assert mode in ("baseline", "recurrence", "sandwich")

        torch.manual_seed(seed)
        np.random.seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        vocab_size = 8192
        data_dir = "/data/datasets/fineweb10B_sp8192"
        header_size = 256 * 4
        bytes_per_token = 3.67
        use_mhc = (mode == "sandwich")
        use_recurrence = (mode == "recurrence")
        mlp_scales = MLP_SCALES_SANDWICH if mode == "sandwich" else MLP_SCALES_BASELINE

        print("=" * 72)
        print(f"Depth Compare: {mode.upper()}")
        print(f"  dim={dim}, n_layers={n_layers}, heads={n_heads}/{n_kv_heads}, steps={steps}")
        print(f"  use_mhc={use_mhc}, use_recurrence={use_recurrence}")
        print(f"  mlp_scales={mlp_scales}")
        if use_recurrence:
            print(f"  recurrence loop: layers {RECURRENCE_LAYER_RANGE[0]}-{RECURRENCE_LAYER_RANGE[1]-1} replayed once")
        print("=" * 72)

        train_files = sorted([f for f in os.listdir(data_dir) if 'train' in f])
        val_files = sorted([f for f in os.listdir(data_dir) if 'val' in f])

        train_data = []
        for f in train_files[:5]:
            with open(os.path.join(data_dir, f), 'rb') as fp:
                fp.seek(header_size)
                data = np.frombuffer(fp.read(), dtype=np.uint16)
            train_data.append(data)
        train_tokens = torch.from_numpy(np.concatenate(train_data).astype(np.int64))

        val_data = []
        for f in val_files:
            with open(os.path.join(data_dir, f), 'rb') as fp:
                fp.seek(header_size)
                data = np.frombuffer(fp.read(), dtype=np.uint16)
            val_data.append(data)
        val_tokens = torch.from_numpy(np.concatenate(val_data).astype(np.int64))

        class RMSNorm(nn.Module):
            def __init__(self, width, eps=1e-6):
                super().__init__()
                self.eps = eps
                self.weight = nn.Parameter(torch.ones(width))

            def forward(self, x):
                return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        class RotaryEmbedding(nn.Module):
            def __init__(self, width, max_seq_len=4096):
                super().__init__()
                inv_freq = 1.0 / (10000 ** (torch.arange(0, width, 2).float() / width))
                t = torch.arange(max_seq_len)
                freqs = torch.outer(t, inv_freq)
                self.register_buffer('cos', freqs.cos())
                self.register_buffer('sin', freqs.sin())

            def forward(self, x, offset=0):
                seq_len_ = x.shape[1]
                cos = self.cos[offset:offset+seq_len_].unsqueeze(0).unsqueeze(2)
                sin = self.sin[offset:offset+seq_len_].unsqueeze(0).unsqueeze(2)
                x1, x2 = x[..., ::2], x[..., 1::2]
                return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

        class AlternatingAttention(nn.Module):
            def __init__(self, width, heads, kv_heads=None, local_window_=128, is_global=True):
                super().__init__()
                self.n_heads = heads
                self.n_kv_heads = kv_heads or heads
                self.head_dim = width // heads
                self.local_window = local_window_
                self.is_global = is_global

                self.wq = nn.Linear(width, width, bias=False)
                self.wk = nn.Linear(width, self.head_dim * self.n_kv_heads, bias=False)
                self.wv = nn.Linear(width, self.head_dim * self.n_kv_heads, bias=False)
                self.wo = nn.Linear(width, width, bias=False)
                self.rope = RotaryEmbedding(self.head_dim)

            def forward(self, x):
                bsz, tlen, width = x.shape
                q = self.wq(x).view(bsz, tlen, self.n_heads, self.head_dim)
                k = self.wk(x).view(bsz, tlen, self.n_kv_heads, self.head_dim)
                v = self.wv(x).view(bsz, tlen, self.n_kv_heads, self.head_dim)
                q = self.rope(q)
                k = self.rope(k)
                if self.n_kv_heads < self.n_heads:
                    rep = self.n_heads // self.n_kv_heads
                    k = k.repeat_interleave(rep, dim=2)
                    v = v.repeat_interleave(rep, dim=2)
                q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
                scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
                diag_mask = torch.eye(tlen, device=x.device, dtype=torch.bool)
                scores = scores.masked_fill(diag_mask, 0.0)
                rows = torch.arange(tlen, device=x.device).view(-1, 1)
                cols = torch.arange(tlen, device=x.device).view(1, -1)
                causal_mask = cols > rows
                if self.is_global:
                    mask = causal_mask
                else:
                    window_mask = (rows - cols) > self.local_window
                    mask = causal_mask | window_mask
                scores = scores.masked_fill(mask, float('-inf'))
                attn = F.softmax(scores, dim=-1)
                out = torch.matmul(attn, v)
                out = out.transpose(1, 2).contiguous().view(bsz, tlen, width)
                return self.wo(out)

        class MLP(nn.Module):
            def __init__(self, width, hidden_dim):
                super().__init__()
                self.w1 = nn.Linear(width, hidden_dim, bias=False)
                self.w2 = nn.Linear(hidden_dim, width, bias=False)

            def forward(self, x):
                h = self.w1(x)
                h = F.leaky_relu(h, 0.5) ** 2
                return self.w2(h)

        class CompareBlock(nn.Module):
            def __init__(self, width, heads, kv_heads, local_window_, layer_idx, total_layers, mlp_scale, use_mhc_):
                super().__init__()
                is_last = (layer_idx == total_layers - 1)
                is_global = (layer_idx % 2 == 0) or is_last
                self.layer_idx = layer_idx
                self.is_global = is_global
                self.attn_type = "Global" if is_global else f"Local(w={local_window_})"
                self.use_mhc = use_mhc_
                self.attn = AlternatingAttention(width, heads, kv_heads, local_window_, is_global)
                self.mlp_scale = mlp_scale
                self.mlp_hidden = int(width * mlp_scale)
                self.mlp = MLP(width, self.mlp_hidden)
                self.ln1 = RMSNorm(width)
                self.ln2 = RMSNorm(width)
                if use_mhc_:
                    self.alpha_attn = nn.Parameter(torch.ones(1))
                    self.beta_attn = nn.Parameter(torch.ones(1))
                    self.alpha_mlp = nn.Parameter(torch.ones(1))
                    self.beta_mlp = nn.Parameter(torch.ones(1))

            def forward(self, x):
                attn_out = self.attn(self.ln1(x))
                if self.use_mhc:
                    x = self.alpha_attn * x + self.beta_attn * attn_out
                else:
                    x = x + attn_out
                mlp_out = self.mlp(self.ln2(x))
                if self.use_mhc:
                    x = self.alpha_mlp * x + self.beta_mlp * mlp_out
                else:
                    x = x + mlp_out
                return x

            def mhc_params(self):
                if not self.use_mhc:
                    return None
                return {
                    'layer': self.layer_idx,
                    'attn_type': self.attn_type,
                    'mlp_scale': self.mlp_scale,
                    'alpha_attn': self.alpha_attn.item(),
                    'beta_attn': self.beta_attn.item(),
                    'alpha_mlp': self.alpha_mlp.item(),
                    'beta_mlp': self.beta_mlp.item(),
                }

        class CompareTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(vocab_size, dim)
                self.blocks = nn.ModuleList([
                    CompareBlock(dim, n_heads, n_kv_heads, local_window, i, n_layers, mlp_scales[i], use_mhc)
                    for i in range(n_layers)
                ])
                self.ln_f = RMSNorm(dim)
                self.lm_head = nn.Linear(dim, vocab_size, bias=False)
                self.embed.weight = self.lm_head.weight

            def forward(self, x, targets=None):
                x = self.embed(x)
                for block in self.blocks:
                    x = block(x)
                if use_recurrence:
                    start, end = RECURRENCE_LAYER_RANGE
                    for idx in range(start, end):
                        x = self.blocks[idx](x)
                x = self.ln_f(x)
                logits = self.lm_head(x)
                loss = None
                if targets is not None:
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
                return logits, loss

        model = CompareTransformer().to(device)
        total_params = sum(p.numel() for p in model.parameters())
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
        min_lr_ratio = 0.1

        def get_batch(split):
            data = train_tokens if split == 'train' else val_tokens
            max_start = len(data) - seq_len - 1
            ix = torch.randint(0, max_start, (batch_size,))
            x = torch.stack([data[i:i+seq_len] for i in ix]).long().to(device)
            y = torch.stack([data[i+1:i+seq_len+1] for i in ix]).long().to(device)
            return x, y

        def cosine_lr(step):
            if step < 200:
                return step / 200
            progress = (step - 200) / (steps - 200)
            return min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + math.cos(math.pi * progress))

        start_time = time.time()
        for step in range(1, steps + 1):
            for group in optimizer.param_groups:
                group['lr'] = lr * cosine_lr(step)
            x, y = get_batch('train')
            _, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step % 500 == 0:
                elapsed = time.time() - start_time
                print(f"Step {step}/{steps} | Loss {loss.item():.4f} | LR {optimizer.param_groups[0]['lr']:.2e} | Time {elapsed:.0f}s")
            if use_mhc and step % 2500 == 0:
                print(f"\n[mHC @ step {step}]")
                for block in model.blocks:
                    p = block.mhc_params()
                    if p is not None:
                        print(f"  L{p['layer']:02d} {p['attn_type']:<12} mlp={p['mlp_scale']:.1f} aA={p['alpha_attn']:.3f} bA={p['beta_attn']:.3f} aM={p['alpha_mlp']:.3f} bM={p['beta_mlp']:.3f}")

        model.eval()
        val_losses = []
        with torch.no_grad():
            for _ in range(100):
                x, y = get_batch('val')
                _, loss = model(x, y)
                val_losses.append(loss.item())
        val_loss = sum(val_losses) / len(val_losses)
        val_bpb = (val_loss / math.log(2)) * (1.0 / bytes_per_token)

        checkpoint_dir = f"/data/checkpoints/depth_compare/{mode}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        result = {
            'mode': mode,
            'config': {
                'seed': seed,
                'dim': dim,
                'n_layers': n_layers,
                'n_heads': n_heads,
                'n_kv_heads': n_kv_heads,
                'local_window': local_window,
                'lr': lr,
                'batch_size': batch_size,
                'seq_len': seq_len,
                'steps': steps,
                'use_mhc': use_mhc,
                'use_recurrence': use_recurrence,
                'mlp_scales': mlp_scales,
                'recurrence_layer_range': RECURRENCE_LAYER_RANGE if use_recurrence else None,
            },
            'val_loss': val_loss,
            'val_bpb': val_bpb,
            'total_params': total_params,
        }
        if use_mhc:
            result['final_mhc'] = [block.mhc_params() for block in model.blocks]

        results_path = os.path.join(checkpoint_dir, f"{mode}_bpb{val_bpb:.4f}.json")
        with open(results_path, 'w') as f:
            json.dump(result, f, indent=2)
        ckpt_path = os.path.join(checkpoint_dir, f"{mode}_step{steps}.pt")
        torch.save({
            'step': steps,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'result': result,
        }, ckpt_path)
        data_volume.commit()

        print("\n" + "=" * 72)
        print(f"[RESULT] {mode.upper()}")
        print(f"  val_loss={val_loss:.4f}")
        print(f"  val_bpb={val_bpb:.4f}")
        print(f"  params={total_params/1e6:.2f}M")
        print(f"  saved={results_path}")
        print("=" * 72)
        return result


    @app.local_entrypoint()
    def main(
        mode: str = "baseline",
        seed: int = 42,
        dim: int = 448,
        n_layers: int = 11,
        n_heads: int = 8,
        n_kv_heads: int = 4,
        local_window: int = 128,
        lr: float = 1e-3,
        batch_size: int = 64,
        seq_len: int = 256,
        steps: int = 5000,
    ):
        assert mode in ("baseline", "recurrence", "sandwich")
        result = train_compare.remote(
            mode=mode,
            seed=seed,
            dim=dim,
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            local_window=local_window,
            lr=lr,
            batch_size=batch_size,
            seq_len=seq_len,
            steps=steps,
        )
        print(f"[FINAL] {result['mode']} bpb={result['val_bpb']:.4f} params={result['total_params']/1e6:.2f}M")
else:
    def main(*args, **kwargs):
        _require_modal()

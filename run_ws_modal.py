"""
Modal training for Weight Sharing + Deep Layers experiment.
Tests correct weight sharing: same params, but DEEPER model.

Configs:
  baseline : 6 unique × 1 pass = depth 6,  dim=256, 4.39M params
  ws_18a   : 3 unique × 6 pass = depth 18, dim=352, 4.18M params  
  ws_18b   : 6 unique × 3 pass = depth 18, dim=256, 4.39M params

All use LeakyReLU² + Sliding Window + AdamW.
Baseline (prior best): 2.2821 BPB on tinyshakespeare.
"""
import modal
import os

app = modal.App("parameter-golf-ws")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["git"])
    .pip_install([
        "torch>=2.0",
        "numpy",
        "sentencepiece",
        "tqdm",
    ])
)

data_volume = modal.Volume.from_name("parameter-golf-data", create_if_missing=True)
output_volume = modal.Volume.from_name("parameter-golf-output", create_if_missing=True)

CODE_REPO = "https://github.com/Elarwei001/parameter-golf-solution.git"

VOCAB_SIZE = 256  # byte-level on tinyshakespeare
SEQ_LEN    = 256
BATCH_SIZE = 64   # A100 can handle larger batches


def setup_code():
    import subprocess, sys
    if not os.path.exists("/root/project"):
        subprocess.run(["git", "clone", CODE_REPO, "/root/project"], check=True)
    else:
        subprocess.run(["git", "-C", "/root/project", "pull"], check=False)
    sys.path.insert(0, "/root/project")


def load_tinyshakespeare():
    """Download or reuse tinyshakespeare."""
    import subprocess
    path = "/tmp/tinyshakespeare.txt"
    if not os.path.exists(path):
        subprocess.run([
            "python3", "-c",
            "import urllib.request; urllib.request.urlretrieve("
            "'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',"
            f"'{path}')"
        ], check=True)
    return path


@app.function(
    image=image,
    gpu="A100",
    volumes={"/output": output_volume},
    timeout=2400,  # 40 min max (3 experiments × 10min + overhead)
)
def run_weight_sharing_experiments(
    duration_sec: int = 600,  # 10 min per experiment
    experiments: str = "all",  # "all", "baseline", "ws_18a", "ws_18b"
):
    """Run weight sharing experiments on A100."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import math
    import time
    import json
    from datetime import datetime

    setup_code()
    
    # Load tinyshakespeare
    data_path = load_tinyshakespeare()
    with open(data_path, "rb") as f:
        raw = f.read()
    data = np.frombuffer(raw, dtype=np.uint8).copy()
    split = int(len(data) * 0.9)
    train_data = data[:split]
    val_data   = data[split:]
    print(f"Train: {len(train_data)/1e6:.1f}M tokens | Val: {len(val_data)/1000:.0f}K tokens")

    device = torch.device("cuda")

    # ── Import model ──────────────────────────────────────────────
    from run_weight_sharing import GPT_WeightShared

    def get_batch(data, seq_len=SEQ_LEN, batch_size=BATCH_SIZE):
        starts = np.random.randint(0, len(data) - seq_len - 1, batch_size)
        batch  = np.stack([data[i:i+seq_len+1] for i in starts])
        return torch.from_numpy(batch.astype(np.int64)).to(device)

    def train_model(model, name, n_unique, n_passes, duration_sec):
        model = model.to(device)
        # Compile for speed
        if hasattr(torch, 'compile'):
            model = torch.compile(model)

        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.01)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        eff_depth = n_unique * n_passes
        
        print(f"\n{'='*65}")
        print(f"Experiment   : {name}")
        print(f"Params       : {n_params/1e6:.3f}M")
        print(f"Unique layers: {n_unique}  passes: {n_passes}  eff_depth: {eff_depth}")
        print(f"Duration     : {duration_sec/60:.0f} min")
        print(f"{'='*65}")

        model.train()
        losses, step, start = [], 0, time.time()

        while time.time() - start < duration_sec:
            batch = get_batch(train_data)
            opt.zero_grad()
            out  = model.compute_loss(batch)
            out['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(out['ce_loss'].item())
            step += 1
            if step % 500 == 0:
                recent  = np.mean(losses[-100:])
                bpb     = recent / math.log(2)
                elapsed = time.time() - start
                print(f"  [{elapsed/60:.1f}min] step {step}: loss={recent:.4f} BPB={bpb:.4f}")

        # Val eval
        model.eval()
        val_losses = []
        with torch.no_grad():
            for _ in range(50):
                batch = get_batch(val_data)
                d     = model.compute_loss(batch)
                val_losses.append(d['ce_loss'].item())

        val_bpb   = np.mean(val_losses) / math.log(2)
        train_bpb = np.mean(losses[-100:]) / math.log(2) if losses else float('nan')
        elapsed   = time.time() - start

        print(f"\n✅ {name}: {step} steps, {elapsed/60:.1f}min")
        print(f"   Train BPB : {train_bpb:.4f}")
        print(f"   Val   BPB : {val_bpb:.4f}")
        return {
            'name': name,
            'steps': step,
            'train_bpb': float(train_bpb),
            'val_bpb': float(val_bpb),
            'params': n_params,
            'n_unique_layers': n_unique,
            'n_passes': n_passes,
            'eff_depth': eff_depth,
        }

    # ── Experiment configs ────────────────────────────────────────
    EXPS = {
        'baseline': dict(
            desc='baseline_no_sharing',
            dim=256, n_unique=6, n_passes=1,
            n_heads=8, n_kv_heads=4,
        ),
        'ws_18a': dict(
            desc='ws_3Lx6pass_dim352',
            dim=352, n_unique=3, n_passes=6,
            n_heads=8, n_kv_heads=4,
        ),
        'ws_18b': dict(
            desc='ws_6Lx3pass_dim256',
            dim=256, n_unique=6, n_passes=3,
            n_heads=8, n_kv_heads=4,
        ),
    }

    run_exps = list(EXPS.keys()) if experiments == 'all' else [experiments]
    results = []

    for key in run_exps:
        cfg = EXPS[key]
        model = GPT_WeightShared(
            vocab_size=VOCAB_SIZE,
            dim=cfg['dim'],
            n_unique_layers=cfg['n_unique'],
            n_passes=cfg['n_passes'],
            n_heads=cfg['n_heads'],
            n_kv_heads=cfg['n_kv_heads'],
            mlp_mult=4,
            max_seq_len=SEQ_LEN + 64,
            window_size=128,
        )
        result = train_model(model, cfg['desc'],
                             cfg['n_unique'], cfg['n_passes'],
                             duration_sec)
        results.append(result)
        del model
        torch.cuda.empty_cache()

    # ── Save results ──────────────────────────────────────────────
    os.makedirs("/output/experiments", exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = f"/output/experiments/weight_sharing_{ts}.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    output_volume.commit()

    # ── Summary ───────────────────────────────────────────────────
    prior_best = 2.2821
    print("\n" + "="*70)
    print("WEIGHT SHARING RESULTS  (prior best: 2.2821 BPB)")
    print("="*70)
    print(f"  {'Model':<30} {'Params':>8} {'Depth':>6} {'Val BPB':>10}")
    print("-"*70)
    print(f"  {'leaky+sw prior best':30} {'4.39M':>8} {'6':>6} {'2.2821':>10}")
    for r in results:
        marker = " 🏆" if r['val_bpb'] < prior_best else ""
        print(f"  {r['name']:<30} {r['params']/1e6:>7.3f}M {r['eff_depth']:>6} {r['val_bpb']:>10.4f}{marker}")
    print("="*70)

    return results


@app.local_entrypoint()
def main():
    print("Weight Sharing Experiment - Modal Runner")
    print("Commands:")
    print("  modal run run_ws_modal.py  (all experiments, 10min each)")
    print("  modal run run_ws_modal.py::run_weight_sharing_experiments --experiments baseline")
    print("  modal run run_ws_modal.py::run_weight_sharing_experiments --duration-sec 60  (quick test)")

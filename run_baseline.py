"""
Run official baseline on Modal to get reference BPB score.
"""
import modal
import os

app = modal.App("parameter-golf-baseline")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["git"])
    .pip_install([
        "torch>=2.0",
        "numpy", 
        "sentencepiece",
        "huggingface-hub",
        "tqdm",
    ])
)

data_volume = modal.Volume.from_name("parameter-golf-data", create_if_missing=True)
output_volume = modal.Volume.from_name("parameter-golf-output", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100",
    volumes={
        "/data": data_volume,
        "/output": output_volume,
    },
    timeout=900,  # 15 min
)
def run_baseline(max_seconds: int = 300, iterations: int = 2000):
    """Run official baseline training"""
    import subprocess
    import sys
    
    # Clone official repo
    subprocess.run([
        "git", "clone", "https://github.com/openai/parameter-golf.git",
        "/tmp/parameter-golf"
    ], check=True)
    
    # Set environment
    env = os.environ.copy()
    env.update({
        "DATA_PATH": "/data/datasets/fineweb10B_sp1024",
        "TOKENIZER_PATH": "/data/tokenizers/fineweb_1024_bpe.model",
        "MAX_WALLCLOCK_SECONDS": str(max_seconds),
        "ITERATIONS": str(iterations),
        "VAL_LOSS_EVERY": "500",
        "TRAIN_LOG_EVERY": "100",
    })
    
    # Run training with real-time output
    print("=" * 60)
    print(f"Running baseline: max {max_seconds}s, {iterations} iters")
    print("=" * 60)
    sys.stdout.flush()
    
    process = subprocess.Popen(
        [sys.executable, "/tmp/parameter-golf/train_gpt.py"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    
    # Stream output
    output_lines = []
    for line in process.stdout:
        print(line, end='', flush=True)
        output_lines.append(line)
    
    process.wait()
    
    # Parse final BPB from output
    final_bpb = None
    for line in reversed(output_lines):
        if "val_bpb" in line:
            import re
            match = re.search(r'val_bpb[:\s]+([\d.]+)', line)
            if match:
                final_bpb = float(match.group(1))
                break
    
    return {"exit_code": process.returncode, "final_bpb": final_bpb}


@app.function(
    image=image,
    gpu="A100",
    volumes={"/data": data_volume},
    timeout=180,
)
def check_data():
    """Check data files exist"""
    import os
    
    data_path = "/data/datasets/fineweb10B_sp1024"
    tokenizer_path = "/data/tokenizers/fineweb_1024_bpe.model"
    
    print("Checking data...")
    
    if os.path.exists(data_path):
        files = os.listdir(data_path)
        train_files = [f for f in files if "train" in f]
        val_files = [f for f in files if "val" in f]
        print(f"✅ Data path exists: {data_path}")
        print(f"   Train shards: {len(train_files)}")
        print(f"   Val shards: {len(val_files)}")
    else:
        print(f"❌ Data path missing: {data_path}")
        
    if os.path.exists(tokenizer_path):
        print(f"✅ Tokenizer exists: {tokenizer_path}")
    else:
        print(f"❌ Tokenizer missing: {tokenizer_path}")
    
    return {"data_ok": os.path.exists(data_path), "tokenizer_ok": os.path.exists(tokenizer_path)}


@app.local_entrypoint()
def main():
    print("Commands:")
    print("  modal run run_baseline.py::check_data")
    print("  modal run run_baseline.py::run_baseline --max-seconds 300")

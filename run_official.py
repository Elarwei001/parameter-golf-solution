"""
Run official baseline exactly as specified, just shorter time.
"""
import modal
import os
import sys

app = modal.App("parameter-golf-official")

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


@app.function(
    image=image,
    gpu="A100",
    volumes={"/data": data_volume},
    timeout=600,
)
def run_official_baseline(max_seconds: int = 300):
    """Run the official train_gpt.py baseline"""
    import subprocess
    import time
    
    # Clone official repo
    print("Cloning official repo...")
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
        "ITERATIONS": "10000",  # High, will be stopped by time limit
        "VAL_LOSS_EVERY": "200",
        "TRAIN_LOG_EVERY": "50",
    })
    
    print(f"\n{'='*60}")
    print(f"Running official baseline: max {max_seconds}s")
    print(f"{'='*60}\n")
    sys.stdout.flush()
    
    # Run training with output capture
    start = time.time()
    
    process = subprocess.Popen(
        [sys.executable, "/tmp/parameter-golf/train_gpt.py"],
        env=env,
        cwd="/tmp/parameter-golf",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    
    output_lines = []
    for line in process.stdout:
        print(line, end='', flush=True)
        output_lines.append(line)
    
    process.wait()
    elapsed = time.time() - start
    
    # Extract final BPB
    final_bpb = None
    for line in reversed(output_lines):
        if "val_bpb:" in line:
            import re
            match = re.search(r'val_bpb:([\d.]+)', line)
            if match:
                final_bpb = float(match.group(1))
                break
    
    print(f"\n{'='*60}")
    print(f"Training completed in {elapsed:.1f}s")
    print(f"Exit code: {process.returncode}")
    if final_bpb:
        print(f"Final BPB: {final_bpb:.4f}")
    print(f"{'='*60}")
    
    return {"exit_code": process.returncode, "time": elapsed, "final_bpb": final_bpb}


@app.local_entrypoint()
def main():
    print("Run: modal run run_official.py::run_official_baseline --max-seconds 300")

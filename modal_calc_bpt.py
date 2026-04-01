"""
计算真实的 bytes_per_token 比率
"""
import modal
import os

app = modal.App("calc-bytes-per-token")

image = modal.Image.debian_slim(python_version="3.11").pip_install(["numpy", "sentencepiece"])

data_volume = modal.Volume.from_name("parameter-golf-data", create_if_missing=True)


@app.function(image=image, volumes={"/data": data_volume}, timeout=300)
def calc_bytes_per_token():
    import numpy as np
    import sentencepiece as spm
    
    # 找 tokenizer
    TOKENIZER_PATH = None
    for root, dirs, files in os.walk("/data"):
        for f in files:
            if "8192" in f and f.endswith(".model"):
                TOKENIZER_PATH = os.path.join(root, f)
                print(f"Found tokenizer: {TOKENIZER_PATH}")
                break
        if TOKENIZER_PATH:
            break
    
    if not TOKENIZER_PATH:
        print("❌ 找不到 tokenizer")
        return None
    
    # 加载 tokenizer
    sp = spm.SentencePieceProcessor()
    sp.Load(TOKENIZER_PATH)
    vocab_size = sp.GetPieceSize()
    print(f"Vocab size: {vocab_size}")
    
    # 读取 tokenized 数据
    DATA_DIR = "/data/datasets/fineweb10B_sp8192"
    val_files = sorted([f for f in os.listdir(DATA_DIR) if 'val' in f])
    
    data = np.fromfile(os.path.join(DATA_DIR, val_files[0]), dtype=np.uint16)
    print(f"Data dtype: {data.dtype}, max value: {data.max()}, min: {data.min()}")
    
    # 只取有效的 token（< vocab_size）
    valid_tokens = data[data < vocab_size][:50000]
    print(f"Sample size: {len(valid_tokens)} tokens")
    
    # 解码并计算字节数
    decoded = sp.DecodeIds(valid_tokens.tolist())
    sample_bytes = len(decoded.encode('utf-8'))
    
    bytes_per_token = sample_bytes / len(valid_tokens)
    
    print(f"\n✅ 结果:")
    print(f"   Sample tokens: {len(valid_tokens):,}")
    print(f"   Sample bytes:  {sample_bytes:,}")
    print(f"   BYTES_PER_TOKEN = {bytes_per_token:.4f}")
    
    # 修正 BPB
    reported_bpb = 0.986
    wrong_bpt = 4.0
    corrected_bpb = reported_bpb * wrong_bpt / bytes_per_token
    
    print(f"\n📊 BPB 修正:")
    print(f"   报告值 (BYTES_PER_TOKEN=4.0): {reported_bpb:.4f}")
    print(f"   修正值 (BYTES_PER_TOKEN={bytes_per_token:.4f}): {corrected_bpb:.4f}")
    
    return bytes_per_token


@app.local_entrypoint()
def main():
    calc_bytes_per_token.remote()

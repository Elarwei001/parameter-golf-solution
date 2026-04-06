"""
Download Scylla FineWeb dataset (subset) and upload to Modal volume.
Only downloads the first N train shards + val shard.
"""
import modal

app = modal.App("download-scylla-subset")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(["huggingface_hub"])
)

data_volume = modal.Volume.from_name("parameter-golf-data", create_if_missing=True)

@app.function(
    image=image,
    volumes={"/data": data_volume},
    timeout=1800,
)
def download_scylla_dataset():
    from huggingface_hub import hf_hub_download
    import os

    repo_id = "anthonym21/fineweb10B-scylla"
    out_dir = "/data/datasets/fineweb10B_scylla"
    os.makedirs(out_dir, exist_ok=True)

    # Download first 10 train shards + val shard (~2GB total)
    files_to_download = [f"fineweb_train_{i:06d}.bin" for i in range(10)]
    files_to_download.append("fineweb_val_000000.bin")

    for i, fname in enumerate(files_to_download):
        out_path = os.path.join(out_dir, fname)
        if os.path.exists(out_path):
            print(f"[{i+1}/{len(files_to_download)}] Already exists: {fname}")
            continue
        print(f"[{i+1}/{len(files_to_download)}] Downloading {fname}...")
        hf_hub_download(
            repo_id=repo_id,
            filename=fname,
            repo_type="dataset",
            local_dir=out_dir,
        )
        data_volume.commit()

    print(f"\nDone! Downloaded {len(files_to_download)} files to {out_dir}")
    print("Files:", os.listdir(out_dir))

@app.local_entrypoint()
def main():
    download_scylla_dataset.remote()

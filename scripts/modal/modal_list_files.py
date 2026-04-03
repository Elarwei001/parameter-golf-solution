import modal
import os

app = modal.App("list-files")
data_volume = modal.Volume.from_name("parameter-golf-data", create_if_missing=True)

@app.function(volumes={"/data": data_volume})
def list_files():
    for root, dirs, files in os.walk("/data"):
        level = root.replace("/data", "").count(os.sep)
        if level < 3:
            print(f"{'  '*level}{os.path.basename(root)}/")
            for f in files[:10]:
                print(f"{'  '*(level+1)}{f}")
            if len(files) > 10:
                print(f"{'  '*(level+1)}... +{len(files)-10} more")

@app.local_entrypoint()
def main():
    list_files.remote()

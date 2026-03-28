"""Quick Mamba test on Modal."""
import modal

app = modal.App("mamba-test")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["git"])
    .pip_install(["torch>=2.0", "numpy"])
)

@app.function(image=image, gpu="T4", timeout=300)
def test_mamba():
    import subprocess
    import sys
    
    # Clone repo
    subprocess.run(["git", "clone", "https://github.com/Elarwei001/parameter-golf-solution.git", "/root/project"], check=True)
    sys.path.insert(0, "/root/project")
    
    import torch
    from models.mamba_lm import MambaLM, MambaConfig
    
    device = torch.device("cuda")
    
    # Test configs
    configs = [
        MambaConfig(vocab_size=1024, d_model=384, n_layer=8, d_state=64),
        MambaConfig(vocab_size=1024, d_model=512, n_layer=6, d_state=64),
        MambaConfig(vocab_size=1024, d_model=448, n_layer=8, d_state=64),
    ]
    
    for i, config in enumerate(configs):
        model = MambaLM(config).to(device)
        n_params = model.count_parameters()
        size_3bit = model.estimate_size(3)
        
        print(f"\n=== Config {i+1} ===")
        print(f"d_model={config.d_model}, n_layer={config.n_layer}")
        print(f"Params: {n_params:,}")
        print(f"Size @ 3-bit: {size_3bit:.2f} MB")
        
        # Test forward
        x = torch.randint(0, config.vocab_size, (4, 256), device=device)
        logits = model(x)
        print(f"Forward: {x.shape} -> {logits.shape}")
        
        # Test loss
        batch = torch.randint(0, config.vocab_size, (4, 257), device=device)
        loss = model.compute_loss(batch)
        print(f"Loss: {loss['loss'].item():.4f}")
        
        # Quick training speed test
        import time
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        start = time.time()
        for step in range(10):
            optimizer.zero_grad()
            batch = torch.randint(0, config.vocab_size, (32, 1025), device=device)
            loss = model.compute_loss(batch)
            loss['loss'].backward()
            optimizer.step()
        elapsed = time.time() - start
        
        print(f"10 steps in {elapsed:.2f}s ({elapsed/10*1000:.0f}ms/step)")
    
    print("\n✅ All Mamba tests passed!")
    return "OK"

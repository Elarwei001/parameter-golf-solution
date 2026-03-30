# Training Techniques

> With the same model and data, training techniques can make a difference of several points. These are the lessons we learned the hard way.

## One-Line Definition

**Training Techniques = Engineering details that help the model learn better**

Architecture sets the ceiling; training techniques determine how close you get to it.

---

## Learning Rate Tuning

### Too Large vs Too Small

```
lr too large:
Loss ─╱╲──╱╲──╱╲─→ oscillates, fails to converge

lr too small:
Loss ────────────→ converges too slowly

lr just right:
Loss ─╲
       ╲
        ╲────→ smooth descent
```

### Our Findings

| lr | Result |
|----|--------|
| 1e-3 | Starts oscillating |
| **6e-4** | Best 🏆 |
| 3e-4 | Converges slowly |
| 1e-4 | Too slow |

**Recommendation**: Start at 3e-4, then multiply/divide by 2 to tune.

---

## Warmup

### Why It's Needed

At the start of training:
- Weights are randomly initialized
- Gradient directions are unstable
- A large learning rate can easily "go off track"

### Implementation

```python
def get_lr(step, warmup_steps, max_lr):
    if step < warmup_steps:
        # Linear warmup
        return max_lr * step / warmup_steps
    else:
        # Normal learning rate
        return max_lr
```

### Our Configuration

```
total_steps = 2000
warmup_steps = 100 (5%)
```

---

## Gradient Clipping

### Why It's Needed

Some batches produce very large gradients → drastic weight updates → training crashes.

```python
# Limit gradient norm
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Rule of Thumb

| Scenario | max_norm |
|----------|----------|
| Stable training | 1.0 |
| Aggressive training | 0.5 |
| Unsure | Start with 1.0 |

---

## Checkpointing and Resuming Training

### Why It Matters

- GPU jobs can be interrupted
- Need for incremental experiments
- Cost control

### Our Implementation

```python
def save_checkpoint(model, optimizer, step, path):
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': step,
    }, path)

def load_checkpoint(model, optimizer, path):
    ckpt = torch.load(path)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    return ckpt['step']
```

### Usage on Modal

```bash
# First training run
modal run experiment.py --experiment-name "v1" --max-seconds 600

# Resume training
modal run experiment.py --experiment-name "v1" --resume-from "v1" --max-seconds 1200
```

---

## Batch Size Selection

### Trade-offs

| Batch Size | Pros | Cons |
|------------|------|------|
| Large | Stable gradients, high GPU utilization | More memory, may generalize worse |
| Small | Less memory, regularization effect | Noisy gradients, slower |

### Our Experience

```
A100 80GB:
- batch_size=32, seq_len=1024 ✓
- batch_size=64, seq_len=1024 → OOM

T4 16GB:
- batch_size=8, seq_len=512 ✓
```

### Effective Batch Size

If memory is limited, use gradient accumulation:

```python
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## Data Loading Optimization

### DataLoader Parameters

```python
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,      # parallel loading
    pin_memory=True,    # faster GPU transfer
    prefetch_factor=2,  # prefetch batches
)
```

### A Pitfall We Hit

```python
# ❌ Wrong: reads the file from scratch every time
for epoch in range(100):
    data = load_all_data()  # slow!
    train(data)

# ✅ Right: preload + memory map
data = np.memmap('data.bin', dtype='uint16', mode='r')
dataloader = DataLoader(MemMapDataset(data), ...)
```

---

## Mixed Precision Training

### Why Use It

- FP16 computation is faster
- Memory usage halved
- Precision is sufficient for most cases

### PyTorch Implementation

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    with autocast():  # FP16 forward pass
        loss = model(batch)
    
    scaler.scale(loss).backward()  # scaled gradients
    scaler.step(optimizer)
    scaler.update()
```

---

## Our Complete Training Configuration

```python
config = {
    # Model
    "dim": 512,
    "n_layers": 9,
    "n_heads": 8,
    
    # Optimizer
    "optimizer": "adamw",
    "lr": 6e-4,
    "weight_decay": 0.1,
    "betas": (0.9, 0.95),
    
    # Training
    "batch_size": 32,
    "seq_len": 1024,
    "warmup_steps": 100,
    "max_steps": 2000,
    "grad_clip": 1.0,
    
    # Efficiency
    "mixed_precision": True,
    "compile": True,  # torch.compile
}
```

---

## Troubleshooting Guide

| Symptom | Possible Cause | Solution |
|---------|---------------|----------|
| Loss not decreasing | lr too small/large | Adjust lr |
| Loss explodes | Gradient explosion | Add gradient clipping |
| OOM | Batch too large | Reduce batch or use gradient accumulation |
| Training too slow | DataLoader bottleneck | Increase num_workers |
| Overfitting | Insufficient regularization | Add weight decay/dropout |

---

## 🚨 Cost Control (Lessons Learned the Hard Way)

### Our $184 Incident

```
Problem: Frequently polling training progress with process poll
Result: $184 spent in 7 minutes

Cause:
- Each poll returned thousands of lines of training logs
- All logs stored in session history
- Session grew to 58MB
- Claude Opus input costs $15/M tokens
```

### Solutions

1. **Don't poll long-running jobs**
2. **Write logs to a file, only send summaries**
3. **Use isolated cron jobs for monitoring**

```python
# ❌ Print every 50 steps
if step % 50 == 0:
    print(f"Step {step}, Loss {loss}")  # too much!

# ✅ Print every 500 steps
if step % 500 == 0:
    print(f"Step {step}, Loss {loss}")
```

---

*Previous: [Alternative Architectures](06-alternative-architectures.md)*

# Experiment Log & Lessons Learned

> Detailed record of our Parameter Golf journey — what we tried, what worked, and what didn't.

---

## Timeline

### Phase 1: Initial Exploration (2026-03-28)

#### Baseline Attempts

| Experiment | Model | BPB | Steps | Time | Params |
|------------|-------|-----|-------|------|--------|
| baseline_v1 | LatentLM | 7.66 | 200 | 57s | 1.2M |
| large_v1 | LatentLM | 6.69 | 492 | 300s | 15.5M |
| standard_gpt_v1 | StandardGPT | 5.21 | 500 | 185s | 26.5M |
| standard_gpt_long | StandardGPT | 4.37 | 868 | 600s | 26.5M |
| **standard_gpt_full_v2** | **StandardGPT** | **4.17** | 1786 | 600s | 26.5M |

**Key Finding**: StandardGPT >> LatentLM. Switched to standard architecture.

#### Optimizer Comparison

| Duration | AdamW | Muon |
|----------|-------|------|
| 15min | 4.07 | 3.99 ✅ |
| 30min | **3.73** | 3.91 |
| 60min | **3.55** 🏆 | 3.85 |
| 90min | **3.47** 🏆 | — |

**Conclusion**: Muon starts fast but AdamW wins long-term. Don't blindly trust paper claims.

---

### Phase 2: Architecture Exploration (2026-03-28 — 2026-03-30)

#### Mamba-3 Attempt

- **Goal**: O(n) complexity might allow more layers
- **Result**: 5.42 BPB (much worse)
- **Problem**: Pure PyTorch implementation too slow
- **Lesson**: Need official CUDA kernels; don't reinvent wheels

#### Weight Sharing (Universal Transformer)

- **Goal**: Save parameters by sharing weights across layers
- **Result**: 4.15 BPB vs 4.08 baseline
- **Conclusion**: Saves params but doesn't help BPB

#### Text Diffusion

- **Goal**: Non-autoregressive generation
- **Result**: Abandoned
- **Problem**: BPB evaluation incompatible with diffusion sampling

---

### Phase 3: Targeted Improvements (2026-03-30)

#### LeakyReLU²

- **Change**: `relu(x).square()` → `leaky_relu(x, 0.5).square()`
- **Result**: 2.3875 BPB (-4.3% vs baseline 2.4939)
- **Status**: ✅ Validated

#### Sliding Window Attention

- **Goal**: Local attention might help on short sequences
- **Problem**: RoPE position encoding overflowed for seq > max_seq_len
- **Fix**: Created `RotaryEmbeddingDynamic` with auto-expand
- **Result**: 2.3568 BPB (-5.5% vs baseline) 🏆
- **Status**: ✅ Current best

---

## Failed Approaches (Don't Repeat These)

### 1. Polling Training Logs

**What happened**: Used `process poll` every few seconds to monitor Modal training.

**Disaster**: 
- Each poll returned thousands of step logs
- All logs stored in session history
- Session grew to 58MB
- Claude Opus input costs $15/M tokens
- **7 minutes = $184** 💸

**Solution**: 
- Never poll long-running jobs
- Write logs to file, only send summary
- Use isolated cron for monitoring

### 2. Larger Model = Better?

**Assumption**: 37M params > 26.5M params → better BPB

**Reality**: 37M model got 4.31 BPB (worse than 26.5M's 4.17)

**Why**: Batch size had to be reduced due to memory, hurting training efficiency.

### 3. Trusting Paper Claims

**Muon paper**: "2x speedup over AdamW"

**Our result**: AdamW wins at 60+ minutes

**Lesson**: Always validate on YOUR specific setup.

---

## What We Haven't Tried Yet

| Technique | Why Promising | Status |
|-----------|---------------|--------|
| **LeakyReLU² + Sliding Window** | Combine two winners | Next up |
| **QAT (Quantization-Aware Training)** | Top teams use it | Research needed |
| **TTT (Test-Time Training)** | Leaderboard top uses it | Complex to implement |
| **Muon + tuned hyperparams** | Maybe our config was wrong | Low priority |

---

## Best Configuration (Current)

```python
model_config = {
    "model_type": "standard_gpt",
    "dim": 512,
    "n_layers": 9,
    "n_heads": 8,
    "vocab_size": 50257,
    "max_seq_len": 1024,
    "activation": "swiglu",  # or "leaky_relu_squared"
    "use_sliding_window": True,
    "window_size": 256,
}

train_config = {
    "optimizer": "adamw",
    "lr": 0.0006,
    "batch_size": 32,
    "max_seconds": 600,  # 10 min for competition
}
```

---

## Cost Summary

| Item | Cost |
|------|------|
| Modal compute | ~$10 |
| Accidental poll disaster | $184 |
| **Total** | ~$194 |

**Lesson**: Infrastructure mistakes cost more than experiments.

---

*Last updated: 2026-03-30*

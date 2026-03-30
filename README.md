# Parameter Golf Solution

> Our attempt at the [Parameter Golf Competition](https://parameter.golf) — training a small language model in 10 minutes on 8×H100 to minimize BPB.

## Current Best: 2.3568 BPB 🏆

| Model | BPB | Notes |
|-------|-----|-------|
| Sliding Window + RoPE Dynamic | **2.3568** | Current best, -5.5% vs baseline |
| LeakyReLU² | 2.3875 | -4.3% vs baseline |
| Baseline (SwiGLU) | 2.4939 | StandardGPT reference |

**Target**: < 1.13 BPB (leaderboard top: 1.1194)

## Competition Rules

- Submit **training code**, not pretrained weights
- Must complete training in **10 minutes on 8×H100**
- Final model: **≤16MB @ 3-bit quantization**
- Metric: **Bits Per Byte (BPB)** on held-out test set

## Architecture

```
StandardGPT
├── dim: 512
├── layers: 9
├── heads: 8
├── params: ~4.4M (experiment) / ~26.5M (full)
└── activation: SwiGLU / LeakyReLU²
```

## Experiments Summary

### ✅ What Worked

| Technique | BPB Impact | Notes |
|-----------|------------|-------|
| **Sliding Window Attention** | -5.5% | Local attention helps on short sequences |
| **LeakyReLU²** | -4.3% | `leaky_relu(x, 0.5).square()` |
| **RoPE Dynamic** | Bug fix | Auto-expand cos/sin table |
| **AdamW** | Better than Muon | Stable for long training |

### ❌ What Didn't Work

| Technique | Result | Lesson |
|-----------|--------|--------|
| **Mamba-3** | 5.42 BPB | Pure PyTorch too slow; needs CUDA kernel |
| **Muon Optimizer** | 3.85 vs 3.55 | Fast start but AdamW wins long-term |
| **Weight Sharing** | 4.15 vs 4.08 | Saves params but not BPB |
| **Text Diffusion** | Abandoned | BPB evaluation incompatible |

### 🔬 Next to Try

- [x] ~~LeakyReLU² + Sliding Window combo~~ → **In progress** (subagent running)
- [ ] QAT (Quantization-Aware Training)
- [ ] TTT (Test-Time Training)
- [ ] Muon optimizer revisit with tuned hyperparams

## Project Structure

```
├── experiment.py          # Main training script
├── run_experiments.py     # Batch experiment runner
├── run_combo.py           # Combined techniques
├── models/
│   ├── standard_gpt.py    # Main model
│   ├── mamba_lm.py        # Mamba attempt
│   └── latent_lm.py       # Earlier architecture
├── configs/               # Experiment configs
└── modal_runner.py        # Modal cloud runner
```

## Running Experiments

```bash
# Local test
python test_local.py

# Modal cloud (A100/H100)
modal run experiment.py::run_experiment \
  --experiment-name "my_exp" \
  --max-seconds 600

# Resume from checkpoint
modal run experiment.py::run_experiment \
  --experiment-name "my_exp" \
  --resume-from "my_exp" \
  --max-seconds 3600
```

## Cost Tracking

- Modal credits used: ~$10
- **Lesson learned**: Don't poll training logs! Cost $184 in 7 minutes due to session history bloat.

## References

- [Parameter Golf Official Site](https://parameter.golf)
- [Leaderboard](https://parameter.golf/leaderboard)
- Top techniques: TTT, QAT, BigramHash, Muon

---

*Last updated: 2026-03-30*

# Experiment Report: Sandwich MLP vs Uniform MLP

## 1. Overview

| Item | Content |
|------|---------|
| Name | Sandwich MLP on Alternating Attention + mHC |
| Purpose | Validate Sandwich MLP (large-small-large) vs Uniform MLP, based on mHC β_mlp observations |
| Hypothesis | Middle layers need less MLP capacity, matching mHC β_mlp pattern |
| Config | 20 layers, dim=384, 11 Global + 9 Local, window=128, mHC from scratch |
| Start | 2026-04-04 11:10 SGT |
| End | 2026-04-04 11:37 SGT |
| Duration | 27 min (1633s) |

## 2. Results

| Metric | Uniform | Sandwich | Delta |
|--------|---------|----------|-------|
| Val Loss | 3.7590 | 3.7734 | +0.38% |
| Val BPB | **1.4777** | **1.4833** | **+0.38%** |
| Total Params | 29.70M | **22.79M** | **-23.3%** |
| MLP Params | 13.27M | **10.79M** | **-18.7%** |

| vs Reference | Uniform | Sandwich |
|-------------|---------|----------|
| vs Baseline (1.5025) | **-1.65%** | **-1.28%** |

## 3. MLP Configuration

| Layers | Uniform MLP Hidden | Sandwich MLP Hidden | Sandwich Scale |
|--------|--------------------|--------------------|---------------| 
| 0-3 | 1152 | 1152 | 3.0x |
| 4-16 | 1152 | **460** | **1.2x** |
| 17-19 | 1152 | 1152 | 3.0x |

## 4. mHC Parameter Analysis

### 4.1 Sandwich mHC Parameters

| Layer | Type | MLP | a_attn | b_attn | a_mlp | b_mlp |
|-------|------|-----|--------|--------|-------|-------|
| 0 | Global | 3.0x | 1.250 | 0.107 | 0.960 | 0.940 |
| 1 | Local | 3.0x | 0.951 | 0.830 | 1.003 | 0.733 |
| 2 | Global | 3.0x | 0.973 | 0.615 | 0.956 | 0.716 |
| 3 | Local | 3.0x | 0.945 | 0.685 | 0.940 | 0.683 |
| 4 | Global | 1.2x | 0.925 | 0.733 | 0.924 | 0.659 |
| 5 | Local | 1.2x | 0.899 | 0.651 | 0.906 | 0.526 |
| 6 | Global | 1.2x | 0.866 | 1.277 | 1.011 | 0.666 |
| 7 | Local | 1.2x | 1.005 | 0.711 | 1.003 | 0.703 |
| 8 | Global | 1.2x | 1.000 | 0.775 | 1.000 | 0.728 |
| 9 | Local | 1.2x | 0.998 | 0.641 | 0.991 | 0.749 |
| 10 | Global | 1.2x | 0.990 | 0.655 | 0.985 | 0.739 |
| 11 | Local | 1.2x | 0.984 | 0.672 | 0.977 | 0.750 |
| 12 | Global | 1.2x | 0.975 | 0.759 | 0.977 | 0.724 |
| 13 | Local | 1.2x | 0.982 | 0.686 | 0.975 | 0.728 |
| 14 | Global | 1.2x | 0.975 | 0.680 | 0.961 | 0.720 |
| 15 | Local | 1.2x | 0.960 | 0.693 | 0.955 | 0.714 |
| 16 | Global | 1.2x | 0.952 | 0.643 | 0.940 | 0.682 |
| 17 | Local | 3.0x | 0.941 | 0.632 | 0.925 | 0.661 |
| 18 | Global | 3.0x | 0.926 | 0.660 | 0.912 | 0.642 |
| 19 | Global | 3.0x | 0.927 | 0.557 | 0.896 | 0.618 |

### 4.2 Observations

1. **β_mlp pattern is flatter**: Unlike Uniform's V-shape (high-low-high), Sandwich β_mlp is more monotone decreasing (0.940 → 0.618)
2. **Layer 0 anomaly**: α_attn=1.250, β_attn=0.107 — first layer strongly prefers residual path over attention
3. **Layer 6 anomaly**: β_attn=1.277 — a Global layer with unusually high attention weight
4. **Deep layers decline**: Both α and β decrease toward the end, suggesting information accumulation

## 5. Findings

### 5.1 As Expected
- Sandwich MLP works: BPB 1.4833, only +0.38% worse than Uniform
- mHC parameters adapt to Sandwich architecture (flatter β_mlp)
- 23% parameter reduction for only 0.38% BPB cost

### 5.2 Unexpected
- **Deep layers don't need larger MLP**: β_mlp keeps decreasing to the end, contradicting the original hypothesis that layers 17-19 need high MLP capacity
- **Sandwich β_mlp lacks V-shape**: The mHC pattern that motivated Sandwich (high-low-high) is weaker in the Sandwich model itself

### 5.3 Implications
- Sandwich MLP is a valid parameter-efficient strategy
- Combined with QAT quantization, Sandwich could enable larger dim within 16MB
- The 1.2x middle scale may be too aggressive — 1.5x might be worth trying

## 6. Conclusion

- **Result**: Partial success. Sandwich saves 23% params for only 0.38% BPB cost
- **Adoption**: Yes for QAT experiments (allows larger dim in 16MB budget)
- **Key lesson**: mHC-informed architecture design is valid but the self-consistency is weaker than expected

## 7. Next Steps

| Direction | Description | Priority |
|-----------|-------------|----------|
| Sandwich + QAT | Quantize Sandwich model, check 16MB budget | High |
| 1.5x middle scale | Try less aggressive compression | Medium |
| Deeper model | Use saved params for Sandwich + deeper (30L) model | Medium |

## 8. Execution

### Script
- File: `scripts/modal/modal_sandwich_mlp.py`
- URL: https://github.com/Elarwei001/parameter-golf-solution/blob/master/scripts/modal/modal_sandwich_mlp.py

### Command
```bash
modal run --detach scripts/modal/modal_sandwich_mlp.py --style sandwich
```

### Identifiers
- Modal App: `ap-aviHMq9hP49Ec0V7rVNryo`
- Checkpoint: `/data/checkpoints/sandwich_mlp/sandwich/sandwich_sandwich_step5000.pt`
- Results: `/data/checkpoints/sandwich_mlp/sandwich/sandwich_sandwich_bpb1.4833.json`

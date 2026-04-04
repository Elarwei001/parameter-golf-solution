# Experiment Report: Front-loaded MLP (No Deep Recovery)

## 1. Overview

| Item | Content |
|------|---------|
| Name | Front-loaded MLP (all deep layers use small MLP) |
| Purpose | Test whether deep layers need 3x MLP, motivated by Sandwich's declining beta_mlp pattern |
| Hypothesis | Sandwich's beta_mlp keeps declining to the end, suggesting deep layers don't need large MLP |
| Config | 20 layers, dim=384, 11 Global + 9 Local, window=128, mHC from scratch |
| Start | 2026-04-04 13:10 SGT |
| End | 2026-04-04 14:30 SGT |
| Duration | ~80 min |

## 2. Results

| Metric | Uniform | Sandwich | Front | 
|--------|---------|----------|-------|
| Val Loss | 3.7590 | 3.7734 | 3.7949 |
| Val BPB | **1.4777** | 1.4833 | 1.4918 |
| Total Params | 29.70M | 22.79M | **21.20M** |
| MLP Params | 13.27M | 10.79M | **9.19M** |

| vs Reference | Uniform | Sandwich | Front |
|-------------|---------|----------|-------|
| vs Baseline (1.5025) | **-1.65%** | -1.28% | -0.71% |
| vs Uniform | - | +0.38% | **+0.95%** |

## 3. MLP Configuration

| Layers | Uniform | Sandwich | Front |
|--------|---------|----------|-------|
| 0-3 | 3x (1152) | 3x (1152) | 3x (1152) |
| 4-16 | 3x (1152) | 1.2x (460) | 1.2x (460) |
| 17-19 | 3x (1152) | **3x (1152)** | **1.2x (460)** |

## 4. mHC Parameter Analysis

### 4.1 Front mHC Parameters

| Layer | Type | MLP | a_attn | b_attn | a_mlp | b_mlp |
|-------|------|-----|--------|--------|-------|-------|
| 0 | Global | 3.0x | 1.234 | 0.139 | 0.988 | 0.932 |
| 1 | Local | 3.0x | 0.944 | 0.835 | 0.994 | 0.731 |
| 2 | Global | 3.0x | 0.967 | 0.678 | 0.961 | 0.718 |
| 3 | Local | 3.0x | 0.941 | 0.779 | 0.937 | 0.667 |
| 4 | Global | 1.2x | 0.929 | 0.955 | 0.952 | 0.670 |
| 5 | Local | 1.2x | 0.945 | 0.958 | 0.979 | 0.743 |
| 6 | Global | 1.2x | 0.974 | 0.713 | 0.973 | 0.735 |
| 7 | Local | 1.2x | 0.969 | 0.731 | 0.968 | 0.714 |
| 8 | Global | 1.2x | 0.960 | 0.695 | 0.957 | 0.691 |
| 9 | Local | 1.2x | 0.961 | 0.639 | 0.947 | 0.681 |
| 10 | Global | 1.2x | 0.937 | 1.141 | 1.004 | 0.706 |
| 11 | Local | 1.2x | 1.006 | 0.599 | 0.982 | 0.730 |
| 12 | Global | 1.2x | 0.972 | 0.714 | 0.972 | 0.725 |
| 13 | Local | 1.2x | 0.976 | 0.715 | 0.971 | 0.712 |
| 14 | Global | 1.2x | 0.977 | 0.614 | 0.964 | 0.716 |
| 15 | Local | 1.2x | 0.964 | 0.632 | 0.954 | 0.700 |
| 16 | Global | 1.2x | 0.954 | 0.734 | 0.954 | 0.676 |
| 17 | Local | 1.2x | 0.964 | 0.512 | 0.941 | 0.656 |
| 18 | Global | 1.2x | 0.953 | 0.454 | 0.922 | 0.654 |
| 19 | Global | 1.2x | 0.953 | 0.578 | 0.908 | 0.604 |

### 4.2 Key Comparison: Sandwich vs Front

**beta_mlp (MLP dependency)**: Nearly identical across both configs. Both show steady decline from ~0.93 to ~0.60. MLP scale does not change how much the model relies on MLP output.

**beta_attn (Attention dependency)**: Most informative difference.

| Feature | Sandwich | Front |
|---------|----------|-------|
| Layers 4-5 (start of 1.2x) | 0.733, 0.651 | **0.955, 0.958** |
| Layer 18 (deep) | 0.660 | **0.454** |
| Layer 10 anomaly | 0.655 | **1.141** |

Front shows **elevated beta_attn at layers 4-5** (0.955 vs 0.733) and **depressed beta_attn at layer 18** (0.454 vs 0.660). This indicates:

1. **Compensation mechanism**: When deep MLP capacity is removed, middle layers increase attention weight to compensate
2. **Signal attenuation**: Deep layers in Front have lower attention weight, suggesting information quality degrades
3. **Incomplete compensation**: Despite higher mid-layer attention, Front still performs worse (+0.57% vs Sandwich)

### 4.3 Visualization

See `mhc_sandwich_vs_front.png` for the full parameter comparison chart.

## 5. Findings

### 5.1 As Expected
- Front-loaded MLP works: BPB 1.4918, still beats baseline (-0.71%)
- mHC adapts to architecture (elevated mid-layer attention in Front)
- Deep layer MLP reduction is possible but costly

### 5.2 Unexpected
- **beta_mlp is nearly identical between Sandwich and Front**: MLP scale doesn't affect how much the model depends on MLP output. The model uses MLP the same amount regardless of its size.
- **Front's mid-layer beta_attn surge**: Layers 4-5 jump to ~0.96, much higher than Sandwich's ~0.69. This is a clear compensation signal.

### 5.3 Self-Consistency Paradox Resolution

The original hypothesis was: "Sandwich's beta_mlp declines to the end, so deep layers don't need large MLP."

**This hypothesis is WRONG.** While beta_mlp does decline, the deep 3x MLP still contributes meaningfully:
- Sandwich (deep 3x): BPB 1.4833
- Front (deep 1.2x): BPB 1.4918 (+0.57%)

The declining beta_mlp does NOT mean "MLP is unimportant" — it means "MLP's relative contribution decreases with depth" while still being essential at full capacity.

**Key insight**: beta_mlp measures relative importance, not absolute necessity. A small beta with a large MLP can still contribute more than a small beta with a small MLP.

## 6. Conclusion

- **Result**: Negative. Front-loaded MLP degrades 0.57% more than Sandwich.
- **Adoption**: No. Sandwich (3x/1.2x/3x) remains the optimal MLP configuration.
- **Key lesson**: Deep layers need full MLP capacity despite declining beta_mlp. The Sandwich design (shallow-large, middle-small, deep-large) is validated by experiment, even though the original mHC pattern that motivated it shows a self-consistency paradox.

## 7. Experiment Ranking

| Rank | Config | BPB | Params | Trade-off |
|------|--------|-----|--------|-----------|
| 1 | Uniform (all 3x) | **1.4777** | 29.70M | Best BPB, most params |
| 2 | Sandwich (3x/1.2x/3x) | 1.4833 | 22.79M | Best efficiency (-23% params, +0.38% BPB) |
| 3 | Front (3x/1.2x/1.2x) | 1.4918 | 21.20M | Over-compressed deep layers |

## 8. Next Steps

| Direction | Description | Priority |
|-----------|-------------|----------|
| Sandwich + QAT | Quantize Sandwich model for 16MB budget | High |
| Uniform + QAT | Quantize Uniform model, check 16MB fit | High |
| Try 1.5x middle | Sandwich with 1.5x instead of 1.2x for middle layers | Medium |

## 9. Execution

### Script
- File: `scripts/modal/modal_sandwich_mlp.py`
- URL: https://github.com/Elarwei001/parameter-golf-solution/blob/master/scripts/modal/modal_sandwich_mlp.py

### Command
```bash
modal run --detach scripts/modal/modal_sandwich_mlp.py --style front
```

### Identifiers
- Modal App: `ap-54p3P6UVNGffNEDY0HNMk2`
- Checkpoint: `/data/checkpoints/sandwich_mlp/front/sandwich_front_step5000.pt`
- Results: `/data/checkpoints/sandwich_mlp/front/sandwich_front_bpb1.4918.json`

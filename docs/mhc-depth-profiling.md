# mHC Depth Profiling — A Method for Optimal Layer Count Determination

## Motivation

When training deep Transformers, how do we know how many layers to use? Traditional approaches rely on brute-force experimentation — trying 10L, 15L, 20L, 30L and comparing BPB.

mHC (multi-Head Calibration) parameters provide a more elegant solution: **train one over-deep model, analyze the β parameter decay curve, and identify where layers become redundant.**

## The β Parameter

mHC adds per-layer learnable coefficients:

```
output = α * input + β * sublayer(input)
```

- **α** (direct path): how much of the original input passes through unchanged
- **β** (residual path): how much the attention/MLP output contributes

When β → 0, the layer is doing near-identity mapping — it's not learning useful transformations.

## Key Findings from Our Experiments

### 20L Model (dim=384, BPE 8192)
- β_attn: 0.56 - 1.07 across all layers
- β_mlp: 0.63 - 0.87 across all layers
- **All 20 layers contribute meaningfully** (β > 0.5 throughout)

### 30L Model (dim=448, BPE 8192)
- β_attn: drops from 0.12 → 0.003 after Layer 10
- β_mlp: drops from 0.56 → 0.05 after Layer 12
- **Layers 12-29 are near-identity** (β < 0.2)

### The Paradox

20L at Layer 17 still has β = 0.6-0.7 (productive), but 30L at the same relative depth (~57%) has β = 0.19 (near-idle). **The model adapts its behavior to total depth** — it doesn't simply "run out of useful work" at a fixed depth.

Possible explanations:
1. **Gradient dilution**: With 30 layers, gradients reaching early layers are weaker, so the model front-loads learning
2. **Optimization difficulty**: 30 layers are harder to optimize, so the model finds a local minimum that under-uses deep layers
3. **Dim effect**: 30L used dim=448 vs 20L's dim=384 — different dim may affect β distribution (this is a confounding variable)

## Methodology: mHC Depth Profiling

### Step 1: Train with mHC
Add mHC coefficients to every layer and train normally.

### Step 2: Plot β curves
After training, extract β_attn and β_mlp for each layer and plot against normalized depth.

### Step 3: Identify the "β floor"
Find the layer where β drops below a threshold (e.g., β < 0.1 = 10% of max β).

### Step 4: Compare across depths
Train models at multiple depths (10L, 15L, 20L, 25L, 30L) and compare β curves. The optimal depth is the maximum where **all layers maintain β significantly above the floor**.

### Important: Control Variables
- Keep dim, vocab, MLP scales constant across depth experiments
- Our current data has confounds (20L=dim384, 30L=dim448) — need controlled ablation

## Implications for Architecture Design

1. **More layers ≠ better** if β decays — wasted parameters
2. **Depth recurrence is well-motivated** — if deep layers do near-identity, sharing weights has minimal cost
3. **Layer pruning is safe** — layers with β ≈ 0 can be removed without performance loss
4. **Adaptive depth** — could use β values to dynamically skip layers during inference

## Open Questions

- Does the β decay pattern hold across different dim values?
- Is the 20L vs 30L difference caused by depth or by dim (384 vs 448)?
- Would a 20L dim=448 model show the same high β throughout?
- Can we use β as a training signal to encourage deeper utilization?

## References

- Deep Equilibrium Models (Bai et al., 2019) — infinite-depth networks converge to fixed points
- Shattered Gradients (Balduzzi et al., 2017) — deep ResNets primarily use skip paths
- Layer Dropout (Fan et al., 2019) — many layers can be dropped with minimal impact

---

*Created: 2026-04-06*
*Based on experiments: dim=384 20L, dim=448 30L*

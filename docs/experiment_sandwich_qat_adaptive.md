# Sandwich QAT Adaptive Experiment Report

**Date**: 2026-04-04
**App ID**: ap-6RAgmDfOfrU4jLslN7hJwv
**Script**: `scripts/modal/modal_sandwich_qat.py`

---

## Objective

Test whether a **warm-start (adaptive) QAT** strategy improves over training with QAT enabled from step 0.

**Hypothesis**: Let the model train in FP32 first, then switch to QAT once loss converges. This should give the model a better starting point before introducing quantization noise.

## Method

### Adaptive QAT Algorithm
- Monitor EMA loss during FP32 training
- When `loss_rate < 0.001` AND `step >= 500` (warmup floor) → switch to QAT
- EMA smoothing: α = 0.99

### Model Configuration
| Parameter | Value |
|-----------|-------|
| Architecture | Sandwich MLP + Alternating Attention |
| Dim | 384 |
| Layers | 20 |
| Heads | 8 (KV: 4) |
| MLP Scales | [3.0]×4 + [1.2]×13 + [3.0]×3 (sandwich) |
| mHC | 4 params/layer, FP32, init=1.0 |
| QAT | Ternary (1.58-bit) with STE |
| Params | 22.79M total (19.63M ternary + 3.16M FP32) |
| Quantized Size | 16.52 MB |

### Training Configuration
| Parameter | Value |
|-----------|-------|
| Steps | 5000 |
| LR | 1e-3, cosine decay |
| Warmup | 500 steps |
| Batch Size | 64 × 256 tokens |
| Optimizer | AdamW (weight_decay=0.1) |
| GPU | A100-40GB |
| Training Time | 1670s (~28 min) |

---

## Results

### QAT Switch Point
- **QAT enabled at step 500** (the minimum warmup floor)
- EMA Loss at switch: 5.5048
- Loss rate at switch: 0.000534 < threshold 0.001
- FP32 phase: 500 steps, QAT phase: 4500 steps

### Performance Comparison

| Experiment | Val Loss | Val BPB | vs Baseline |
|------------|----------|---------|-------------|
| **Sandwich QAT Adaptive** | **3.8973** | **1.5321** | **+1.97%** |
| Sandwich QAT (from step 0) | 3.9049 | 1.5350 | +2.16% |
| mHC v2 20L baseline (FP32) | 3.8222 | 1.5025 | baseline |
| Sandwich MLP FP32 | - | 1.4833 | -1.28% |

### Key Observations

1. **Adaptive QAT slightly better than from-step-0 QAT**: 1.5321 vs 1.5350 BPB (-0.19% improvement)
2. **Still worse than FP32 baseline**: +1.97% degradation vs the 1.5025 baseline
3. **Model exceeds 16MB budget**: 16.52 MB quantized size
4. **QAT switched very early**: Loss converged at step 500, possibly too early — the model was still at loss ~5.2

---

## mHC Parameter Evolution

See: `docs/mhc_3d_evolution_adaptive_qat.png`

### Final mHC Parameters (Step 5000)

| Layer | Type | MLP | α_attn | β_attn | α_mlp | β_mlp |
|-------|------|-----|--------|--------|-------|-------|
| 0 | Global | 3.0x | 1.251 | 0.183 | 1.264 | 0.870 |
| 1 | Local | 3.0x | 0.925 | 0.808 | 0.986 | 0.864 |
| 2 | Global | 3.0x | 0.987 | 0.591 | 0.952 | 0.854 |
| 3 | Local | 3.0x | 0.960 | 0.664 | 0.949 | 0.837 |
| 4-16 | Mix | 1.2x | 0.96±0.03 | 0.70±0.10 | 0.97±0.02 | 0.76±0.03 |
| 17 | Local | 3.0x | 0.943 | 0.671 | 0.924 | 0.687 |
| 18 | Global | 3.0x | 0.929 | 0.576 | 0.904 | 0.665 |
| 19 | Global | 3.0x | 0.925 | 0.560 | 0.883 | 0.632 |

**Patterns**:
- Layer 0 has unique behavior: high α (preserves input), low β (weak attention) — first layer acts as a passthrough
- Deeper layers show monotonic decrease in all parameters
- β values (sublayer scaling) are consistently lower than α values (residual scaling)

---

## Conclusions

1. **Adaptive QAT provides marginal improvement** over from-step-0 QAT (0.19% BPB), but the gain is small
2. **QAT still degrades performance** compared to FP32 baseline — the 1.58-bit ternary quantization costs ~2% BPB
3. **Switch timing may be suboptimal** — step 500 is very early, loss was still high (5.2). Consider:
   - Increasing `qat_warmup_steps` to 1000-2000
   - Decreasing `qat_switch_threshold` to 0.0005
   - Using a stricter convergence criterion (e.g., 3 consecutive checks below threshold)
4. **Model size issue**: 16.52 MB exceeds the 16MB budget — need to reduce params or increase quantization

## Next Steps

- [ ] Try longer warmup (1000, 2000 steps) before QAT switch
- [ ] Reduce model size to fit 16MB budget
- [ ] Consider mixed-precision: keep more layers in FP32, fewer in ternary
- [ ] Compare with post-training quantization (train FP32, then quantize)

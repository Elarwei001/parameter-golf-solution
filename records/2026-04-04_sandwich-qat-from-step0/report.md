# Experiment Report: Sandwich MLP + QAT + mHC

## 1. Overview

| Item | Content |
|------|---------|
| Name | Sandwich MLP + QAT (Ternary 1.58-bit) + mHC from scratch |
| Purpose | Combine Sandwich MLP architecture with QAT quantization; observe mHC parameter evolution under quantized training |
| Hypothesis | QAT + Sandwich can achieve good BPB within 16MB budget; mHC parameters adapt to quantized weights |
| Config | 20 layers, dim=384, Sandwich MLP (3x/1.2x/3x), QAT from step 0, mHC from scratch |
| Start | 2026-04-04 14:53 SGT |
| End | 2026-04-04 15:21 SGT |
| Duration | 28 min (1663s) |

## 2. Results

| Metric | Sandwich FP32 | Sandwich + QAT | Delta |
|--------|--------------|----------------|-------|
| Val Loss | 3.7734 | 3.9049 | +3.5% |
| Val BPB | **1.4833** | **1.5350** | **+3.5%** |
| Total Params | 22.79M | 22.79M | same |
| Quantized Size | N/A | **16.52 MB** | - |

| vs Reference | Sandwich FP32 | Sandwich + QAT |
|-------------|--------------|----------------|
| vs Baseline (1.5025) | **-1.28%** | **+2.16%** |
| vs Alt-A mHC-scratch Uniform (1.4777) | +0.38% | **+3.90%** |

**Verdict**: QAT from step 0 causes significant degradation (+3.5%). The quantized model is worse than baseline.

## 3. Model Size Analysis

| Component | Ternary (1.58-bit) | FP32 | Total |
|-----------|-------------------|------|-------|
| Attention weights | - | - | - |
| MLP weights | - | - | - |
| **All QATLinear** | **19.63M params** | **3.88 MB** | - |
| Embedding (shared) | - | 3.15M params | 12.60 MB |
| RMSNorm + mHC | - | 0.01M params | 0.05 MB |
| **Total** | **19.63M** | **3.16M** | **16.52 MB** |

**Problem**: Embedding/LM Head (shared, FP32) alone takes 12.60 MB, leaving only 3.40 MB for quantized weights. This is the bottleneck.

## 4. mHC Parameter Evolution

### 4.1 Training Loss Progression

| Step | Loss | LR |
|------|------|-----|
| 500 | 5.1714 | 1.00e-3 |
| 1000 | 4.6980 | 9.73e-4 |
| 2000 | 4.2637 | 7.75e-4 |
| 3000 | 4.0695 | 4.72e-4 |
| 4000 | 4.0446 | 2.05e-4 |
| 5000 | 3.8996 | 1.00e-4 |

### 4.2 Key mHC Evolution Patterns

**beta_attn (attention weight)**:
- Starts high (~0.85-0.96 at step 1000), gradually decreases
- Layer 0 drops sharply: 0.453 → 0.210 (first layer learns to ignore attention)
- Layer 5 stays elevated (~1.0) throughout - a Local layer compensating
- By step 5000, most layers settle to 0.5-0.8 range

**beta_mlp (MLP weight)**:
- Very stable across all steps! Changes <0.05 per layer between steps
- QAT beta_mlp values are systematically HIGHER than FP32 Sandwich
  - QAT layer 10: 0.833 vs FP32: 0.739
  - QAT layer 19: 0.636 vs FP32: 0.618
- This suggests the model compensates for quantized attention by relying more on MLP

### 4.3 QAT vs FP32 Final Comparison

**beta_attn**: QAT systematically higher in middle layers (0.6-1.0 vs FP32's 0.5-0.8). Model compensates for quantized attention weights by increasing attention weight via mHC.

Wait - that's counterintuitive. If attention weights are worse (quantized), why increase beta_attn? Actually the pattern is mixed:
- Layer 0: QAT 0.210 vs FP32 0.107 (both low, QAT slightly higher)
- Layers 1-4: QAT 0.78-0.88 vs FP32 0.62-0.83 (QAT higher)
- Layer 6: FP32 has anomaly 1.277, QAT only 0.674
- Deep layers 17-19: QAT 0.50-0.61 vs FP32 0.56-0.66 (similar)

**beta_mlp**: QAT systematically higher (0.63-0.84 vs FP32 0.52-0.94). The model relies more heavily on MLP to compensate for information loss from quantized weights.

### 4.4 Visualization

See `mhc_qat_evolution.png` for the full parameter evolution chart.

## 5. Findings

### 5.1 As Expected
- QAT causes performance degradation (+3.5%)
- mHC parameters adapt to quantized training
- Quantized size 16.52 MB, slightly over 16MB limit

### 5.2 Unexpected
- **Embedding is the bottleneck, not the linear layers!** The shared embedding takes 12.60 MB in FP32, leaving only 3.40 MB for all quantized weights
- **QAT from step 0 is too aggressive** — the model never gets a chance to learn good representations before quantization noise is introduced
- **beta_mlp is higher under QAT** — the model uses MLP more to compensate for quantized attention/MLP weights, which is somewhat counterintuitive since MLP weights are also quantized

### 5.3 Critical Insight: Embedding Bottleneck

The 16MB constraint with FP32 embedding leaves almost no room for the model:

```
Budget:           16.00 MB
Embedding (FP32): 12.60 MB (78.8% of budget!)
Remaining:         3.40 MB for all 19.63M quantized params
```

This means either:
1. Quantize the embedding too (but this hurts token representations)
2. Use a smaller vocab (but BPE-8192 is standard for the competition)
3. Use weight tying + quantized embedding

## 6. Conclusion

- **Result**: Failed to improve. BPB 1.5350 is worse than baseline.
- **Root cause**: QAT from step 0 is too aggressive; embedding bottleneck leaves insufficient budget
- **Key lesson**: Need to either (a) warm-start with FP32 then QAT, or (b) quantize embedding too

## 7. Next Steps

| Direction | Description | Priority |
|-----------|-------------|----------|
| QAT warm-start | Train FP32 2000 steps, then enable QAT | High |
| Quantize embedding | Try quantizing embedding/LM head too | Medium |
| Larger dim + QAT | Since QAT saves space, try dim=448+ with QAT | Medium |
| FP16 embedding | Use FP16 for embedding (saves 50%) | Easy win |

## 8. Log Truncation Note

`modal app logs` defaults to a limited tail buffer. Use `--tail 500` to see full logs:
```bash
modal app logs --tail 500 <app-id>
```

## 9. Execution

### Script
- File: `scripts/modal/modal_sandwich_qat.py`
- URL: https://github.com/Elarwei001/parameter-golf-solution/blob/master/scripts/modal/modal_sandwich_qat.py

### Command
```bash
modal run --detach scripts/modal/modal_sandwich_qat.py
```

### Identifiers
- Modal App: `ap-5vP4DjouTi7QkXUTmio8T5`
- Checkpoint: `/data/checkpoints/sandwich_qat/sandwich_qat_step5000.pt`
- Results: `/data/checkpoints/sandwich_qat/sandwich_qat_bpb1.5350.json`

### mHC History (saved in results JSON)
- Steps logged: 1000, 2000, 3000, 4000, 5000
- Full mHC evolution tracked in `mhc_history` field

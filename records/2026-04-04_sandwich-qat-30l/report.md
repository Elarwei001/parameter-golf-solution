# Sandwich QAT 30-Layer Experiment

**Date**: 2026-04-04
**App ID**: ap-SQaNNoc4QrGXezXR5Hxt6H
**Script**: `scripts/modal/modal_sandwich_qat_30l.py`
**Training Log**: `train.log`

---

## Objective

Test whether increasing model depth from 20L to 30L improves BPB, while keeping dim=384 and all other hyperparameters unchanged.

**Single variable change**: layers 20 → 30

## Configuration

| Parameter | Value | vs 20L |
|-----------|-------|--------|
| Layers | **30** | +10 |
| Dim | 384 | same |
| Heads | 8 (KV: 4) | same |
| Sandwich MLP | [3.0]×6 + [1.2]×21 + [3.0]×3 | [3.0]×4 + [1.2]×13 + [3.0]×3 |
| mHC | 4 params/layer, FP32 | same |
| QAT | Ternary (1.58-bit), adaptive | same |
| Params | 31.82M (ternary: 28.65M, FP32: 3.17M) | was 22.79M |
| Steps | 5000 | same |
| LR | 1e-3, cosine decay | same |

## Results

### QAT Switch Point
- QAT enabled at step 500 (minimum warmup floor, same as 20L)
- EMA Loss: 5.5107, Loss rate below threshold

### Performance Comparison

| Experiment | Layers | Params | Val Loss | Val BPB | vs FP32 Baseline |
|------------|--------|--------|----------|---------|-----------------|
| FP32 baseline | 20 | 22.79M | 3.8222 | 1.5025 | baseline |
| QAT adaptive | 20 | 22.79M | 3.8973 | 1.5321 | +1.97% |
| **QAT adaptive** | **30** | **31.82M** | **3.8706** | **1.5215** | **+1.27%** |

### Key Findings

1. **30L 比 20L QAT 好了 0.0106 BPB** (-0.69%)，加深有效
2. **仍然比 FP32 baseline 差 +1.27%**，QAT 量化退化依然存在
3. **参数量增加 40%**（22.79M → 31.82M），但 BPB 改善不大，说明 QAT 限制了深层模型的收益
4. **训练时间增加 46%**（1670s → 2439s），符合预期

### mHC Parameter Observations

- Layer 0: α_attn=1.225, α_mlp=1.199 (高残差保留，与 20L 一致)
- 中间层 α ≈ 0.97-0.99, β_mlp ≈ 0.74-0.82
- 深层 (27-29): 所有参数都在下降，最深 β_mlp=0.621
- Layer 10 β_attn 出现异常高值 1.052

## Conclusions

1. **加深层数有效**：30L (1.5215) > 20L (1.5321)，但改善幅度有限
2. **QAT 退化是主要瓶颈**：无论 20L 还是 30L，QAT 都比 FP32 差 ~1-2%
3. **下一步建议**：
   - 加大 dim（30L dim=470 可塞进 16MB）
   - 或尝试更长 warmup 减少 QAT 退化

## Next Steps

- [ ] 30L + dim=470 实验（45M params, 15.97 MB）
- [ ] 30L + 更长 QAT warmup (1000-2000 steps)

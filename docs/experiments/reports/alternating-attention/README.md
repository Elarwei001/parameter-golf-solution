# Alternating Attention 实验系列

> 偶数层 Global + 奇数层 Local (window=128) + 最后一层强制 Global

## 实验列表

| 实验 | 日期 | BPB | vs Baseline | 状态 |
|------|------|-----|-------------|------|
| [Alt-A (Vanilla)](2026-04-03_alt-a-vanilla.md) | 2026-04-03 | 1.5001 | -0.16% | ✅ |
| [Alt-A + mHC (迁移)](2026-04-03_alt-a-mhc.md) | 2026-04-03 | 1.4883 | -0.95% | ✅ |
| [Alt-B (dim=448)](2026-04-04_alt-b.md) | 2026-04-04 | 1.4952 | -0.49% | ✅ |
| [**Alt-A mHC-scratch**](2026-04-04_alt-a-mhc-scratch.md) | 2026-04-04 | **1.4777** | **-1.65%** | 🏆 最佳 |
| [Sandwich MLP](2026-04-04_alt-a-mhc-sandwich.md) | 2026-04-04 | 1.4833 | -1.28% | ✅ 参数省23% |

## 系列结论

- Alternating Attention 几乎不掉点，层级分工有效
- **mHC 从头学习 > 迁移参数**：BPB 1.4777 vs 1.4883
- 加大 dim 有效但效率不高

## 待做

- [ ] Sandwich MLP 对比 (Uniform vs Sandwich)
- [ ] LoRA + TTT 集成
- [ ] QAT 量化验证 16MB 约束

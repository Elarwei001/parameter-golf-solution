# Alternating Attention 实验系列

> 偶数层 Global + 奇数层 Local (window=128) + 最后一层强制 Global

## 实验列表

| 实验 | 日期 | BPB | vs Baseline | 状态 |
|------|------|-----|-------------|------|
| [Alt-A (Vanilla)](2026-04-03_alt-a-vanilla.md) | 2026-04-03 | 1.5001 | -0.16% | ✅ |
| [Alt-A + mHC](2026-04-03_alt-a-mhc.md) | 2026-04-03 | **1.4883** | **-0.95%** | 🏆 最佳 |
| [Alt-B (dim=448)](2026-04-04_alt-b.md) | 2026-04-04 | 1.4952 | -0.49% | ✅ |

## 系列结论

- Alternating Attention 几乎不掉点，层级分工有效
- mHC 参数可以跨架构迁移，组合效果最好
- 加大 dim 有效但效率不高

## 待做

- [ ] Alt-C: FlashAttention + seq_len=1024
- [ ] LoRA + TTT 集成
- [ ] 训练 Alternating 专用 mHC

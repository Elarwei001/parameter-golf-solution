# 待验证实验

> 每次尝试前先问：**当前状态下，这个改进还适用吗？**

**当前最佳 (FP32)**: BPB 1.4777 (Alt-A mHC-scratch, records/2026-04-04_alt-a-mhc-scratch/)
**当前最佳 (量化)**: BPB 1.5321 / 16.52MB (Sandwich QAT Adaptive, records/2026-04-04_sandwich-qat-adaptive/)

---

## 优先级高

- [ ] **30L Sandwich QAT Adaptive** — dim=384, 30层 Sandwich + Adaptive QAT（正在跑，见 records/2026-04-04_sandwich-qat-30l/）
- [ ] **FP16 Embedding + 更大 dim** — dim=478, 20L, 45M params → ~15.98MB，利用 Embedding FP16 省出的 6.29MB
  - Sandwich (384): 16.52MB → 10.20MB ✅
  - Sandwich (512): → 15.33MB ✅
  - **dim=478, 20L, 45M params → 15.98MB ✅** (推荐方案)
- [ ] **更长 warmup 再切 QAT** — step 500 太早（loss=5.2 还很高），试 warmup_floor=1000 或 2000

---

## 优先级中

- [ ] **Post-training quantization vs QAT 对比** — 先训 FP32，再 GPTQ/PTQ 量化，对比 QAT 的差距
- [ ] **Sandwich 1.5x middle** — 现在是 1.2x 中段，1.5x 可能找到更好平衡点（参数减少 vs BPB）
- [ ] **Muon Optimizer** — 专为 transformer 设计的优化器，榜首在用
- [ ] **BigramHash** — 4096 buckets, dim=128，显式 bigram 信息，榜首在用
- [ ] **seed=43 验证 32L 异常层** — 确认 Layer 27 β_attn 峰值是否稳定复现
- [ ] **48L/64L mHC 实验** — 验证异常层出现在 ~Layer 41-43 (48L) 或 ~55-58 (64L)

---

## 优先级低

- [ ] **FlashAttention + 长上下文** — seq_len=256→1024，需要真正稀疏计算
- [ ] **Mixed-precision 训练** — FP16 embedding + ternary QAT weights（混合精度）
- [ ] **VRL (Value Residual Learning)** — layer 0 的 V 通过 sigmoid gate 共享 (arxiv:2410.17897)
- [ ] **SmearGate** — 学习与前一个 token 的混合，增强局部上下文
- [ ] **U-Net Skip Connections** — 对称层跳跃连接，对 16+ 层更重要
- [ ] **Partial RoPE** — 只对部分维度应用 RoPE
- [ ] **更激进 Sandwich** — 只保留浅层大 MLP，深层全用 1.2x

---

## 已完成 ✅

| 技术 | 结果 | 日期 |
|------|------|------|
| BPE-8192 Tokenizer | -35% BPB (2.17→1.40) | 2026-03-31 |
| QAT 量化 (1.58-bit) | 几乎无损 | 2026-03-31 |
| XSA | -2.6% | 2026-04-01 |
| LoRA TTT | -27.6% | 2026-04-01 |
| XSA + TTT 组合 | **0.986 BPB** 🏆 | 2026-04-01 |
| mHC v2 (11L) | +0.13% | 2026-04-03 |
| mHC v2 (20L/32L) | 1.4978/1.4651 | 2026-04-03 |
| Alternating Attention (Alt-A) | -0.16% | 2026-04-03 |
| Alt-A + mHC transfer | -0.95% | 2026-04-03 |
| Alt-A mHC-scratch | -1.65% ✅ | 2026-04-04 |
| Alt-B (dim=448) | -0.49%, 效率低 | 2026-04-04 |
| Sandwich MLP (3x/1.2x/3x) | -1.28%, 省 23% 参数 | 2026-04-04 |
| Front-loaded MLP (3x/1.2x/1.2x) | -0.71%, over-compressed | 2026-04-04 |
| Sandwich + QAT (from step 0) | +2.16% vs baseline | 2026-04-04 |
| Sandwich + QAT Adaptive | +1.97% vs baseline | 2026-04-04 |
| FP16 full model | +0.20%, 模型减半 | 2026-04-04 |
| Embedding FP16 | +0.16%, -6.29MB ⭐ | 2026-04-04 |

## 已放弃 ❌

| 技术 | 原因 | 日期 |
|------|------|------|
| dim=128 + whitening | +15.2% 效果差 | 2026-04-01 |
| Data quality filtering | 多样性 > 质量 | 2026-04-03 |
| EMA (decay=0.999) | 5000 步内更差 | 2026-04-03 |
| PLE (Per-Layer Embedding) | +1.93% 负面结果 | 2026-04-03 |

---

*最后更新: 2026-04-04*

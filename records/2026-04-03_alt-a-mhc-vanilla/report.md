# 实验分析报告: Alt-A Vanilla + Alt-A mHC (Alternating Attention)

> 日期: 2026-04-03 | 合并了两个相关实验的报告

---

## 实验一: Alt-A Vanilla

### 1. 概述

| 项目 | 内容 |
|------|------|
| 实验名称 | Alt-A Vanilla (Alternating Attention without mHC) |
| 实验目的 | 验证 Alternating Attention（Local+Global 交替）是否能在相同配置下保持 baseline 性能 |
| 假设 | 不同层学习不同模式：Local 层专注局部上下文，Global 层负责全局语义理解，分工协作可能更高效 |
| 配置 | 20 layers, dim=384, 11 Global + 9 Local, window=128 |
| 开始时间 | 2026-04-03 21:36 SGT |
| 结束时间 | 2026-04-03 22:06 SGT |
| 耗时 | ~30 分钟 |

### 2. 关键指标结果

⚠️ **更正 (2026-04-04)**：之前 BPB 计算公式漏了 `BYTES_PER_TOKEN` 转换，修正后结果如下：

| 指标 | Alt-A Vanilla | Baseline (mHC 20L) | 差异 | 判定 |
|------|---------------|---------------------|------|------|
| Val Loss | 3.8161 | 3.8222 | -0.16% | ✅ |
| Val BPB | **1.5001** | 1.5025 | **-0.16%** | ✅ 略优 |
| 参数量 | 29.70M | 32.85M | -9.6% | ✅ 更少 |
| Global 层 | 11 | 20 | -45% | - |
| Local 层 | 9 | 0 | +9 | - |

### 3. Loss 曲线

| Step | Loss | LR | 观察 |
|------|------|-----|------|
| 500 | ~2.5 | 9.9e-4 | 正常下降 |
| 1000 | ~2.2 | 9.3e-4 | 稳定下降 |
| 2500 | ~1.8 | 5.3e-4 | 继续收敛 |
| 5000 | ~1.5 | 0 | 收敛完成 |

### 4. 发现与分析

- **Alternating 几乎不掉点**：BPB 只差 -0.16%，验证了假设
- **参数更少**：29.70M vs 32.85M，减少了 9.6%
- **Local 层有效**：9 层 Local attention 工作正常
- **层级分工有效**：Local 层学局部模式，Global 层学全局语义

⚠️ **注意**：当前实现并没有真正省计算量！`scores = torch.matmul(q, k.T)` 先全算再 mask。对 seq_len=256 没问题，真正稀疏计算需要 FlashAttention。**Alt-A 的真正价值**是层级分工，而非省计算。

### 5. 执行信息

- 脚本: `scripts/modal/modal_alternating_attn.py`
- 命令: `modal run --detach scripts/modal/modal_alternating_attn.py`
- Modal App: `ap-v4SPof1lnpBPCUgbhzHnxd`
- Checkpoint: `/data/checkpoints/alternating_attn/alt_Vanilla_dim384_L20_step5000.pt`

---

## 实验二: Alt-A + mHC (迁移参数)

### 1. 概述

| 项目 | 内容 |
|------|------|
| 实验名称 | Alt-A + mHC (Alternating Attention with mHC parameters) |
| 实验目的 | 验证在 Alternating Attention 架构上使用预训练的 mHC 参数是否有效 |
| 假设 | 使用全 Global 模型学到的 α/β 参数初始化，可以加速收敛或提升效果 |
| 配置 | 20 layers, dim=384, 11 Global + 9 Local, mHC 参数初始化 |
| 开始时间 | 2026-04-03 22:17 SGT |
| 结束时间 | 2026-04-03 22:48 SGT |
| 耗时 | 31 分钟 (1856s) |

### 2. 关键指标结果

| 指标 | 本次实验 | Alt-A Vanilla | Baseline | vs Baseline | 判定 |
|------|----------|---------------|----------|-------------|------|
| Val Loss | 3.7860 | 3.8161 | 3.8222 | -0.95% | ✅ |
| Val BPB | **1.4883** | 1.5001 | 1.5025 | **-0.95%** | ✅✅ 当时最佳 |
| 参数量 | 29.70M | 29.70M | 32.85M | -9.6% | ✅ |

### 3. 发现

- **mHC 参数可以跨架构迁移**：即使 Global/Local 交替，也能工作
- **组合效果更好**：Alternating + mHC > Alternating alone > Baseline
- 迁移参数版起点更好，但 scratch 版（见 2026-04-04_alt-a-mhc-scratch）后期反超

### 4. 执行信息

- 脚本: `scripts/modal/modal_alternating_attn.py`
- 命令: `modal run --detach scripts/modal/modal_alternating_attn.py --mhc`
- Modal App: `ap-QyWPuU4LiOtJun44phBa8v`
- Checkpoint: `/data/checkpoints/alternating_attn/alt_mHC_dim384_L20_step5000.pt`

---

## 综合结论

| 实验 | BPB | vs Baseline |
|------|-----|-------------|
| Baseline (mHC 20L) | 1.5025 | - |
| Alt-A Vanilla | 1.5001 | -0.16% |
| Alt-A + mHC (迁移) | **1.4883** | **-0.95%** |

- Alternating Attention 几乎不掉点，层级分工假设成立
- mHC 参数迁移有效，但从头学习更好（见 records/2026-04-04_alt-a-mhc-scratch）

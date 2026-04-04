# 实验分析报告: Alt-A mHC-scratch (Uniform MLP)

## 1. 实验概述

| 项目 | 内容 |
|------|------|
| 实验名称 | Alt-A mHC-scratch |
| 实验目的 | 在 Alternating Attention 架构下从头学习 mHC 参数（初始化为 1.0），对比迁移参数的效果 |
| 假设 | Alternating 架构下从头学习 mHC 可能比迁移全 Global 模型的参数更优 |
| 配置 | 20 layers, dim=384, 11 Global + 9 Local, window=128, mHC 从 1.0 学习 |
| 开始时间 | 2026-04-04 10:22 SGT |
| 结束时间 | 2026-04-04 10:54 SGT |
| 耗时 | 32 分钟 (1857s) |

## 2. 关键指标结果

| 指标 | mHC-scratch | mHC (迁移) | Baseline | vs Baseline | 判定 |
|------|-------------|-----------|----------|-------------|------|
| Val Loss | 3.7590 | 3.7860 | 3.8222 | -1.65% | ✅✅ |
| Val BPB | **1.4777** | 1.4883 | 1.5025 | **-1.65%** | ✅✅ 最佳 |
| 参数量 | 29.70M | 29.70M | 32.85M | -9.6% | ✅ |
| 训练时间 | 1857s | 1856s | ~1800s | 持平 | ➖ |

## 3. 过程指标分析

### 3.1 Loss 曲线

| Step | Loss | LR | 观察 |
|------|------|-----|------|
| 500 | 5.0112 | 9.91e-4 | 正常下降 |
| 1000 | 4.4569 | 9.40e-4 | 稳定下降 |
| 2500 | 3.9482 | 5.79e-4 | 继续收敛 |
| 5000 | 3.6834 | 1.00e-4 | 收敛完成 |

### 3.2 与迁移参数版对比

| Step | mHC-scratch Loss | mHC (迁移) Loss | 差异 |
|------|-----------------|-----------------|------|
| 500 | 5.0112 | 4.8760 | scratch 起点更高 |
| 2500 | 3.9482 | 3.9599 | scratch 反超 |
| 5000 | 3.6834 | 3.7052 | scratch 更优 |

**分析**：迁移参数版起点更好（预训练值），但 scratch 版后期反超，说明从头学习能找到更适合 Alternating 架构的参数。

## 4. 发现与分析

### 4.1 符合预期
- **从头学习 mHC 有效**：BPB 1.4777，所有实验中最佳
- **训练过程正常**：Loss 持续下降，无异常波动

### 4.2 不符合预期
- **从头学习比迁移更好**：之前假设迁移参数可能更好，实际相反
- **提升幅度超出预期**：-1.65% vs -0.95%（迁移版）

### 4.3 新发现

- **mHC 参数有架构特异性**：Alternating 架构下学到的参数比全 Global 模型迁移的更优
- **从 1.0 初始化的优势**：不受预训练偏差影响，能充分探索 Alternating 的信息流特点
- **迁移版是"预训练"而非"最优"**：迁移参数相当于 warm start，但不一定是最优解

### 4.4 对后续实验的启示

- Sandwich MLP 实验应使用 **scratch 学到的 mHC 参数** 而非迁移参数
- mHC 参数已保存在 checkpoint，可作为 Sandwich 实验的初始化

## 5. 结论

- **实验结果**：✅✅ 成功，所有实验中最佳
- **是否采用**：是，作为 Sandwich MLP 对比的 baseline
- **关键教训**：
  - mHC 参数从头学习 > 迁移参数
  - 架构特异性比预训练初始化更重要

## 6. 后续方向

| 方向 | 描述 | 优先级 |
|------|------|--------|
| **Sandwich MLP (Uniform vs Sandwich)** | 用本次学到的 mHC 参数，对比 Uniform MLP 和 Sandwich MLP | 🔴 高 |
| **QAT 量化** | 验证 16MB 限制下的最优配置 | 高 |
| **更深层模型** | 用 Sandwich MLP + QAT 尝试 30 层 | 中 |

## 7. 执行信息

### 7.1 脚本
- 文件：`scripts/modal/modal_alternating_attn.py`
- 链接：https://github.com/Elarwei001/parameter-golf-solution/blob/master/scripts/modal/modal_alternating_attn.py

### 7.2 执行命令
```bash
modal run --detach scripts/modal/modal_alternating_attn.py --mhc-scratch
```

### 7.3 关键参数
| 参数 | 值 | 说明 |
|------|------|------|
| --mhc-scratch | true | mHC 从 1.0 学习 |
| --dim | 384 (默认) | 模型维度 |
| --n_layers | 20 (默认) | 层数 |
| --local_window | 128 (默认) | Local attention 窗口大小 |
| --steps | 5000 (默认) | 训练步数 |

### 7.4 其他文件
- Modal App: `ap-U7BZVKGV1auNuAGpBe2bEy`
- 日志: https://modal.com/apps/elarweis/main/ap-U7BZVKGV1auNuAGpBe2bEy
- Checkpoint: `/data/checkpoints/alternating_attn/alt_mHC_dim384_L20_step5000.pt`

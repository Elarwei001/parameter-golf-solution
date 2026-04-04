# 实验分析报告: Alt-A (Vanilla)

## 1. 实验概述

| 项目 | 内容 |
|------|------|
| 实验名称 | Alt-A Vanilla (Alternating Attention without mHC) |
| 实验目的 | 验证 Alternating Attention（Local+Global 交替）是否能在相同配置下保持 baseline 性能 |
| 假设 | Local attention 计算量更小，通过 Local+Global 交替，可以在相同参数量下达到接近全 Global 的效果 |
| 配置 | 20 layers, dim=384, 11 Global + 9 Local, window=128 |
| 开始时间 | 2026-04-03 21:36 SGT |
| 结束时间 | 2026-04-03 22:06 SGT |
| 耗时 | ~30 分钟 |

## 2. 关键指标结果

| 指标 | Alt-A Vanilla | Baseline (mHC 20L) | 差异 | 判定 |
|------|---------------|---------------------|------|------|
| Val BPB | 1.5112 | 1.5025 | +0.58% | ➖ 持平 |
| Val Loss | 1.0475 | 1.0411 | +0.61% | ➖ 持平 |
| 参数量 | 29.70M | 32.85M | -9.6% | ✅ 更少 |
| Global 层 | 11 | 20 | -45% | 计算量减少 |
| Local 层 | 9 | 0 | +9 | 新增 |

## 3. 过程指标分析

### 3.1 Loss 曲线

| Step | Loss | LR | 观察 |
|------|------|-----|------|
| 500 | ~2.5 | 9.9e-4 | 正常下降 |
| 1000 | ~2.2 | 9.3e-4 | 稳定下降 |
| 2500 | ~1.8 | 5.3e-4 | 继续收敛 |
| 5000 | ~1.5 | 0 | 收敛完成 |

**分析**：
- Loss 曲线正常，没有异常波动
- 收敛速度与 baseline 相近
- 最终 Loss 略高于 baseline，但差距很小

### 3.2 学习率
- Cosine schedule，200 步 warmup
- 从 1e-3 衰减到 0（已修复为最低 0.1）

### 3.3 层配置

```
Layer  0: Global
Layer  1: Local(w=128)
Layer  2: Global
Layer  3: Local(w=128)
...
Layer 18: Global
Layer 19: Global (最后一层强制 Global)
```

## 4. 发现与分析

### 4.1 符合预期
- **Alternating 几乎不掉点**：BPB 只差 0.58%，验证了假设
- **参数更少**：29.70M vs 32.85M，减少了 9.6%
- **Local 层有效**：9 层 Local attention 工作正常

### 4.2 不符合预期
- 无明显不符合预期的情况

### 4.3 新发现
- **Local+Global 交替是可行的**：不需要每层都做全局 attention
- **window=128 足够**：对于 seq_len=256，128 的窗口已经覆盖一半上下文

### 4.4 潜在优化
- 既然 Local 层省计算，可以尝试**加大 dim**（Alt-B 实验）
- 或者**加深层数**，用省下的计算量换更多层

## 5. 结论

- **实验结果**：✅ 成功
- **是否采用**：是，作为后续实验的基础架构
- **关键教训**：
  - Alternating Attention 是有效的技术
  - 可以用更少的全局 attention 达到相近效果
  - 为 Alt-B（加大 dim）实验提供了信心

## 6. 后续方向

| 方向 | 描述 | 优先级 |
|------|------|--------|
| **Alt-B (dim=448)** | 用省下的计算量换更大 dim | 高（进行中） |
| **Alt-C (更深层)** | 保持 dim=384，增加层数到 24-28 | 中 |
| **训练专用 mHC** | 用 Alternating 架构训练 mHC 参数 | 中 |
| **调整 Local/Global 比例** | 尝试更多 Local 层（如 2:1 或 3:1） | 低 |

## 7. 执行信息

### 7.1 脚本
- 文件：`scripts/modal/modal_alternating_attn.py`
- 链接：https://github.com/Elarwei001/parameter-golf-solution/blob/master/scripts/modal/modal_alternating_attn.py

### 7.2 执行命令
```bash
modal run --detach scripts/modal/modal_alternating_attn.py
```

### 7.3 关键参数
| 参数 | 值 | 说明 |
|------|------|------|
| --mhc | false (默认) | 不使用 mHC 参数 |
| --dim | 384 (默认) | 模型维度 |
| --n_layers | 20 (默认) | 层数 |
| --local_window | 128 (默认) | Local attention 窗口大小 |
| --steps | 5000 (默认) | 训练步数 |

### 7.4 其他文件
- Modal App: `ap-v4SPof1lnpBPCUgbhzHnxd`
- 日志: https://modal.com/apps/elarweis/main/ap-v4SPof1lnpBPCUgbhzHnxd
- Checkpoint: `/data/checkpoints/alternating_attn/alt_Vanilla_dim384_L20_step5000.pt`

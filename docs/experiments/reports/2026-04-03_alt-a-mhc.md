# 实验分析报告: Alt-A + mHC

## 1. 实验概述

| 项目 | 内容 |
|------|------|
| 实验名称 | Alt-A + mHC (Alternating Attention with mHC parameters) |
| 实验目的 | 验证在 Alternating Attention 架构上使用预训练的 mHC 参数是否有效 |
| 假设 | 使用全 Global 模型学到的 α/β 参数初始化，可以加速收敛或提升效果 |
| 配置 | 20 layers, dim=384, 11 Global + 9 Local, mHC 参数初始化 |
| 开始时间 | 2026-04-03 22:17 SGT |
| 结束时间 | 2026-04-03 22:48 SGT |
| 耗时 | 31 分钟 (1856s) |

## 2. 关键指标结果

| 指标 | 本次实验 | Alt-A Vanilla | Baseline | vs Baseline | 判定 |
|------|----------|---------------|----------|-------------|------|
| Val BPB | 5.4621 | 1.5112 | 1.5025 | +264% | ❌❌ |
| Val Loss | 3.7860 | 1.0475 | 1.0411 | +264% | ❌❌ |
| 参数量 | 29.70M | 29.70M | 32.85M | -9.6% | ✅ |
| 训练时间 | 1856s | ~1800s | ~1800s | 持平 | ➖ |

## 3. 过程指标分析

### 3.1 Loss 曲线

| Step | Loss | 观察 |
|------|------|------|
| 500 | 4.8760 | 起点就很高 |
| 1000 | 4.4723 | 下降缓慢 |
| 2000 | 4.1101 | 继续下降 |
| 3000 | 3.9334 | 收敛变慢 |
| 5000 | 3.7052 | 最终 loss 仍然很高 |

**分析**：
- Loss 起点就远高于正常（Alt-A Vanilla 的 500 步 Loss 约 2.5）
- 虽然在下降，但最终 Loss 仍是正常值的 3.5 倍
- 说明 mHC 参数导致模型从一个很差的起点开始

### 3.2 学习率
- Cosine schedule，从 1e-3 衰减到 0
- Schedule 本身正常

### 3.3 mHC 参数
- 使用的是全 Global 模型的 α/β 值
- 这些值对 Local attention 层不适用

## 4. 发现与分析

### 4.1 符合预期
- 无

### 4.2 不符合预期
- mHC 参数完全不兼容 Alternating Attention 架构
- 预期最多小幅下降，实际 BPB 差了 3.6 倍

### 4.3 新发现
- **mHC 参数具有架构特异性**：不同架构学到的 α/β 不能互相复用
- **Local attention 和 Global attention 的最优残差权重不同**

### 4.4 可能原因

1. **参数不匹配**
   - 全 Global 模型的 α/β 是针对 Global attention 的信息流设计的
   - Local attention 只看 128 token 窗口，信息量和模式完全不同
   - 比如 Global 深层 β_attn > 1.0，但 Local 层可能需要更小的权重

2. **信息流断裂**
   - Alternating 架构中，信息在 Global ↔ Local 之间交替
   - 使用错误的 α/β 可能导致信息无法正确传递

3. **初始化灾难**
   - 错误的 α/β 让模型从一个很差的初始化开始
   - 5000 步不足以恢复到正常状态

## 5. 结论

- **实验结果**：❌ 失败
- **是否采用**：否
- **关键教训**：
  - mHC 参数是架构特定的，不能跨架构复用
  - 在新架构上使用 mHC 必须重新训练参数
  - 参数迁移需要考虑架构兼容性

## 6. 后续方向

| 方向 | 描述 | 优先级 |
|------|------|--------|
| **训练专用 mHC** | 用 Alternating 架构从头训练 mHC，让它学习 Local/Global 各自的最优 α/β | 高 |
| **分析 Local vs Global α/β** | 对比两种层类型的参数分布差异 | 中 |
| **渐进式迁移** | 只用 Global 层的 mHC 参数，Local 层初始化为 1.0 | 低 |

## 7. 相关文件

- 脚本：`scripts/modal/modal_alternating_attn.py`
- 日志：Modal App `ap-QyWPuU4LiOtJun44phBa8v`
- Checkpoint：`/data/checkpoints/alternating_attn/alt_mHC_dim384_L20_step5000.pt`

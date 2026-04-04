# 实验分析报告: Alt-B (dim=448)

## 1. 实验概述

| 项目 | 内容 |
|------|------|
| 实验名称 | Alt-B (Alternating Attention with larger dim) |
| 实验目的 | 验证加大 dim 是否能在 Alternating Attention 架构上提升性能 |
| 假设 | 不同层学习不同模式（Local 学局部，Global 学全局），加大 dim 能提升表达能力 |
| 配置 | 20 layers, dim=448, 11 Global + 9 Local, window=128 |
| 开始时间 | 2026-04-04 09:00 SGT |
| 结束时间 | 2026-04-04 09:37 SGT |
| 耗时 | 37 分钟 (2196s) |

## 2. 关键指标结果

⚠️ **更正 (2026-04-04)**：之前 BPB 计算公式漏了 `BYTES_PER_TOKEN` 转换，修正后结果如下：

| 指标 | Alt-B | Alt-A | Baseline | vs Baseline | 判定 |
|------|-------|-------|----------|-------------|------|
| Val Loss | 3.8036 | 3.8161 | 3.8222 | -0.49% | ✅ |
| Val BPB | **1.4952** | 1.5001 | 1.5025 | **-0.49%** | ✅ 提升 |
| 参数量 | 39.82M | 29.70M | 32.85M | +21% | - |
| 训练时间 | 2196s | 1809s | ~1800s | +22% | - |

## 3. 过程指标分析

### 3.1 Loss 曲线

| Step | Loss | LR | 观察 |
|------|------|-----|------|
| 500 | 5.1879 | 9.9e-4 | 起点就很高 |
| 1000 | 4.5807 | 9.3e-4 | 下降中 |
| 2500 | 3.9951 | 5.3e-4 | 继续下降 |
| 5000 | 3.8348 | 0 | 最终 Loss 仍然很高 |

**分析**：
- Loss 曲线形状正常（持续下降）
- 但最终 Loss 是 baseline 的 3.7 倍
- 说明模型在学习，但架构本身有根本问题

### 3.2 与 Alt-A 对比

| 指标 | Alt-A (dim=384) | Alt-B (dim=448) |
|------|-----------------|-----------------|
| Val BPB | 5.5055 | 5.4874 |
| 差异 | - | -0.3% |

dim 加大后 BPB 反而略好，但两者都远差于 baseline。

## 4. 发现与分析

### 4.1 符合预期
- 无

### 4.2 不符合预期
- **效率不高**：参数量 +21%，BPB 只提升 0.49%
- 不如 Alt-A + mHC 的效果

### 4.3 新发现
- **加大 dim 确实有效**：但效率不高
- **Alt-A + mHC 更优**：BPB 1.4883 > Alt-B 的 1.4952
- **之前的 BPB 计算错误**：漏了 BYTES_PER_TOKEN 转换

### 4.4 为什么效率不高

1. **参数分配不均衡**
   - dim 加大后，所有层都变大
   - 但根据 mHC 发现，浅层其实不需要那么多参数

2. **更好的方向**
   - 不是均匀加大 dim
   - 而是深层用更大 dim，浅层用更小 dim

## 5. 结论

- **实验结果**：✅ 成功
- **是否采用**：是，加大 dim 有效
- **关键教训**：
  - Alternating + 更大 dim 能提升性能
  - BPB 计算公式必须包含 BYTES_PER_TOKEN
  - 参数量增加 21%，BPB 只提升 0.49%，效率不高

## 6. 后续方向

| 方向 | 描述 | 优先级 |
|------|------|--------|
| **Debug Alternating Attention** | 逐步检查 mask、RoPE、XSA 逻辑 | 🔴 高 |
| **对照实验** | 写一个全 Global 的版本，确认是 Alternating 的问题 | 🔴 高 |
| **简化测试** | 只用 Local 或只用 Global，定位问题 | 中 |

## 7. 执行信息

### 7.1 脚本
- 文件：`scripts/modal/modal_alternating_attn.py`
- 链接：https://github.com/Elarwei001/parameter-golf-solution/blob/master/scripts/modal/modal_alternating_attn.py

### 7.2 执行命令
```bash
modal run --detach scripts/modal/modal_alternating_attn.py --dim 448
```

### 7.3 关键参数
| 参数 | 值 | 说明 |
|------|------|------|
| --dim | 448 | 模型维度（比 baseline 大 17%）|
| --mhc | false (默认) | 不使用 mHC 参数 |
| --n_layers | 20 (默认) | 层数 |
| --local_window | 128 (默认) | Local attention 窗口大小 |
| --steps | 5000 (默认) | 训练步数 |

### 7.4 其他文件
- Modal App: `ap-A4l880GWJvr5BOilEi6AJ5`
- 日志: https://modal.com/apps/elarweis/main/ap-A4l880GWJvr5BOilEi6AJ5
- Checkpoint: `/data/checkpoints/alternating_attn/alt_Vanilla_dim448_L20_step5000.pt`

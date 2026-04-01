# Parameter Golf 实验记录

> 记录我们的探索之旅：从 2.17 BPB 到 1.40 BPB

## 📊 进度总览

| 日期 | 实验 | BPB | 改进 | 关键发现 |
|------|------|-----|------|---------|
| Day 1 | Baseline (字节级) | 2.28 | - | 起点 |
| Day 1 | LeakyReLU² | 2.18 | -4.3% | 激活函数很重要 |
| Day 1 | Sliding Window | 2.18 | -5.5% | 中距离依赖 |
| Day 2 | BPE-1024 | 1.68 | -23% | **Tokenizer 是关键！** |
| Day 2 | BPE-8192 | 1.40 | -35% | 更大词表更好 |
| Day 2 | QAT (1.58-bit) | 1.40 | 0% | **量化几乎无损！** |
| Day 3 | XSA (Exclusive Self Attention) | 1.44 | -2.6% | **去除 self-similarity bias** |

**总进步：2.28 → 1.44 = -37%** 🎉

---

## 🔬 实验详情

### 实验 1: 激活函数对比

**假设**：不同激活函数影响模型表达能力

| 激活函数 | BPB | 说明 |
|---------|-----|------|
| GELU | 2.28 | baseline |
| LeakyReLU² | 2.18 | **最佳** |
| ReLU² | 2.20 | 接近 |
| Swish | 2.25 | 一般 |

**结论**：LeakyReLU² 稳定优于其他激活函数

---

### 实验 2: 滑动窗口注意力

**假设**：限制注意力范围可以让模型更好地利用有限参数

| Window Size | BPB | 说明 |
|-------------|-----|------|
| Full (无限制) | 2.28 | baseline |
| 256 | 2.19 | 有效 |
| 192 | 2.18 | **最佳** |
| 128 | 2.19 | 略差 |
| 64 | 2.22 | 太小 |

**结论**：192 是 TinyShakespeare 的甜蜜点

---

### 实验 3: Tokenizer 对比 ⭐

**假设**：更好的 tokenizer 可以大幅提升性能

| Tokenizer | Vocab | BPB | 改进 |
|-----------|-------|-----|------|
| 字节级 | 256 | 2.17 | baseline |
| BPE | 1024 | 1.68 | **-23%** |
| BPE | 8192 | 1.40 | **-35%** |

**关键发现**：
1. Tokenizer 选择对 BPB 影响**远超**模型架构调整
2. 更大词表 = 更短序列 = 更少计算 = 同样时间学更多
3. 8192 是 16MB 限制下的甜蜜点（更大词表 embedding 太占空间）

---

### 实验 4: QAT 量化 ⭐

**假设**：1.58-bit 量化会损失性能

| 方案 | BPB | 模型大小 |
|------|-----|---------|
| FP32 | 1.402 | ~120 MB |
| QAT (1.58-bit) | 1.403 | 13.5 MB |

**惊人发现**：
1. **几乎无损！** 差距只有 0.07%
2. 模型小了 **9 倍**
3. STE (Straight-Through Estimator) 真的有效
4. Warmup 策略：先 FP16 训练 500 步，再切 QAT

---

## 💡 核心洞察

### 1. Tokenizer 比架构更重要
```
投入 10 小时调模型架构 → 5% 改进
换个好的 tokenizer → 35% 改进
```

### 2. 量化不一定损失性能
```
传统认知：量化 = 性能下降
实际发现：QAT 训练的模型几乎无损
```

### 3. BPB 是公平的比较指标
```
BPB = (Cross Entropy / ln(2)) × (tokens / bytes)

不同 tokenizer 都归一化到"每字节多少 bit"
```

### 4. 16MB 限制决定了甜蜜点
```
Vocab 8192 × dim 512 × 1.58 bits = 1.5 MB ✓
Vocab 32K × dim 512 × 1.58 bits = 6 MB ⚠️ 
Vocab 100K × dim 512 × 1.58 bits = 19 MB ❌
```

---

## 🎯 与榜首的差距

| 指标 | 我们 | 榜首 | 差距 |
|------|------|------|------|
| BPB | 1.40 | 1.11 | -20% |
| 技术 | 基础 QAT | GPTQ + XSA + TTT | 复杂 |

**榜首使用的高级技术**（我们还没学）：
- **XSA** (Cross-Sample Attention)：跨样本注意力
- **TTT** (Test-Time Training)：推理时微调
- **GPTQ**：更先进的量化算法
- **Muon Optimizer**：专门优化器
- **BigramHash**：输入增强

---

### 实验 5: 12 层模型

**假设**：用满 16MB 限制可以提升性能

| 配置 | 层数 | 大小 | BPB |
|------|------|------|-----|
| 原版 | 9 | 13.5 MB | 1.402 |
| 加深 | 12 | 15.2 MB | **1.396** |

**结论**：多 3 层带来小幅提升（-0.4%），但还有 0.8 MB 空间可以利用

---

### 实验 6: Embedding 空间效率分析 ⭐⭐

**背景**：Elar 的洞察 — 训练后的向量空间可能仍然是稀疏的，存在大量冗余

**测量指标**：
- **Participation Ratio (PR)**：有效维度数，$PR = (\sum \sigma_i^2)^2 / \sum \sigma_i^4$
- **Anisotropy**：向量分布各向异性，$\mathbb{E}[\cos(x,y)]$

**实验结果**（dim=512, 9 层, 2000 步）：

| 指标 | 训练前（随机） | 训练后 | 变化 |
|------|---------------|--------|------|
| **Participation Ratio** | 482 / 512 | **104 / 512** | **-78%** |
| **效率** | 94.1% | **20.4%** | 📉 |
| **95% 方差需要** | 471 维 | 427 维 | -44 |
| **Anisotropy** | 0.0001 | 0.0422 | +42x |

**🔥 关键发现**：
1. 随机初始化时，embedding 空间几乎完美利用（94%）
2. **训练后，有效维度从 482 暴跌到 104**
3. 这意味着 **~400 个维度（80%）是"死"维度**
4. 模型实际只在用 ~100 维的子空间

**潜在优化方向**：
- 使用 dim=128-150 + whitening 变换
- 可能获得相同表达能力，但节省 ~75% embedding 参数
- 省下的空间可用于增加层数

**相关文件**：
- `docs/EMBEDDING_SPACE_EFFICIENCY.md` - 完整研究方向文档
- `analyze_embedding_space.py` - 测量工具脚本
- `modal_analyze_embedding.py` - Modal 云端分析脚本

---

### 实验 7: XSA (Exclusive Self Attention) ⭐

**背景**：来自论文 "Exclusive Self Attention" (2026)

**核心思想**：标准 attention 的输出往往和自己的 value 向量很相似（similarity bias）。XSA 把这部分投影去掉，强迫 attention 只关注上下文信息。

**实现**：只需 2 行代码！
```python
v_norm = F.normalize(v_self, dim=-1)
z = y - (y * v_norm).sum(dim=-1, keepdim=True) * v_norm
```

**实验结果**（dim=512, 9 层, 3000 步）：

| 配置 | BPB | 改进 |
|------|-----|------|
| Standard Attention | ~1.48 | baseline |
| **XSA** | **1.441** | **-2.6%** |

**结论**：XSA 有效！简单的 2 行代码改动带来了明显的提升。

**原理**：
- 标准 SA：输出 y 包含自己 v 的信息（冗余，因为有 residual connection）
- XSA：z = y - proj(y, v)，去除自己的投影，专注上下文

**相关文件**：
- `modal_xsa.py` - XSA 训练脚本

---

### 实验 8: Whitening / 小 dim 实验

**假设**：既然 PR 分析显示有效维度只有 ~104，能否直接用小 dim？

**实验结果**：

| 配置 | Dim | Layers | Size | BPB | vs Baseline |
|------|-----|--------|------|-----|-------------|
| Baseline | 512 | 9 | 14.8M | 1.479 | - |
| dim=128 | 128 | 14 | 2.8M | 1.709 | +15.5% ❌ |
| dim=128 + whitening | 128 | 14 | 2.9M | 1.704 | +15.2% ❌ |
| dim=256 | 256 | 12 | 6.5M | 1.576 | +6.5% |

**结论**：
1. 直接用小 dim 效果不好，即使加更多层
2. Whitening 层几乎没帮助
3. PR 测的是训练后状态，但训练过程需要大空间"探索"
4. 这是一个有价值的负面结果！

---

## 📁 代码文件

```
parameter-golf-solution/
├── train_gpt.py          # 基础训练脚本
├── modal_bpe.py          # BPE-1024 Modal 训练
├── modal_bpe8k.py        # BPE-8192 Modal 训练
├── modal_qat.py          # QAT 量化训练 ⭐
├── EXPERIMENTS.md        # 本文件
└── SYSTEMATIC_EXPERIMENTS.md  # 实验方法论
```

---

## 🚀 下一步计划

1. ✅ ~~增加模型容量~~：已测试 12 层
2. ✅ ~~Embedding 稠密化~~：dim=128 + whitening 实验（负面结果）
3. ✅ ~~XSA~~：有效！-2.6%
4. **TTT (Test-Time Training)**：榜首的秘密武器，达到 1.08 BPB！
5. **Muon Optimizer**：专门为 transformer 设计
6. **更长训练**：5000 → 10000 步
7. **组合最佳技术**：QAT + XSA + TTT

---

*最后更新: 2026-04-01*

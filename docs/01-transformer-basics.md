# Transformer 基础

> 这是一切的起点。理解 Transformer，才能理解为什么某些改进有效。

## 一句话定义

**Transformer = Attention + FFN + 残差连接**

它是一种神经网络架构，擅长处理序列数据（文本、代码、音频等）。

---

## 核心组件

### 1. Self-Attention（自注意力）

**作用**：让每个词"看到"序列中的所有其他词，决定该关注谁。

**直觉**：读到"它"这个词时，你会回头看前文找"它"指代什么。Attention 就是让模型自动学会这种"回头看"。

```
输入: "The cat sat on the mat because it was tired"
                                      ↑
                                    "it" 指谁？
                                      ↓
Attention 学会: it → cat (高权重)
```

**数学**（简化版）：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) V
$$

- **Q (Query)**：我在找什么？
- **K (Key)**：我有什么特征？
- **V (Value)**：我的内容是什么？
- **softmax**：把分数变成概率（总和为 1）
- **√d**：缩放因子，防止数值太大

### 2. FFN（前馈网络）

**作用**：对每个位置独立做非线性变换，增加表达能力。

**直觉**：Attention 负责"信息交流"，FFN 负责"信息加工"。

```python
# 最简单的 FFN
def ffn(x):
    hidden = relu(x @ W1)  # 先扩展维度
    output = hidden @ W2    # 再压缩回来
    return output
```

### 3. 残差连接 + LayerNorm

**残差连接**：`output = layer(x) + x`

**直觉**：让梯度更容易流动，训练更稳定。就像修路时保留老路，新路坏了还能走老路。

**LayerNorm**：把每个样本的特征标准化到均值 0、方差 1。

---

## Transformer Block

一个完整的 Transformer Block：

```
输入 x
    │
    ├── LayerNorm
    ├── Self-Attention
    ├── 残差连接 (+x)
    │
    ├── LayerNorm
    ├── FFN
    ├── 残差连接 (+x)
    │
    ▼
输出
```

堆叠多个 Block = 更深的网络 = 更强的表达能力

---

## 在 Parameter Golf 中

我们的 `StandardGPT` 就是标准 Transformer decoder：

```python
model_config = {
    "dim": 512,        # 特征维度
    "n_layers": 9,     # 堆叠 9 个 Block
    "n_heads": 8,      # 8 个注意力头
    "vocab_size": 50257,
}
```

**为什么用 Transformer？**
- 并行计算效率高
- 长距离依赖建模强
- 是目前语言模型的标准架构

---

## 延伸阅读

1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — 原论文
2. [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — 可视化讲解
3. [LLM Fundamentals Day 4](https://github.com/Elarwei001/llm-fundamentals/blob/master/articles/en/day04-transformer-architecture.md) — 我们的课程

---

*下一篇：[激活函数](02-activation-functions.md)*

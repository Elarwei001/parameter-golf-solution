# 激活函数

> FFN 里的"非线性魔法"。选对激活函数，BPB 能降好几个点。

## 一句话定义

**激活函数 = 给神经网络加入非线性**

没有激活函数，不管堆多少层，神经网络就是个线性变换，表达能力极弱。

---

## 常见激活函数

### 1. ReLU（最经典）

$$
\text{ReLU}(x) = \max(0, x)
$$

```
     │    ╱
     │   ╱
─────┼──╱────
     │ ╱
     │╱
```

**优点**：简单、计算快
**缺点**：负数全变 0，可能"神经元死亡"

### 2. GELU（GPT 系列用）

$$
\text{GELU}(x) = x \cdot \Phi(x)
$$

其中 Φ 是标准正态分布的 CDF。

**直觉**：比 ReLU 更平滑，负数不会完全变 0。

### 3. SwiGLU（我们的 baseline）

$$
\text{SwiGLU}(x) = \text{Swish}(xW_1) \otimes (xW_2)
$$

其中 Swish(x) = x · sigmoid(x)，⊗ 是逐元素乘法。

**特点**：
- 两个线性变换 + 门控机制
- LLaMA、PaLM 等模型使用
- 效果好但参数多一些

### 4. LeakyReLU²（我们的改进 🏆）

$$
\text{LeakyReLU}^2(x) = \text{LeakyReLU}(x, 0.5)^2
$$

```python
def leaky_relu_squared(x, negative_slope=0.5):
    return F.leaky_relu(x, negative_slope).square()
```

**为什么有效？**
1. **平方**：让输出更平滑，梯度更稳定
2. **LeakyReLU**：负数不会完全消失（slope=0.5）
3. **简单**：比 SwiGLU 计算量少

---

## 我们的实验结果

| 激活函数 | BPB | vs Baseline |
|----------|-----|-------------|
| SwiGLU (baseline) | 2.4939 | — |
| **LeakyReLU²** | **2.3875** | **-4.3%** 🏆 |

**结论**：简单的 LeakyReLU² 竟然比复杂的 SwiGLU 更好！

---

## 代码实现

```python
import torch
import torch.nn.functional as F

class LeakyReLUSquaredMLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
    
    def forward(self, x):
        # LeakyReLU with negative_slope=0.5, then square
        hidden = F.leaky_relu(self.w1(x), negative_slope=0.5)
        hidden = hidden.square()
        return self.w2(hidden)
```

---

## 为什么激活函数这么重要？

1. **非线性**：没有它，深度网络 = 浅层网络
2. **梯度流动**：选错了可能梯度消失/爆炸
3. **表达能力**：决定网络能学什么函数

---

## 选择建议

| 场景 | 推荐 |
|------|------|
| 通用场景 | GELU 或 SwiGLU |
| 追求效率 | ReLU 或 LeakyReLU |
| Parameter Golf | LeakyReLU²（我们验证过！）|

---

*上一篇：[Transformer 基础](01-transformer-basics.md) | 下一篇：[注意力变体](03-attention-variants.md)*

# Activation Functions

> The "nonlinear magic" inside FFN. Choosing the right activation can drop BPB by several points.

## One-Sentence Definition

**Activation function = Adding nonlinearity to neural networks**

Without activation functions, no matter how many layers you stack, a neural network is just a linear transformation with very limited expressiveness.

---

## Common Activation Functions

### 1. ReLU (The Classic)

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

**Pros**: Simple, fast to compute
**Cons**: Negative values become 0, may cause "dead neurons"

### 2. GELU (Used by GPT series)

$$
\text{GELU}(x) = x \cdot \Phi(x)
$$

Where Φ is the CDF of the standard normal distribution.

**Intuition**: Smoother than ReLU, negative values don't completely become 0.

### 3. SwiGLU (Our baseline)

$$
\text{SwiGLU}(x) = \text{Swish}(xW_1) \otimes (xW_2)
$$

Where Swish(x) = x · sigmoid(x), and ⊗ is element-wise multiplication.

**Characteristics**:
- Two linear transforms + gating mechanism
- Used by LLaMA, PaLM, etc.
- Good performance but more parameters

### 4. LeakyReLU² (Our improvement 🏆)

$$
\text{LeakyReLU}^2(x) = \text{LeakyReLU}(x, 0.5)^2
$$

```python
def leaky_relu_squared(x, negative_slope=0.5):
    return F.leaky_relu(x, negative_slope).square()
```

**Why does it work?**
1. **Squaring**: Smoother output, more stable gradients
2. **LeakyReLU**: Negative values don't completely vanish (slope=0.5)
3. **Simple**: Less computation than SwiGLU

---

## Our Experiment Results

| 激活函数 | BPB | vs Baseline |
|----------|-----|-------------|
| SwiGLU (baseline) | 2.4939 | — |
| **LeakyReLU²** | **2.3875** | **-4.3%** 🏆 |

**Conclusion**: Simple LeakyReLU² actually beats the more complex SwiGLU!

---

## Code Implementation

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

## Why Are Activation Functions So Important?

1. **Nonlinearity**: Without it, deep networks = shallow networks
2. **Gradient flow**: Wrong choice can cause vanishing/exploding gradients
3. **Expressiveness**: Determines what functions the network can learn

---

## Recommendations

| Scenario | Recommendation |
|----------|----------------|
| General use | GELU or SwiGLU |
| Efficiency-focused | ReLU or LeakyReLU |
| Parameter Golf | LeakyReLU² (we validated it!) |

---

*Previous: [Transformer Basics](01-transformer-basics.md) | Next: [Attention Variants](03-attention-variants.md)*

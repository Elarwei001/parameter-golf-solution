# Optimizers

> The "driver" of neural network training. Choose the right optimizer for faster convergence and better results.

## One-Line Definition

**Optimizer = Decides how to update weights based on gradients**

Gradients tell you "which direction to go"; the optimizer decides "how fast and how to get there".

---

## Gradient Descent Basics

The simplest update rule:

$$
w_{new} = w_{old} - \eta \cdot \nabla L
$$

- **w**: weights
- **η (eta)**: learning rate
- **∇L**: gradient of the loss function

**Problems**:
- Learning rate too large → oscillation
- Learning rate too small → too slow
- Using the same learning rate for all parameters is unreasonable

---

## Adam (Most Common)

**Core idea**: Maintain an individual learning rate for each parameter, adapting based on historical gradients.

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \quad \text{(momentum)}
$$
$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \quad \text{(second moment)}
$$
$$
w_t = w_{t-1} - \eta \cdot \frac{m_t}{\sqrt{v_t} + \epsilon}
$$

**Intuition**:
- **m (momentum)**: Smooths gradients to avoid oscillation
- **v (second moment)**: Estimates gradient "volatility" — parameters with high volatility get smaller learning rates
- **ε**: Prevents division by zero

---

## AdamW (Our Choice 🏆)

**Improvement**: Decouples weight decay from the Adam update formula.

```python
# Adam's weight decay (problematic)
gradient = gradient + weight_decay * weight

# AdamW's weight decay (correct)
weight = weight - learning_rate * weight_decay * weight
```

**Why it's better**:
- Cleaner regularization effect
- Decoupled from learning rate
- More stable training

---

## Muon (The New Kid)

**Paper claims**: Converges 2x faster than Adam.

**Core idea**: Replaces SGD with Newton's method for a more accurate estimate of update direction.

**Our experiments**:

| Time | AdamW | Muon |
|------|-------|------|
| 15min | 4.07 | **3.99** ✓ |
| 30min | **3.73** | 3.91 |
| 60min | **3.55** 🏆 | 3.85 |

**Conclusion**:
- Muon starts fast
- But AdamW is more stable long-term
- **Don't blindly trust papers** — always run your own experiments!

---

## Learning Rate Scheduling

Choosing an optimizer isn't enough — how the learning rate changes over time matters too.

### Warmup

```
lr
│     ╱────────
│    ╱
│   ╱
│  ╱
│ ╱
└─────────────── step
   warmup
```

**Why**: Early in training, weights are random and gradients are unstable. A large learning rate can easily cause divergence. Start small, then increase.

### Cosine Decay

```
lr
│────╮
│    ╲
│     ╲
│      ╲
│       ╰────
└─────────────── step
```

**Why**: Later in training, a smaller learning rate helps with fine-grained adjustments.

---

## Our Configuration

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0006,
    betas=(0.9, 0.95),
    weight_decay=0.1,
)

# Cosine schedule with warmup
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,
    num_training_steps=total_steps,
)
```

---

## Tuning Guide

| Parameter | Typical Range | Our Value |
|-----------|---------------|-----------|
| **lr** | 1e-4 ~ 1e-3 | 6e-4 |
| **weight_decay** | 0.01 ~ 0.1 | 0.1 |
| **warmup_steps** | 1% ~ 5% of total | 100 |
| **beta1** | 0.9 | 0.9 |
| **beta2** | 0.95 ~ 0.999 | 0.95 |

---

## Common Questions

### Q: Training not converging?

1. Learning rate too large → reduce lr
2. No warmup → add it
3. Exploding gradients → add gradient clipping

### Q: Converging too slowly?

1. Learning rate too small → increase lr
2. Batch size too small → increase it
3. Try a more aggressive optimizer (e.g., Muon)

### Q: Overfitting?

1. weight_decay too small → increase it
2. Add dropout
3. Reduce model size

---

*Previous: [Attention Variants](03-attention-variants.md) | Next: [Quantization](05-quantization.md)*

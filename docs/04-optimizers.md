# 优化器

> 训练神经网络的"司机"。选对优化器，收敛更快、效果更好。

## 一句话定义

**优化器 = 决定如何根据梯度更新权重**

梯度告诉你"往哪走"，优化器决定"走多快、怎么走"。

---

## 梯度下降基础

最简单的更新规则：

$$
w_{new} = w_{old} - \eta \cdot \nabla L
$$

- **w**：权重
- **η (eta)**：学习率
- **∇L**：损失函数的梯度

**问题**：
- 学习率太大 → 震荡
- 学习率太小 → 太慢
- 所有参数用同一个学习率不合理

---

## Adam（最常用）

**核心思想**：为每个参数维护独立的学习率，基于历史梯度自适应调整。

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \quad \text{(动量)}
$$
$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \quad \text{(二阶矩)}
$$
$$
w_t = w_{t-1} - \eta \cdot \frac{m_t}{\sqrt{v_t} + \epsilon}
$$

**直觉**：
- **m (动量)**：平滑梯度，避免震荡
- **v (二阶矩)**：估计梯度的"波动性"，波动大的参数学习率小
- **ε**：防止除以零

---

## AdamW（我们的选择 🏆）

**改进**：把 weight decay 从 Adam 更新公式里分离出来。

```python
# Adam 的 weight decay（有问题）
gradient = gradient + weight_decay * weight

# AdamW 的 weight decay（正确）
weight = weight - learning_rate * weight_decay * weight
```

**为什么更好**：
- 正则化效果更干净
- 和学习率解耦
- 训练更稳定

---

## Muon（新秀）

**论文声称**：比 Adam 快 2 倍收敛。

**核心思想**：用 Newton 法代替 SGD，更精确地估计更新方向。

**我们的实验**：

| 时间 | AdamW | Muon |
|------|-------|------|
| 15min | 4.07 | **3.99** ✓ |
| 30min | **3.73** | 3.91 |
| 60min | **3.55** 🏆 | 3.85 |

**结论**：
- Muon 开局快
- 但 AdamW 长期更稳
- **不要盲信论文**，要实测！

---

## 学习率调度

光选优化器不够，学习率怎么变化也很重要：

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

**为什么**：训练初期权重很随机，大学习率容易崩。先小后大。

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

**为什么**：后期需要更小的学习率来精细调整。

---

## 我们的配置

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0006,
    betas=(0.9, 0.95),
    weight_decay=0.1,
)

# 带 warmup 的 cosine 调度
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,
    num_training_steps=total_steps,
)
```

---

## 调参建议

| 参数 | 常见范围 | 我们的值 |
|------|----------|----------|
| **lr** | 1e-4 ~ 1e-3 | 6e-4 |
| **weight_decay** | 0.01 ~ 0.1 | 0.1 |
| **warmup_steps** | 1% ~ 5% of total | 100 |
| **beta1** | 0.9 | 0.9 |
| **beta2** | 0.95 ~ 0.999 | 0.95 |

---

## 常见问题

### Q: 训练不收敛？

1. 学习率太大 → 降低 lr
2. 没有 warmup → 加上
3. 梯度爆炸 → 加 gradient clipping

### Q: 收敛太慢？

1. 学习率太小 → 提高 lr
2. batch size 太小 → 增大
3. 换更激进的优化器（如 Muon）

### Q: 过拟合？

1. weight_decay 太小 → 增大
2. 加 dropout
3. 减少模型大小

---

*上一篇：[注意力变体](03-attention-variants.md) | 下一篇：[量化](05-quantization.md)*

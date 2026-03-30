# 非 Transformer 架构

> Transformer 不是唯一选择。Mamba 等 SSM 在长序列上有优势，但我们的实验失败了。

## 一句话定义

**SSM (State Space Model) = 用状态方程代替 Attention**

把序列建模变成一个连续动态系统的离散化。

---

## 为什么探索替代架构？

Transformer 的问题：

| 问题 | 原因 |
|------|------|
| O(n²) 复杂度 | 每个 token 看所有 token |
| KV Cache 膨胀 | 推理时缓存线性增长 |
| 长序列 OOM | 100K tokens 内存爆炸 |

**理想替代**：O(n) 复杂度 + 常数内存 + 保持性能

---

## Mamba（我们试过的）

### 核心思想

把离散 token 序列当作连续信号采样：

$$
\begin{aligned}
h'(t) &= Ah(t) + Bx(t) \quad \text{(状态更新)} \\
y(t) &= Ch(t) \quad \text{(输出)}
\end{aligned}
$$

- **h(t)**：隐藏状态
- **A, B, C**：可学习参数
- 关键：**A, B, C 随输入变化**（选择性 SSM）

### Mamba vs Transformer

| 特性 | Transformer | Mamba |
|------|-------------|-------|
| 复杂度 | O(n²) | O(n) |
| 长序列 | 慢/OOM | 高效 |
| 短序列 | 非常强 | 还行 |
| 并行训练 | 高效 | 需要技巧 |
| 推理 | KV cache 膨胀 | 常数内存 |

### 我们的实验结果

```
Baseline (Transformer): 2.49 BPB
Mamba:                  5.42 BPB ← 差很多！
```

**为什么失败？**

1. **序列太短**（128 tokens）：Mamba 优势在长序列
2. **纯 PyTorch 实现太慢**：没用官方 CUDA kernel
3. **参数量小**：Mamba 可能需要更多参数才能发挥

---

## RWKV（另一个替代）

### 核心思想

结合 RNN 和 Transformer：
- 训练时像 Transformer（并行）
- 推理时像 RNN（O(1) 内存）

```
RWKV = R(eceptance) W(eighted) K(ey) V(alue)
```

### 特点

- 完全无 Attention
- 线性复杂度
- 已有 14B 参数模型

我们没试，因为实现复杂度高。

---

## 为什么 Mamba 在 Parameter Golf 失败

### 1. 短序列劣势

```
序列长度: 128 tokens

Transformer O(n²) = 128² = 16,384 次操作
Mamba O(n) = 128 次操作（但常数大）

短序列时 Transformer 的 O(n²) 根本不是瓶颈！
```

### 2. Attention 的建模优势

短序列上，Attention 能直接看到所有 token 关系：

```
Transformer: token_50 直接 attend to token_1
Mamba: token_50 通过状态间接"记住" token_1
```

直接 > 间接，尤其是序列短时。

### 3. 工程问题

```python
# 我们的实现（慢）
def mamba_forward(x, A, B, C):
    h = torch.zeros(...)
    outputs = []
    for t in range(seq_len):
        h = A @ h + B @ x[t]
        outputs.append(C @ h)
    return torch.stack(outputs)

# 官方实现用 CUDA kernel，快 10 倍+
```

---

## 教训总结

| 教训 | 详情 |
|------|------|
| **场景匹配** | Mamba 适合长序列，我们是短序列 |
| **不要造轮子** | 用官方实现，别自己写 |
| **基准先行** | 先确认 baseline 再探索 |

---

## 什么时候用替代架构？

| 场景 | 推荐 |
|------|------|
| 短序列（<1K） | **Transformer** |
| 中等序列（1K-10K） | Transformer + Flash Attention |
| 长序列（10K-100K） | Mamba 或混合架构 |
| 超长序列（>100K） | **Mamba / RWKV** |
| 实时流式 | Mamba（常数内存）|

---

## 结论

对于 Parameter Golf（128 tokens 短序列）：

> **Transformer 仍然是最佳选择**

Mamba 等替代架构的优势在长序列场景，我们的场景用不上。

---

*上一篇：[量化](05-quantization.md) | 下一篇：[训练技巧](07-training-techniques.md)*

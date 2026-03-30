# 注意力变体

> Full Attention 太贵？试试这些变体。Sliding Window 在短序列上效果出奇地好。

## 一句话定义

**注意力变体 = 用不同方式限制"谁能看到谁"**

Full Attention 让每个 token 看所有 token，O(n²) 复杂度。变体通过限制视野来降低成本。

---

## Full Attention（标准版）

```
Token:  1  2  3  4  5
    1   ✓  ✓  ✓  ✓  ✓
    2   ✓  ✓  ✓  ✓  ✓
    3   ✓  ✓  ✓  ✓  ✓
    4   ✓  ✓  ✓  ✓  ✓
    5   ✓  ✓  ✓  ✓  ✓

每个 token 能看到所有其他 token
```

**问题**：
- 计算量 O(n²)
- 内存 O(n²)
- 序列长了就爆炸

---

## Causal Attention（GPT 用的）

```
Token:  1  2  3  4  5
    1   ✓  ✗  ✗  ✗  ✗
    2   ✓  ✓  ✗  ✗  ✗
    3   ✓  ✓  ✓  ✗  ✗
    4   ✓  ✓  ✓  ✓  ✗
    5   ✓  ✓  ✓  ✓  ✓

只能看到自己和之前的 token（不能偷看未来）
```

**为什么需要**：语言模型是自回归的，预测下一个 token 时不能知道答案。

---

## Sliding Window Attention（我们用的 🏆）

```
窗口大小 = 3

Token:  1  2  3  4  5
    1   ✓  ✓  ✓  ✗  ✗
    2   ✓  ✓  ✓  ✓  ✗
    3   ✓  ✓  ✓  ✓  ✓
    4   ✗  ✓  ✓  ✓  ✓
    5   ✗  ✗  ✓  ✓  ✓

每个 token 只看前后 k 个（局部窗口）
```

**优点**：
- 计算量 O(n × k)，k 是窗口大小
- 局部信息通常最重要
- 通过堆叠层数，间接获得全局视野

**我们的配置**：
```python
window_size = 256  # 每个 token 看前后 256 个
```

---

## 我们的实验结果

| Attention | BPB | vs Baseline |
|-----------|-----|-------------|
| Full (Causal) | 2.4939 | — |
| **Sliding Window** | **2.3568** | **-5.5%** 🏆 |

**意外发现**：在短序列（128 tokens）上，Sliding Window 竟然比 Full Attention 更好！

**可能原因**：
1. 局部注意力提供了有用的归纳偏置
2. 减少了噪声（不需要关注太远的 token）
3. 类似于 CNN 的局部感受野

---

## 实现细节：RoPE 的坑

我们遇到的 Bug：

```
RuntimeError: index out of bounds
```

**原因**：RoPE 位置编码预分配了固定大小的 cos/sin 表，序列超长就越界。

**解决**：写了 `RotaryEmbeddingDynamic`，自动扩展表大小。

```python
class RotaryEmbeddingDynamic(nn.Module):
    def __init__(self, dim, max_seq_len=1024):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len):
        # 构建 cos/sin 缓存
        ...
    
    def forward(self, x, seq_len):
        if seq_len > self.max_seq_len:
            # 自动扩展！
            self.max_seq_len = seq_len * 2
            self._build_cache(self.max_seq_len)
        return self._apply_rotary(x, seq_len)
```

---

## 其他变体（我们没试）

| 变体 | 思路 | 适用场景 |
|------|------|----------|
| **Sparse Attention** | 只关注特定模式（如隔一个看一个）| 超长序列 |
| **Linear Attention** | 用核技巧把 O(n²) 变 O(n) | 效率优先 |
| **Flash Attention** | 不改数学，优化 GPU 内存访问 | 通用加速 |
| **Multi-Query Attention** | 多个 Q 共享 K/V | 推理加速 |

---

## 选择建议

| 序列长度 | 推荐 |
|----------|------|
| 短（<512） | Sliding Window 或 Full |
| 中（512-4K） | Full + Flash Attention |
| 长（>4K） | Sliding Window 或 Sparse |
| 超长（>100K） | Mamba/SSM（非 Attention）|

---

*上一篇：[激活函数](02-activation-functions.md) | 下一篇：[优化器](04-optimizers.md)*

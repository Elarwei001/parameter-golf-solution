# 量化

> 把模型从 32-bit 压缩到 3-bit，大小缩减 10 倍。Parameter Golf 的关键技术。

## 一句话定义

**量化 = 用更少的 bit 表示权重**

FP32 → INT8 → INT4 → 3-bit，精度换大小。

---

## 为什么需要量化？

Parameter Golf 的规则：

> 模型必须 ≤ 16MB @ 3-bit 量化

假设有 50M 参数：
- FP32: 50M × 4 bytes = **200 MB** ❌
- INT8: 50M × 1 byte = **50 MB** ❌
- 3-bit: 50M × 0.375 bytes = **18.75 MB** ❌
- 3-bit + 40M params: 40M × 0.375 = **15 MB** ✅

---

## 量化基础

### 线性量化

把浮点数映射到整数：

$$
x_q = \text{round}\left(\frac{x - z}{s}\right)
$$

- **s (scale)**：缩放因子
- **z (zero-point)**：零点偏移
- **x_q**：量化后的整数

**反量化**：

$$
x \approx s \cdot x_q + z
$$

### 例子

```
原始值: [-0.5, 0.0, 0.5, 1.0, 1.5]
量化到 INT8 (0-255):
  scale = (1.5 - (-0.5)) / 255 = 0.00784
  zero_point = 64

量化后: [0, 64, 128, 192, 255]
```

---

## 常见量化方式

| 类型 | Bits | 大小比例 | 精度损失 |
|------|------|----------|----------|
| FP32 | 32 | 1.0x | 无 |
| FP16 | 16 | 0.5x | 很小 |
| INT8 | 8 | 0.25x | 小 |
| INT4 | 4 | 0.125x | 中等 |
| **3-bit** | 3 | 0.09x | **需要技巧** |

---

## Post-Training Quantization (PTQ)

**流程**：训练完 → 量化

```python
# 简化示例
def quantize_weights(model, bits=8):
    for param in model.parameters():
        # 找到 min/max
        min_val, max_val = param.min(), param.max()
        # 计算 scale
        scale = (max_val - min_val) / (2**bits - 1)
        # 量化
        param_q = torch.round((param - min_val) / scale)
        # 存储
        ...
```

**问题**：3-bit PTQ 精度损失大，因为训练时没考虑量化误差。

---

## Quantization-Aware Training (QAT) 🔥

**流程**：训练时就模拟量化效果

```python
# 前向传播时
def forward(self, x):
    # 模拟量化
    w_q = fake_quantize(self.weight, bits=3)
    return F.linear(x, w_q)

def fake_quantize(x, bits):
    scale = x.abs().max() / (2**(bits-1) - 1)
    x_q = torch.round(x / scale)
    x_q = torch.clamp(x_q, -2**(bits-1), 2**(bits-1)-1)
    # Straight-Through Estimator: 前向量化，反向不变
    return x_q * scale
```

**优点**：
- 模型学会适应量化误差
- 3-bit 也能保持精度
- **榜首团队都用 QAT**

---

## 3-bit 量化的挑战

3 bits = 8 个离散值，比如：`-3, -2, -1, 0, 1, 2, 3, 4`

**问题**：
1. 太少的值无法表示连续分布
2. 异常值（outliers）会浪费量化范围
3. 不同层的分布不同，需要不同的 scale

**解决方案**：
- **Per-channel quantization**：每个 channel 独立量化
- **Mixed precision**：关键层用更多 bit
- **Outlier handling**：特殊处理异常值

---

## 我们的现状

⏳ **还没实现 QAT**

当前流程：
1. FP32 训练
2. 验证时假设会量化到 3-bit
3. 实际提交时再量化

**下一步**：实现 QAT，可能是提升最大的方向之一。

---

## 代码框架

```python
class QuantizedLinear(nn.Module):
    def __init__(self, in_features, out_features, bits=3):
        super().__init__()
        self.bits = bits
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.scale = nn.Parameter(torch.ones(out_features))
    
    def forward(self, x):
        if self.training:
            # QAT: 模拟量化
            w_q = self._fake_quantize(self.weight)
        else:
            # 推理: 真正量化
            w_q = self._real_quantize(self.weight)
        return F.linear(x, w_q)
    
    def _fake_quantize(self, w):
        # Straight-Through Estimator
        ...
```

---

## 榜首技术预览

榜首团队（1.1194 BPB）使用的量化相关技术：
- **QAT**：训练时就考虑量化
- **Per-channel scale**：每个输出通道独立 scale
- **Learned step size**：让 scale 也可学习

这是我们**最需要研究的方向**。

---

*上一篇：[优化器](04-optimizers.md) | 下一篇：[非 Transformer 架构](06-alternative-architectures.md)*

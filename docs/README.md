# Parameter Golf 技术文档

> 让入门者也能看懂的技术解释。每篇文档覆盖一类技术。

## 文档目录

| 文档 | 内容 | 我们用了吗？ |
|------|------|-------------|
| [01-transformer-basics.md](01-transformer-basics.md) | Transformer 基础：Attention、FFN、LayerNorm | ✅ 基础架构 |
| [02-activation-functions.md](02-activation-functions.md) | 激活函数：ReLU、SwiGLU、LeakyReLU² | ✅ LeakyReLU² 有效 |
| [03-attention-variants.md](03-attention-variants.md) | 注意力变体：Full、Sliding Window、Sparse | ✅ Sliding Window 有效 |
| [04-optimizers.md](04-optimizers.md) | 优化器：Adam、AdamW、Muon | ✅ AdamW 胜出 |
| [05-quantization.md](05-quantization.md) | 量化：INT8、3-bit、QAT | ⏳ 待研究 |
| [06-alternative-architectures.md](06-alternative-architectures.md) | 非 Transformer：Mamba、SSM、RWKV | ❌ Mamba 失败 |
| [07-training-techniques.md](07-training-techniques.md) | 训练技巧：Learning Rate、Warmup、Checkpoint | ✅ 在用 |

## 阅读建议

**新手路线**：01 → 02 → 04 → 03

**想理解我们的实验**：02（LeakyReLU²）→ 03（Sliding Window）→ 06（为什么 Mamba 失败）

---

*文档持续更新中*

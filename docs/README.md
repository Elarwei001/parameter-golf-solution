# Parameter Golf Technical Docs

> Beginner-friendly explanations of the techniques we explored. Each doc covers one category.

## Table of Contents

| Doc | Content | Did we use it? |
|-----|---------|----------------|
| [01-transformer-basics.md](01-transformer-basics.md) | Transformer fundamentals: Attention, FFN, LayerNorm | ✅ Base architecture |
| [02-activation-functions.md](02-activation-functions.md) | Activation functions: ReLU, SwiGLU, LeakyReLU² | ✅ LeakyReLU² worked |
| [03-attention-variants.md](03-attention-variants.md) | Attention variants: Full, Sliding Window, Sparse | ✅ Sliding Window worked |
| [04-optimizers.md](04-optimizers.md) | Optimizers: Adam, AdamW, Muon | ✅ AdamW won |
| [05-quantization.md](05-quantization.md) | Quantization: INT8, 3-bit, QAT | ⏳ To explore |
| [06-alternative-architectures.md](06-alternative-architectures.md) | Non-Transformer: Mamba, SSM, RWKV | ❌ Mamba failed |
| [07-training-techniques.md](07-training-techniques.md) | Training tricks: LR, Warmup, Checkpoint | ✅ In use |

## Reading Suggestions

**Beginner path**: 01 → 02 → 04 → 03

**To understand our experiments**: 02 (LeakyReLU²) → 03 (Sliding Window) → 06 (Why Mamba failed)

---

*Docs are continuously updated*

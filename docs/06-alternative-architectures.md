# Alternative Architectures

> Transformer isn't the only option. Mamba and other SSMs have advantages on long sequences, but our experiments failed.

## One-Line Definition

**SSM (State Space Model) = Replace Attention with state equations**

Turns sequence modeling into a discretized continuous dynamical system.

---

## Why Explore Alternative Architectures?

Transformer's problems:

| Problem | Cause |
|---------|-------|
| O(n²) complexity | Every token attends to all tokens |
| KV Cache bloat | Cache grows linearly during inference |
| Long sequence OOM | 100K tokens = memory explosion |

**Ideal alternative**: O(n) complexity + constant memory + maintained performance

---

## Mamba (What We Tried)

### Core Idea

Treat a discrete token sequence as samples from a continuous signal:

$$
\begin{aligned}
h'(t) &= Ah(t) + Bx(t) \quad \text{(state update)} \\
y(t) &= Ch(t) \quad \text{(output)}
\end{aligned}
$$

- **h(t)**: hidden state
- **A, B, C**: learnable parameters
- Key: **A, B, C vary with input** (selective SSM)

### Mamba vs Transformer

| Feature | Transformer | Mamba |
|---------|-------------|-------|
| Complexity | O(n²) | O(n) |
| Long sequences | Slow/OOM | Efficient |
| Short sequences | Very strong | Decent |
| Parallel training | Efficient | Needs tricks |
| Inference | KV cache bloat | Constant memory |

### Our Experimental Results

```
Baseline (Transformer): 2.49 BPB
Mamba:                  5.42 BPB ← Much worse!
```

**Why did it fail?**

1. **Sequences too short** (128 tokens): Mamba's advantage is in long sequences
2. **Pure PyTorch implementation too slow**: Didn't use the official CUDA kernel
3. **Small parameter count**: Mamba may need more parameters to shine

---

## RWKV (Another Alternative)

### Core Idea

Combines RNN and Transformer:
- Trains like a Transformer (parallel)
- Infers like an RNN (O(1) memory)

```
RWKV = R(eceptance) W(eighted) K(ey) V(alue)
```

### Features

- Completely attention-free
- Linear complexity
- Models up to 14B parameters already exist

We didn't try it due to high implementation complexity.

---

## Why Mamba Failed in Parameter Golf

### 1. Short Sequence Disadvantage

```
Sequence length: 128 tokens

Transformer O(n²) = 128² = 16,384 operations
Mamba O(n) = 128 operations (but large constant)

With short sequences, Transformer's O(n²) is not even a bottleneck!
```

### 2. Attention's Modeling Advantage

On short sequences, Attention can directly see all token relationships:

```
Transformer: token_50 directly attends to token_1
Mamba: token_50 indirectly "remembers" token_1 through state
```

Direct > indirect, especially when sequences are short.

### 3. Engineering Issues

```python
# Our implementation (slow)
def mamba_forward(x, A, B, C):
    h = torch.zeros(...)
    outputs = []
    for t in range(seq_len):
        h = A @ h + B @ x[t]
        outputs.append(C @ h)
    return torch.stack(outputs)

# Official implementation uses CUDA kernel — 10x+ faster
```

---

## Lessons Learned

| Lesson | Details |
|--------|---------|
| **Match use case** | Mamba suits long sequences; ours are short |
| **Don't reinvent the wheel** | Use official implementations, don't write your own |
| **Baseline first** | Confirm baseline before exploring alternatives |

---

## When to Use Alternative Architectures?

| Scenario | Recommendation |
|----------|----------------|
| Short sequences (<1K) | **Transformer** |
| Medium sequences (1K–10K) | Transformer + Flash Attention |
| Long sequences (10K–100K) | Mamba or hybrid architectures |
| Very long sequences (>100K) | **Mamba / RWKV** |
| Real-time streaming | Mamba (constant memory) |

---

## Conclusion

For Parameter Golf (128-token short sequences):

> **Transformer is still the best choice**

Alternative architectures like Mamba shine in long-sequence settings — which doesn't apply here.

---

*Previous: [Quantization](05-quantization.md) | Next: [Training Techniques](07-training-techniques.md)*

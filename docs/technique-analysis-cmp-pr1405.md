# Parameter Golf: Leaderboard Technique Analysis

> Analysis of techniques used by top submissions (PR #1405: 1.0856 BPB, PR #1395: 1.0924 BPB, PR #1019: 1.1147 BPB) compared to our dim=448 Sandwich QAT solution.

---

## 1. Scylla Tokenizer (PR #1143)

**What it is**: A custom tokenizer using TokenMonster with only **998 vocabulary entries** — dramatically smaller than standard BPE (8192 or 32768).

**Why it matters**:
- **37% fewer tokens per byte** of text. This means each token encodes more information.
- BPB (bits per byte) = loss × (1/ln2) × (1/tokens_per_byte). Fewer tokens/byte → lower BPB directly.
- Smaller vocab = smaller embedding matrix = more budget for model weights within 16MB.

**How it works**:
- TokenMonster is a greedy longest-match tokenizer optimized for compression ratio
- 998 tokens is surprisingly small but covers common byte patterns efficiently
- The tokenizer is designed specifically to minimize the token-to-byte ratio on FineWeb data
- Requires careful byte accounting in evaluation to ensure fair comparison

**Our gap**: We use standard BPE with vocab=8192. Switching to Scylla would directly improve our BPB by ~20-30% just from the tokenization side.

**Difficulty to adopt**: Medium. Need to re-tokenize the dataset and ensure eval accounts for bytes correctly.

---

## 2. LZMA-9 Compression

**What it is**: Using LZMA (Lempel-Ziv-Markov chain Algorithm) at preset 9 (maximum compression) to compress the model artifact.

**Why it matters**:
- The 16MB limit is on the **compressed artifact size**
- Raw model weights might be 20-30MB, but after LZMA-9 compression fit within 16MB
- This means you can have a **larger, more capable model** that still fits the budget
- LZMA-9 achieves much better compression than gzip or zstd at lower presets

**How it works**:
- LZMA uses a dictionary-based compression with adaptive binary range coding
- Preset 9 uses maximum dictionary size (64MB), slowest but best compression
- Quantized weights (int6) have patterns that compress very well
- Some teams use Brotli-10 (PR #1395) which is faster with similar compression

**Our gap**: We store raw ternary weights without compression. Adding LZMA could let us increase dim or layers while staying under 16MB.

**Impact**: Could allow ~20-30% more parameters within the same 16MB budget.

**Difficulty to adopt**: Easy. Just add compression/decompression to the save/load pipeline.

---

## 3. Parallel Muon Optimizer

**What it is**: A variant of the **Muon optimizer** (Momentum-driven Orthogonalization by Newton's method), adapted for multi-GPU parallel training.

**Why it matters**:
- Muon orthogonalizes the gradient matrix, leading to more efficient optimization
- Converges faster than AdamW for the same number of steps
- Better final loss than AdamW given the same compute budget
- "Parallel" means it works correctly across 8 GPUs with data parallelism

**How it works**:
- Standard Adam: uses per-element adaptive learning rates
- Muon: treats the weight matrix as a whole, applies matrix orthogonalization via Newton-Schulz iterations
- This encourages the weight matrices to stay well-conditioned during training
- Key insight: for matrix-shaped parameters, element-wise adaptation (Adam) is suboptimal; matrix-level optimization (Muon) is better
- Reference: arXiv:2603.28254 (MuonEq-R variant used in PR #1395)

**Our gap**: We use AdamW. Switching to Muon could improve convergence speed and final loss.

**Difficulty to adopt**: Medium. Need to implement the Newton-Schulz orthogonalization, but implementations exist on GitHub.

---

## 4. EMA (Exponential Moving Average)

**What it is**: Maintain a shadow copy of model weights as an exponential moving average of training weights.

**Why it matters**:
- Training weights oscillate around the optimum; EMA smooths this out
- EMA weights typically achieve 0.5-1.0% lower loss than final training weights
- Almost free (no extra compute, just memory for the shadow copy)
- The model used for quantization and eval is the EMA version

**How it works**:
```
ema_weight = α * ema_weight + (1 - α) * current_weight
```
- α = 0.997 means the EMA averages over roughly 1/(1-0.997) ≈ 333 recent steps
- Applied to all parameters
- At eval time, use ema_weight instead of current_weight

**Our gap**: We don't use EMA. Easy to add and would likely improve our BPB.

**Difficulty to adopt**: Very easy. Just a few lines of code.

---

## 5. SWA (Stochastic Weight Averaging)

**What it is**: Periodically save a snapshot of weights and average them together.

**Why it matters**:
- Similar to EMA but simpler — just average the last N checkpoints
- Combined with EMA for even better results
- Helps find a flatter minimum (better generalization)

**How it works**:
```
# Every 50 steps, save current (or EMA) weights
swa_weight = average(all_saved_snapshots)
```
- PR #1019 uses SWA every 50 steps
- The final model uses both EMA and SWA

**Difficulty to adopt**: Very easy.

---

## 6. Late QAT (Quantization-Aware Training)

**What it is**: Enable quantization noise in training only during the **late phase** (when learning rate is very small).

**Why it matters**:
- Early QAT: model is still learning features → quantization noise hurts learning
- Late QAT: model is fine-tuning → quantization noise helps it adapt to the precision loss
- The key insight is **when** to enable QAT, not just whether to use it

**How it works**:
```
# Enable QAT only when LR scale < 0.15 (i.e., near the end of training)
if current_lr / base_lr < 0.15:
    enable_qat()
```
- Uses STE (Straight-Through Estimator) for gradients through quantization
- The "late" threshold (0.15 of base LR) means QAT kicks in during the last ~15% of training

**Our gap**: We use "adaptive QAT" which switches based on loss convergence rate. Late QAT based on LR schedule might be more reliable.

**Difficulty to adapt**: Easy. Just change the QAT trigger condition.

---

## 7. Partial RoPE (Rotary Position Embedding)

**What it is**: Apply RoPE to only a **subset of the head dimensions** instead of all.

**Why it matters**:
- Standard RoPE: applies position-dependent rotation to all head_dim dimensions
- Partial RoPE (16/64): only the first 16 dimensions get position encoding, the remaining 48 are position-independent
- This frees up 75% of head dimensions to encode content rather than position
- For short sequences (256-512 tokens), you don't need 64 dims of positional info

**How it works**:
```python
# Standard: rotate all head_dim dimensions
x_rotated = apply_rope(x)  # all 64 dims

# Partial: rotate only first 16 dims
x[:, :, :, :16] = apply_rope(x[:, :, :, :16])  # 16 dims
x[:, :, :, 16:] = x[:, :, :, 16:]  # 48 dims unchanged
```

**Our gap**: We apply RoPE to all head dimensions (head_dim=58). Using partial RoPE could improve content representation.

**Difficulty to adopt**: Easy. Just slice the tensor before/after RoPE.

---

## 8. QK-Gain 4.0

**What it is**: A learnable scalar multiplier applied to the Q and K projections before attention.

**Why it matters**:
- Controls the magnitude of attention logits
- Standard scaled dot-product: `scores = (Q @ K^T) / sqrt(d)`
- With QK-Gain: `scores = (gain * Q) @ (gain * K)^T / sqrt(d)` = `gain² * (Q @ K^T) / sqrt(d)`
- A gain of 4.0 means attention logits are 16× larger, making attention more **peaked/selective**
- This acts as a learnable temperature for attention

**How it works**:
- Initialize gain to 1.0, let it learn
- The optimal value (4.0) suggests attention should be more selective than the default scaling
- Higher gain → sharper attention distribution → each position attends to fewer positions

**Our gap**: We use standard 1/sqrt(d) scaling. Adding QK-Gain could improve attention quality.

**Difficulty to adopt**: Very easy. One learnable scalar parameter per layer.

---

## 9. VRL (Value Residual Learning)

**What it is**: Add a **residual connection from the value projection to the output**, inspired by ResFormer (arXiv:2410.17897).

**Why it matters**:
- Standard attention: output = softmax(QK^T) @ V, then project with Wo
- VRL: output = softmax(QK^T) @ V + α * V (add a skip from input values)
- This ensures the original value information isn't lost through the attention mechanism
- Similar spirit to our mHC, but specifically for the attention value path

**How it works**:
```python
# Standard attention
attn_out = softmax(scores) @ V
output = Wo(attn_out)

# With VRL
attn_out = softmax(scores) @ V
output = Wo(attn_out) + alpha * V  # residual from V
```
- α is learnable (initialized to a small value or 1.0)
- Helps gradient flow through deep attention stacks

**Our gap**: Our mHC provides similar benefits at the block level (α*x + β*attn(x)). VRL is more targeted at the attention mechanism specifically.

**Difficulty to adapt**: Easy. One line change in the attention forward pass.

---

## Priority Ranking (What to Adopt First)

Based on expected BPB improvement vs implementation effort:

| Priority | Technique | Expected Impact | Effort |
|----------|-----------|----------------|--------|
| 🥇 | **Scylla tokenizer** | Very High (37% fewer tokens) | Medium |
| 🥈 | **LZMA compression** | High (20-30% more params) | Easy |
| 🥉 | **Full Hessian GPTQ int6** | High (better quantization) | Hard |
| 4 | EMA + SWA | Medium (0.5-1% lower loss) | Very Easy |
| 5 | Late QAT (LR-based) | Medium | Easy |
| 6 | QK-Gain | Small-Medium | Very Easy |
| 7 | Parallel Muon | Medium | Medium |
| 8 | Partial RoPE | Small | Easy |
| 9 | VRL | Small | Easy |

---

## Summary

The leaderboard teams' advantage comes primarily from **3 things**:

1. **Tokenizer efficiency** (Scylla): This is the single biggest factor. 37% fewer tokens/byte is a massive head start in BPB.

2. **Better quantization** (Full Hessian GPTQ int6 + compression): They fit more effective model capacity into 16MB by using smarter quantization + LZMA compression.

3. **Training tricks stack** (Muon, EMA, SWA, Late QAT, QK-Gain, etc.): Many small improvements that compound.

Our advantage is **LoRA+TTT** — a genuinely different approach that adapts the model at eval time. If we combine TTT with the above techniques, we could potentially achieve significantly better results.

---

*Generated: 2026-04-06*

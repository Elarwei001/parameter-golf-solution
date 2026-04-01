# Embedding Space Efficiency: A Research Direction

> **Origin**: This idea came from Elar during the Parameter Golf competition (2026-04-01).
> 
> **Core Insight**: Whether we're optimizing tokenizers or applying quantization, we're fundamentally optimizing the vector space. But what if the learned vector space itself is still sparse and redundant?

---

## The Problem

Neural networks learn high-dimensional embeddings (e.g., 512 dimensions), but research shows these spaces are often:

1. **Low-rank**: Most information concentrated in a few principal components
2. **Anisotropic**: Vectors clustered in a narrow cone instead of uniformly distributed
3. **Redundant**: Many dimensions carry little unique information

If we could **densify** this space — pack the same semantic information into fewer, more efficiently-used dimensions — we could:

- Maintain complex semantic relationships
- Reduce model size without losing capacity
- Speed up inference (fewer dimensions = less computation)

---

## Theoretical Framework

### 1. Measuring Space Efficiency

#### Participation Ratio (Effective Dimensionality)

For an embedding matrix with singular values $\sigma_1, \sigma_2, ..., \sigma_d$:

$$PR = \frac{(\sum_i \sigma_i^2)^2}{\sum_i \sigma_i^4}$$

| PR Value | Interpretation |
|----------|----------------|
| PR ≈ d | All dimensions equally utilized (ideal) |
| PR << d | Information concentrated in few dimensions (redundant) |
| PR = 1 | Effectively 1-dimensional (extreme redundancy) |

**Intuition**: If a 512-dim embedding has PR = 100, then ~400 dimensions are "wasted".

#### Cumulative Explained Variance

How many principal components needed to capture X% of variance:

$$\text{Explained}(k) = \frac{\sum_{i=1}^{k} \sigma_i^2}{\sum_{i=1}^{d} \sigma_i^2}$$

If 95% variance is captured in 150 dims, the remaining 362 dims contribute only 5%.

#### Anisotropy Score

Average cosine similarity between random vector pairs:

$$\text{Anisotropy} = \mathbb{E}_{x,y}[\cos(x, y)]$$

| Score | Interpretation |
|-------|----------------|
| ~0 | Isotropic — vectors uniformly distributed (good) |
| >0.3 | Anisotropic — vectors clustered together (bad) |

BERT/GPT embeddings often have anisotropy 0.5-0.8, indicating severe clustering.

---

### 2. Potential Solutions

#### A. Whitening (Post-hoc)

Transform embeddings to be isotropic:

$$z = W(x - \mu)$$

where $W = \Sigma^{-1/2}$ (inverse square root of covariance).

**Pros**: Simple, no retraining needed  
**Cons**: Linear transformation, may lose non-linear structure

**Reference**: *"Whitening Sentence Representations for Better Semantics"* (Su et al., 2021)

#### B. Matryoshka Representation Learning (MRL)

Train embeddings so that prefixes are also meaningful:

```
768-dim embedding where:
- First 64 dims: coarse semantics (usable alone)
- First 256 dims: medium detail
- Full 768 dims: complete representation
```

At inference, use only as many dimensions as needed.

**Reference**: *"Matryoshka Representation Learning"* (Kusupati et al., 2022)

#### C. Product Quantization (PQ)

Split vector into subvectors, quantize each independently:

```
768 dims → 8 subvectors of 96 dims
Each subvector → 256 centroids (8 bits)
Total: 8 × 8 = 64 bits per vector
```

**Pros**: Extreme compression  
**Cons**: Lossy, requires codebook

#### D. Learned Densification (Novel Direction)

Learn a projection that maximizes space utilization:

$$z = f_\theta(x) \quad \text{where } z \in \mathbb{R}^{d'}, d' < d$$

**Training objectives**:
1. **Reconstruction**: $\|g_\phi(z) - x\|^2$ (can recover original)
2. **Isotropy**: $\text{KL}(p(z) \| \mathcal{U})$ (z uniformly distributed)
3. **Independence**: Encourage orthogonal dimensions

```python
class DensifyProjection(nn.Module):
    def __init__(self, d_in, d_out):
        self.encoder = nn.Linear(d_in, d_out)
        self.decoder = nn.Linear(d_out, d_in)
    
    def forward(self, x):
        z = self.encoder(x)
        
        # Regularization for uniform distribution
        # Could use: batch normalization, spectral normalization,
        # or adversarial training against uniform prior
        
        return z
    
    def loss(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        
        recon_loss = F.mse_loss(x_recon, x)
        isotropy_loss = self.compute_isotropy_penalty(z)
        
        return recon_loss + λ * isotropy_loss
```

---

## Application to Parameter Golf

### Current Situation

- Model: 512 dimensions, ~30M parameters
- Constraint: ≤16MB at 3-bit quantization
- Question: Is 512 the right dimension, or is it wasteful?

### Hypothesis

If trained embeddings have PR ≈ 200 (out of 512):
- ~60% of embedding dimensions are redundant
- Could potentially use dim=256 with densification
- Saved parameters → more layers or larger vocab

### Potential Experiment

```python
# After training baseline model:
# 1. Measure embedding efficiency
baseline_PR = measure_participation_ratio(model.embedding)
print(f"Effective dims: {baseline_PR} / 512")

# 2. If PR < 300, try smaller dim with whitening
if baseline_PR < 300:
    # Train dim=256 model
    # Apply whitening to embeddings
    # Compare BPB
```

### Expected Gains

| Scenario | Embedding Params | Available for Layers | Potential BPB |
|----------|-----------------|---------------------|---------------|
| Baseline (dim=512) | 8M (vocab×dim) | 22M | 1.40 |
| Densified (dim=384) | 6M | 24M (+9%) | ? |
| Densified (dim=256) | 4M | 26M (+18%) | ? |

---

## Related Work

1. **Intrinsic Dimensionality of Language Models**
   - *"Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning"* (Aghajanyan et al., 2020)
   - Found that fine-tuning happens in very low-dimensional subspaces

2. **Embedding Compression**
   - *"Compressing BERT: Studying the Effects of Weight Pruning on Transfer Learning"* (Gordon et al., 2020)
   
3. **Isotropy in Embeddings**
   - *"How Contextual are Contextualized Word Representations?"* (Ethayarajh, 2019)
   - Showed BERT embeddings are highly anisotropic

4. **Whitening for Better Representations**
   - *"Whitening Sentence Representations for Better Semantics and Faster Retrieval"* (Su et al., 2021)

---

## Next Steps

1. **Measure**: Run `analyze_embedding_space.py` on trained models
2. **Compare**: Track PR/anisotropy across training
3. **Experiment**: Try whitening on final embeddings
4. **Innovate**: Design learnable densification layer

---

## Open Questions

1. Does quantization (QAT) affect embedding space geometry?
2. Is there an optimal PR for language modeling? (Not too high, not too low?)
3. Can we train with an isotropy regularizer from the start?
4. How does tokenizer choice (BPE size) affect embedding efficiency?

---

*Last updated: 2026-04-01*
*Author: Elar + Arae*

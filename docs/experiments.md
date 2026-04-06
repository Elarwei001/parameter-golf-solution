# Parameter Golf - Experiment Log

## Experiment 1: dim=464, 30L, BPE 8192

**Date**: 2026-04-04 ~ 2026-04-05

**Config**:
- dim=464, n_heads=8, n_kv=4, head_dim=58
- 30 layers: 6× 3.0x MLP + 21× 1.2x MLP + 3× 3.0x MLP
- Vocab: BPE 8192
- mHC, alt-attn (global/local), adaptive QAT, ternary quantization
- 40k steps, single A100-40GB

**Results**:
- Val BPB: **1.3546** (standard eval)
- Model size: ~18.09MB (FP16 embedding) — **over 16MB limit**
- Pre-TTT BPB: 1.6232
- Post-TTT BPB (LoRA rank=8, 2 epochs, lr=0.01): **1.0715** (34.0% improvement)

**Key findings**:
- LoRA+TTT is extremely effective, closing gap from 1.62 → 1.07
- Model too large for 16MB constraint
- Shape mismatch error at dim=460 (head_dim not integer) — fixed by using dim=464

---

## Experiment 2: dim=448, 30L, BPE 8192

**Date**: 2026-04-05

**Config**:
- dim=448, n_heads=8, n_kv=4, head_dim=56
- 30 layers: 2× 3.0x MLP + 28× 1.2x MLP
- Vocab: BPE 8192
- Same architecture as dim=464 but reduced MLP scale to fit 16MB
- 40k steps, single A100-40GB

**Results**:
- Val BPB: **1.3805** (standard eval)
- Model size: ~14.36MB (FP16 embedding + ternary) — **fits 16MB** ✅
- Pre-TTT BPB: 1.6564
- Post-TTT BPB (LoRA rank=8, 2 epochs, lr=0.01): **1.0885** (34.3% improvement)

**Comparison with leaderboard**:
| Model | Post-TTT BPB | Size |
|-------|-------------|------|
| dim=464 | **1.0715** | 18.09MB ❌ |
| dim=448 | **1.0885** | 14.36MB ✅ |
| Leaderboard #1 (PR #1405) | **1.0856** | 15.8MB ✅ |
| PR #1399 | **1.0898** | ~16MB ✅ |

**Key findings**:
- dim=448 with LoRA+TTT is competitive with leaderboard top entries
- Only 0.003 BPB behind #1, within striking distance
- Reducing 3.0x layers from 9 to 2 saved significant space

---

## Experiment 3: dim=544, 30L, Scylla 998

**Date**: 2026-04-06 (in progress)

**Config**:
- dim=544, n_heads=8, n_kv=4, head_dim=68
- 30 layers: 9× 3.0x MLP + 21× 1.2x MLP
- Vocab: **Scylla 998** (TokenMonster)
- Same architecture (mHC, alt-attn, adaptive QAT, ternary)
- 40k steps, single A100-40GB
- Dataset: FineWeb10B retokenized with Scylla

**Expected advantages**:
- Embedding shrinks from 7.34MB → 0.90MB (vocab 8192→998)
- Model params: 58.09M (vs 41.3M for dim=448) — 41% more capacity
- Estimated size: 14.75MB ✅
- Scylla tokens/byte ~4.13 (vs ~3.67 for BPE 8192) → direct BPB improvement
- After LoRA+TTT, expect BPB significantly lower than dim=448's 1.0885

**Status**: Training started, ~4% complete

---

## Technique Gap Analysis

Compared to leaderboard #1 (PR #1405, 1.0856 BPB):

| Technique | Us | PR #1405 | Priority |
|-----------|-----|----------|----------|
| Tokenizer | BPE 8192 → Scylla 998 ✅ | Scylla 998 | Done |
| Quantization | Ternary (1.58-bit) | Full Hessian GPTQ int6 | Future |
| Compression | None | LZMA-9 | Future |
| Optimizer | AdamW | Parallel Muon | Future |
| EMA/SWA | None | Yes | Easy win |
| LoRA+TTT | ✅ | No | Our advantage |
| BigramHash | No | 3072×112 | Future |
| QK-Gain | No | 4.0 | Easy win |

---

*Last updated: 2026-04-06*

# dim=448 Sandwich QAT — LoRA TTT Evaluation

Date: 2026-04-05

## Status: ✅ Complete

## Config
- dim=448, n_heads=8, n_kv=4, head_dim=56
- 30 layers: 2× 3.0x MLP + 28× 1.2x MLP
- Vocab: BPE 8192
- mHC, alt-attn, adaptive QAT, ternary quantization
- 40k steps, single A100-40GB

## Results

### Training
- Val BPB: **1.3805**
- Model size: ~14.36MB (FP16 embedding + ternary) ✅

### LoRA TTT Evaluation
- Pre-TTT BPB: 1.6564
- Post-TTT BPB: **1.0885** (rank=8, epochs=2, lr=0.01)
- Improvement: 34.3%

### Leaderboard Comparison
| Model | BPB | Size |
|------|-----|------|
| dim=448 + LoRA TTT | 1.0885 | 14.36MB |
| Leaderboard #1 (PR #1405) | 1.0856 | 15.8MB |

## Files
- `train.log` — Training log
- `lora_ttt.log` — LoRA TTT evaluation log

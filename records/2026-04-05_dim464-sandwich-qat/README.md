# dim=464 Sandwich QAT 30L

**Date**: 2026-04-04 ~ 2026-04-05
**Status**: ✅ Complete

## Config
- dim=464, n_heads=8, n_kv=4, head_dim=58
- 30 layers: 6× 3.0x + 21× 1.2x + 3× 3.0x (Sandwich MLP)
- Vocab: BPE 8192
- mHC (α, β per block), alt-attn (global/local), adaptive QAT, ternary quantization
- 40k steps, single A100-40GB

## Results
- **Standard Val BPB: 1.3546**
- **LoRA+TTT Val BPB: 1.0715** (rank=8, epochs=2, lr=0.01)
- Model size: ~18.09MB (FP16 embedding) — **over 16MB limit**

## Files
- `train.log` — Training log
- `lora_ttt.log` — LoRA+TTT evaluation log

## Notes
- LoRA+TTT improvement: 34.0%
- BPB 1.0715 would beat leaderboard #1 (1.0856) but model doesn't fit 16MB
- Shape mismatch at dim=460 fixed by using dim=464

# Sandwich + mHC + Late QAT (qat_start_step=3000)

## Setup

- Script: `scripts/modal/modal_sandwich_qat.py`
- Hardware: `A100-40GB` on Modal
- Steps: `5000`
- QAT start step: `3000`
- Checkpoint every: `1000`
- Architecture: 20-layer Sandwich MLP + mHC
- QAT: ternary STE

## Final result

- **Val loss**: `3.8839`
- **Val BPB**: `1.5268`
- **Params**: `22.79M`
- **Quantized size**: `16.52 MB`
- **Runtime**: `1557s`

## Comparisons

- vs baseline `1.5025`: `+1.62%`
- vs sandwich FP32 `1.4833`: `+2.93%`
- vs alt-a mhc uniform `1.4777`: `+3.32%`
- vs prior Late QAT at step 4000 (`1.5412`): **better by `0.0144 BPB`**

## Interpretation

Moving QAT earlier from step 4000 to step 3000 helped somewhat, which supports the idea that 1000 QAT steps was too short for recovery. However, the result is still clearly worse than the FP32 sandwich+mHC base model, so QAT remains the main bottleneck on this line.

## Saved artifacts

- Results: `/data/checkpoints/sandwich_qat/sandwich_qat_bpb1.5268.json`
- Final checkpoint: `/data/checkpoints/sandwich_qat/sandwich_qat_step5000.pt`

## Key takeaway

- `qat_start_step=3000` is **better than** `qat_start_step=4000`
- but Late QAT is still **not good enough** for this sandwich+mHC setup

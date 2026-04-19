# Sandwich + mHC + QAT Ramp (start=2000, ramp=1000)

## Setup

- Script: `scripts/modal/modal_sandwich_qat.py`
- Hardware: `A100-40GB` on Modal
- Steps: `5000`
- QAT start step: `2000`
- QAT ramp steps: `1000`
- Checkpoint every: `1000`
- Architecture: 20-layer Sandwich MLP + mHC
- QAT: ternary STE with linear FP32->QAT ramp

## Final result

- **Val loss**: `3.8753`
- **Val BPB**: `1.5234`
- **Params**: `22.79M`
- **Quantized size**: `16.52 MB`

## Comparisons

- vs late QAT step 3000 (`1.5268`): **better by `0.0034 BPB`**
- vs adaptive QAT 20L (`1.5321`): **better by `0.0087 BPB`**
- vs sandwich FP32 (`1.4833`): `+2.70%`
- vs alt-a mhc uniform (`1.4777`): `+3.09%`

## Interpretation

A linear ramp from FP32 to QAT improved over the previous hard-switch late-QAT setup. The gain is modest, but it supports the hypothesis that switch smoothness matters in addition to switch timing. QAT remains the dominant bottleneck, yet ramping appears to reduce some of the transition shock.

## Saved artifacts

- Results: `/checkpoints/sandwich_qat/sandwich_qat_bpb1.5234.json`
- Final checkpoint: `/checkpoints/sandwich_qat/sandwich_qat_step5000.pt`

## Key takeaway

- **QAT ramp beats hard-switch late QAT (3000)**
- The improvement is small but real
- Worth continuing with earlier or longer ramps

# 11L Depth Utilization Compare (baseline vs recurrence vs sandwich+mHC)

## Purpose

A clean 3-way comparison to answer one question:

> Under the same small-budget setup, is `sandwich + mHC` actually better than a simpler `depth recurrence` baseline?

This experiment intentionally compares **base-model quality only**. No TTT was used.

## Common setup

- Tokenizer: `SP8192`
- Hardware: `A100-40GB` on Modal
- Steps: `5000`
- Dim: `448`
- Physical layers: `11`
- Heads / KV heads: `8 / 4`
- Local window: `128`
- Batch size: `64`
- Seq len: `256`
- LR: `1e-3` with warmup + cosine decay
- Seeds: `42`

## Variants

### 1. baseline
- 11 plain layers
- uniform MLP scale `3.0x`
- no mHC
- no recurrence

### 2. recurrence
- same as baseline
- replay middle layers `4-5` once more during forward pass
- no mHC

### 3. sandwich
- 11 layers
- MLP scales: `3,3,3,1.2,1.2,1.2,1.2,1.2,3,3,3`
- mHC enabled
- no recurrence

## Results

| Variant | Val loss | Val BPB | Params | Runtime |
|---|---:|---:|---:|---:|
| baseline | 3.7995 | 1.4936 | 23.55M | 1192s |
| recurrence | not finished | not finished | ~23.55M | stopped after 4500 steps / 1241s |
| sandwich | 3.7871 | 1.4887 | 19.93M | 1194s |

## Key takeaways

1. **sandwich + mHC beat plain baseline** in this 5k-step test.
   - `1.4936 -> 1.4887`
   - Improvement: about `0.0049 BPB`

2. **sandwich used fewer params** while still doing slightly better.
   - baseline: `23.55M`
   - sandwich: `19.93M`

3. **recurrence did not complete** because the Modal app stopped when the local client disconnected.
   - Last visible training log was `step 4500`, `loss 3.7816`, runtime `1241s`
   - No final validation result was captured for this run

4. For future reruns, do not rely on local OpenClaw `SIGTERM` as a failure signal.
   - Modal remote jobs can continue after local process tracking changes
   - Use `modal app list`, `modal app logs <app_id>`, and saved remote artifacts to judge job state

## Logged Modal runs

- baseline app id: `ap-AzFAcbZKXpjCWJtT6qbLom`
- recurrence app id: `ap-R4lMVKid8qfw1H1fftnRtd`
- sandwich app id: `ap-3Fz0Ike7Zt0qXrsWdananP`

## Raw logs extracted

### baseline
- `Step 500/5000 | Loss 5.0633 | LR 9.91e-04 | Time 121s`
- `Step 1000/5000 | Loss 4.5459 | LR 9.40e-04 | Time 240s`
- `Step 1500/5000 | Loss 4.3832 | LR 8.47e-04 | Time 359s`
- `Step 2000/5000 | Loss 4.1494 | LR 7.22e-04 | Time 478s`
- `Step 2500/5000 | Loss 4.0416 | LR 5.79e-04 | Time 597s`
- `Step 3000/5000 | Loss 3.8423 | LR 4.34e-04 | Time 716s`
- `Step 3500/5000 | Loss 3.9693 | LR 3.00e-04 | Time 835s`
- `Step 4000/5000 | Loss 3.9586 | LR 1.93e-04 | Time 954s`
- `Step 4500/5000 | Loss 3.7775 | LR 1.24e-04 | Time 1073s`
- `Step 5000/5000 | Loss 3.7890 | LR 1.00e-04 | Time 1192s`
- final: `val_loss=3.7995`, `val_bpb=1.4936`

### recurrence
- `Step 500/5000 | Loss 5.0821 | LR 9.91e-04 | Time 140s`
- `Step 1000/5000 | Loss 4.5694 | LR 9.40e-04 | Time 277s`
- `Step 1500/5000 | Loss 4.3995 | LR 8.47e-04 | Time 415s`
- `Step 2000/5000 | Loss 4.1707 | LR 7.22e-04 | Time 553s`
- `Step 2500/5000 | Loss 4.0441 | LR 5.79e-04 | Time 690s`
- `Step 3000/5000 | Loss 3.8550 | LR 4.34e-04 | Time 828s`
- `Step 3500/5000 | Loss 3.9731 | LR 3.00e-04 | Time 966s`
- `Step 4000/5000 | Loss 3.9647 | LR 1.93e-04 | Time 1103s`
- `Step 4500/5000 | Loss 3.7816 | LR 1.24e-04 | Time 1241s`
- no final validation captured

### sandwich
- `Step 500/5000 | Loss 5.0613 | LR 9.91e-04 | Time 123s`
- `Step 1000/5000 | Loss 4.3955 | LR 9.40e-04 | Time 242s`
- `Step 1500/5000 | Loss 4.2622 | LR 8.47e-04 | Time 361s`
- `Step 2000/5000 | Loss 4.0931 | LR 7.22e-04 | Time 480s`
- `Step 2500/5000 | Loss 4.0157 | LR 5.79e-04 | Time 599s`
- `Step 3000/5000 | Loss 4.0700 | LR 4.34e-04 | Time 718s`
- `Step 3500/5000 | Loss 3.9330 | LR 3.00e-04 | Time 837s`
- `Step 4000/5000 | Loss 3.8060 | LR 1.93e-04 | Time 956s`
- `Step 4500/5000 | Loss 3.7770 | LR 1.24e-04 | Time 1075s`
- `Step 5000/5000 | Loss 3.8080 | LR 1.00e-04 | Time 1194s`
- final: `val_loss=3.7871`, `val_bpb=1.4887`

## mHC snapshot at step 5000 (sandwich)

Notable pattern:
- L04 global had the strongest attention amplification among the middle 1.2x layers: `beta_attn=1.213`
- Deep 3.0x layers did not dominate as strongly as expected in this short 5k-step run
- Layer 0 remained unusual, with `alpha_attn=1.232`, `beta_attn=0.132`

Selected final values:
- `L00 Global mlp=3.0 aA=1.232 bA=0.132 aM=1.015 bM=0.894`
- `L04 Global mlp=1.2 aA=0.875 bA=1.213 aM=0.963 bM=0.674`
- `L10 Global mlp=3.0 aA=0.922 bA=0.609 aM=0.890 bM=0.625`

## Files

- script: `scripts/modal/modal_depth_compare.py`
- this note: `records/2026-04-13_depth-compare-11l/README.md`

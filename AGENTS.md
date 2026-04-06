# AGENTS.md - Repository Guidelines

## Language

**All code and documentation must be in English.**

- README.md, EXPERIMENTS.md, all docs/*.md files
- Code comments
- **Print statements and log output** (logs will be submitted)
- Commit messages
- Exception: Chinese is allowed in personal notes or scratch files (not committed)

## No Emoji in Code

**Do not use emoji in code, comments, or print statements.**

- Use text markers instead: `[TRAIN]`, `[RESULTS]`, `[CHECKPOINT]`, etc.
- Logs should be professional and terminal-friendly
- Emoji can cause encoding issues in some environments

## Code Style

- Python: Follow PEP 8
- Use type hints where practical
- Docstrings for public functions

## Commit Messages

Format: `<type>: <description>`

Types:
- `feat`: New feature or experiment
- `fix`: Bug fix
- `docs`: Documentation only
- `refactor`: Code refactoring
- `exp`: Experiment results/logs

Example: `feat: Add MHC residual coefficient analysis`

## Experiment Workflow

All experiments are classified as **Pending** or **Verified**.

### Pending (待验证)
- All untested ideas live in **`docs/todo.md`**
- Include hypothesis, expected impact, and priority
- When moving to verified, check the item off

### Verified (已验证)
After an experiment completes, do ALL of the following:

1. **Update the log**: Add entry to `docs/experiment.md` (航海日志, append-only)
2. **Record results**: Create a subdirectory under `results/` with:
   - **Analysis with figures** — experiment result analysis with plots/charts
   - **Training logs** — `train.log`, `lora_ttt.log`, etc.
   - **Script backup** — `scripts/modal/*.py.bak` or copy of the exact script used
3. **Update `docs/todo.md`**: Check off the verified item

Example structure:
```
results/
├── 2026-04-05_dim448-sandwich-qat/
│   ├── analysis.md          # Results analysis with embedded figures
│   ├── figures/
│   │   └── mhc_params.png
│   ├── train.log
│   ├── lora_ttt.log
│   └── modal_sandwich_qat_30l_dim448.py.bak
```

### Experiment Tracking Legacy

Old records live in `records/` directory. New experiments should use `results/` structure above.
Visualization files go to `results/figures/` or per-experiment subdirectory.

## File Organization

```
parameter-golf-solution/
├── AGENTS.md                   # This file
├── README.md                   # Project overview
├── BASELINE_CONFIG.md          # ⭐ Standard experiment config (must read!)
├── scripts/
│   ├── modal/                  # Modal cloud training scripts
│   │   ├── modal_mhc_v2_deep.py    # Baseline (mHC v2)
│   │   ├── modal_alternating_attn.py # Alternating Attention
│   │   ├── modal_ple_v2.py         # PLE experiment
│   │   └── ...                     # Other experiments
│   └── local/                  # Local test scripts
├── models/                     # Model implementations
├── docs/
│   ├── experiments/            # Experiment logs
│   │   ├── EXPERIMENTS.md      # ⭐ Main experiment log (append-only)
│   │   ├── MHC_ANALYSIS.md     # mHC deep-dive
│   │   └── TODO.md             # Ideas to try
│   ├── analysis/               # Deep analysis docs
│   └── techniques/             # Tech notes (01-07)
└── results/
    ├── figures/                # All images (PNG/SVG)
    └── logs/                   # JSON logs, metrics
```

## Modal Runs

- Use descriptive app names: `mhc-v2-deepseek-20L`
- Don't poll training logs (expensive!) — just wait for completion
- Save outputs to Modal volumes when needed
- **Always use `modal run --detach`** — prevents task termination if local CLI disconnects
  - Check progress with: `modal app logs <app-id>`
  - App ID is shown in the "View run at" URL (e.g., `ap-xxxxxxxx`)

### Writing New Modal Scripts

**Always copy boilerplate from existing working scripts** — don't write from scratch!

Reference script: `scripts/modal/modal_mhc_v2_deep.py`

Copy these sections verbatim:
1. **Image definition** — use exact torch version (`torch==2.5.1`)
2. **Volume name** — `parameter-golf-data` (not `training-data`)
3. **Data paths** — `fineweb_train_000000.bin` (6 digits, not 3)
4. **`get_batch()` function** — includes `.long()` conversion for targets
5. **HEADER_SIZE** — `256 * 4` bytes to skip

Only modify the model architecture. This avoids:
- CUDA version incompatibility
- FileNotFoundError from wrong paths
- dtype errors (Int vs Long)

### Experiment Discipline

**Before running any experiment:**
1. Read `BASELINE_CONFIG.md` — contains the fixed config
2. Only modify model architecture — keep data/training config identical
3. Compare against baseline BPB: **1.5025** (mHC v2, 20L)

This ensures apples-to-apples comparison.

---

*Last updated: 2026-04-03*

# AGENTS.md - Repository Guidelines

## Language

**All documentation must be in English.**

- README.md, EXPERIMENTS.md, all docs/*.md files
- Code comments
- Commit messages
- Exception: Chinese is allowed in personal notes or scratch files (not committed)

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

## Experiment Tracking

**IMPORTANT**: After every experiment, record results and insights in:
`docs/experiments/EXPERIMENTS.md`

Use **append-only** format — never edit or delete existing entries. This preserves full history.

Each entry should include:
- Date
- Experiment name/ID
- Hypothesis
- Configuration (model size, hyperparams)
- Results (BPB, loss, key metrics)
- Conclusions/insights
- Link to code/logs if applicable

Visualization files go to `results/figures/`, logs to `results/logs/`.

## File Organization

```
parameter-golf-solution/
├── AGENTS.md                   # This file
├── README.md                   # Project overview
├── scripts/
│   ├── modal/                  # Modal cloud training scripts
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

---

*Last updated: 2026-04-03*

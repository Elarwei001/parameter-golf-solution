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

- Log all experiments in `EXPERIMENTS.md`
- Include: date, hypothesis, results, conclusions
- Save training logs as `train_<experiment>.log`
- Save visualizations as PNG (and optionally SVG)

## File Organization

```
parameter-golf-solution/
├── AGENTS.md              # This file
├── README.md              # Project overview
├── EXPERIMENTS.md         # Experiment log
├── docs/                  # Detailed analyses
│   ├── *.md               # Analysis documents
│   └── *.py               # Visualization scripts
├── models/                # Model implementations
├── modal_*.py             # Modal cloud training scripts
└── *.png                  # Generated figures
```

## Modal Runs

- Use descriptive app names: `mhc-v2-deepseek-20L`
- Don't poll training logs (expensive!) — just wait for completion
- Save outputs to Modal volumes when needed

---

*Last updated: 2026-04-03*

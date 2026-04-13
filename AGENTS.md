# AGENTS.md - Repository Maintenance Guidelines

**Read this before making any changes to the Parameter Golf repository.**

## 🌍 Language Policy

**All code and documentation must be in English.**

- README.md, EXPERIMENTS.md, all docs/*.md files
- Code comments and docstrings
- **Print statements and log output** (logs will be submitted)
- Commit messages
- Exception: Chinese is allowed in personal notes or scratch files (not committed)

## 🚫 No Emoji in Code

**Do not use emoji in code, comments, or print statements.**

- Use text markers instead: `[TRAIN]`, `[RESULTS]`, `[CHECKPOINT]`, etc.
- Logs should be professional and terminal-friendly
- Emoji can cause encoding issues in some environments

## 📋 Code Standards

- **Python**: Follow PEP 8, use type hints where practical
- **Docstrings**: Required for all public functions
- **Imports**: Group by standard library, third-party, local imports
- **Error handling**: Use specific exceptions, not bare `except:`

## 📝 Commit Message Format

```
<type>: <description>

<optional body>
```

**Types:**
- `feat`: New feature or experiment
- `fix`: Bug fix  
- `docs`: Documentation updates
- `refactor`: Code refactoring
- `exp`: Experiment results/logs
- `config`: Configuration changes

**Examples:**
- `feat: add mHC depth analysis for 30L models`
- `exp: complete QAT adaptive threshold experiment`
- `docs: update experiment pipeline workflow`

## 🧪 Experiment Pipeline Workflow

### Experiment Lifecycle Management

All experiments follow this standardized pipeline:

```
💡 Idea → 📋 Queue → 🔬 Running → ✅ Completed → 📊 Analyzed → 🏆 Verified
```

### 1. Ideas Management (`docs/todo.md`)

**Before adding new ideas:**
- Check existing ideas in `docs/todo.md` to avoid duplicates
- Categorize by priority: High/Medium/Low
- Include estimated effort and expected impact
- Note any dependencies on other experiments

**Format for new ideas:**
```markdown
- [ ] **Technique Name** — Brief description of the approach
  - Category: architecture/quantization/optimization
  - Estimated effort: X days
  - Expected impact: high/medium/low
  - Dependencies: [list any blocking experiments]
```

### 2. Experiment Execution

**CRITICAL: Read `BASELINE_CONFIG.md` first!**
- Use the exact same data/training configuration
- Only modify model architecture components
- Compare against baseline BPB: **1.5025** (mHC v2, 20L)

**During experiment:**
- Document hypothesis and method clearly
- Save intermediate checkpoints if needed
- Log key observations and unexpected behaviors

### 3. Results Documentation

**After experiment completion, create ALL of the following:**

1. **Experiment record** in `records/YYYY-MM-DD_experiment-name/`:
   ```
   records/2026-04-13_new-technique/
   ├── report.md              # Detailed analysis with methodology
   ├── figures/               # All plots and visualizations
   ├── train.log             # Training logs
   ├── eval.log              # Evaluation logs (if any)
   └── script.py             # Exact script used
   ```

2. **Update main experiment log** (`docs/experiment.md`):
   - Add one-line summary with date, technique, and BPB result
   - This is the master timeline of all experiments

3. **Mark TODO item as completed** in `docs/todo.md`:
   - Check off the item: `- [x] **Technique Name**`
   - Add result: `— Result: X.XX BPB (-Y.Y% vs baseline)`

4. **Update leaderboard** if result is significant:
   - Add to README.md progress table
   - Update best BPB if applicable

### 4. Knowledge Management

**Add insights to appropriate documentation:**
- **Architecture discoveries** → Update relevant model documentation
- **Training insights** → Add to training technique notes  
- **Analysis patterns** → Document in analysis methodology
- **Failed approaches** → Document what didn't work and why

### 5. Follow-up Planning

**Generate follow-up ideas:**
- What variations could be tested?
- What unexpected behaviors need investigation?
- How can this technique be combined with others?
- Add new ideas to `docs/todo.md` with proper categorization

## 📁 Repository Structure

```
parameter-golf-solution/
├── AGENTS.md                   # 👈 This file - READ FIRST!
├── README.md                   # Project overview and progress
├── BASELINE_CONFIG.md          # ⭐ Standard experiment config (MUST READ!)
├── scripts/
│   ├── modal/                  # Modal cloud training scripts  
│   │   ├── modal_mhc_v2_deep.py    # 📌 Baseline reference (mHC v2)
│   │   ├── modal_alternating_attn.py # Alternating Attention
│   │   ├── modal_qat.py            # Quantization experiments
│   │   └── modal_*.py              # Other experiments
│   └── local/                  # Local testing scripts
├── models/                     # Core model implementations
│   ├── latent_lm.py           # Latent space models
│   ├── mamba_lm.py            # Mamba architecture  
│   └── standard_gpt.py        # Standard GPT variants
├── docs/
│   ├── experiment.md          # 📋 Master experiment timeline
│   ├── todo.md                # 💡 Ideas and TODO items
│   ├── mhc-depth-profiling.md # Deep analysis documents
│   └── reference/             # Technical reference docs
├── records/                   # 📊 Experiment records (by date)
│   ├── 2026-04-XX_experiment-name/
│   │   ├── report.md          # Detailed experiment report
│   │   ├── figures/           # Experiment-specific plots
│   │   ├── train.log          # Training logs
│   │   └── script.py          # Script used for experiment
├── results/                   # Centralized results
│   ├── figures/               # All visualization assets
│   └── logs/                  # JSON metrics and logs
├── configs/                   # Configuration management
├── adapters/                  # Cloud platform adapters
├── optimizers/                # Custom optimizers (Muon, etc.)
└── quant/                     # Quantization implementations
```

### 📋 Key Files to Know

| File | Purpose | When to Update |
|------|---------|---------------|
| `BASELINE_CONFIG.md` | Standard experiment settings | Never (reference only) |
| `docs/todo.md` | Ideas and TODO items | When adding/completing ideas |
| `docs/experiment.md` | Master experiment timeline | After every experiment |
| `README.md` | Progress and best results | When achieving new best BPB |
| `records/*/report.md` | Detailed experiment analysis | After completing experiments |

## ☁️ Modal Cloud Execution

### Running Experiments

**ALWAYS use detached mode:**
```bash
modal run --detach scripts/modal/modal_experiment_name.py
```

**Best practices:**
- Use descriptive app names: `mhc-v2-deepseek-20L`
- **Never poll training logs** (expensive!) — wait for completion
- Check progress: `modal app logs <app-id>` (get app-id from "View run at" URL)
- Save important outputs to Modal volumes for persistence

### Creating New Modal Scripts

**🚨 CRITICAL: Always copy from existing working scripts!**

**Reference template:** `scripts/modal/modal_mhc_v2_deep.py`

**Copy these sections EXACTLY:**
1. **Image definition** — exact torch version (`torch==2.5.1`)
2. **Volume configuration** — `parameter-golf-data` (not `training-data`)  
3. **Data paths** — `fineweb_train_000000.bin` (6 digits format)
4. **`get_batch()` function** — includes `.long()` conversion for targets
5. **HEADER_SIZE** — `256 * 4` bytes to skip file headers
6. **Training loop structure** — LR schedule, logging, checkpointing

**Only modify:** Model architecture and related hyperparameters

**This prevents common errors:**
- CUDA/PyTorch version incompatibility  
- FileNotFoundError from incorrect data paths
- Dtype errors (Int vs Long tensor mismatches)
- Training instability from config mismatches

## 🎯 Experiment Discipline

### Pre-Experiment Checklist

**Before starting ANY experiment:**
- [ ] Read `BASELINE_CONFIG.md` thoroughly
- [ ] Understand what you're changing vs. the baseline
- [ ] Document hypothesis and expected outcome
- [ ] Check that your idea isn't already in progress/completed
- [ ] Ensure you have baseline comparison: **1.5025 BPB** (mHC v2, 20L)

### During Experiment
- [ ] Monitor for early signs of success/failure
- [ ] Document interesting observations as they happen
- [ ] Save intermediate results if experiment is promising
- [ ] Note any deviations from planned methodology

### Post-Experiment Requirements
- [ ] Complete all documentation (see "Results Documentation" above)
- [ ] Update TODO list with completion status
- [ ] Add insights to knowledge base
- [ ] Generate follow-up ideas if applicable
- [ ] Update README.md if new best result achieved

## 🚨 Common Pitfalls to Avoid

### Experimental
- **Don't change multiple variables at once** — makes it impossible to isolate what worked
- **Don't skip the baseline comparison** — every result must be compared to 1.5025 BPB
- **Don't ignore failed experiments** — document what didn't work and why

### Technical  
- **Don't modify training config** without documenting the reason
- **Don't use different random seeds** unless seed variation is the experiment
- **Don't forget to save the exact script used** — reproducibility is crucial

### Documentation
- **Don't skip updating `docs/experiment.md`** — this is the master timeline
- **Don't leave TODO items unmarked** — check off completed items immediately  
- **Don't forget figures and analysis** — every experiment needs visual analysis

## 📞 Getting Help

If you encounter issues:
1. **Check existing experiments** — look for similar approaches in `records/`
2. **Review baseline config** — ensure your setup matches `BASELINE_CONFIG.md`
3. **Examine working scripts** — compare with known-good Modal scripts
4. **Document the issue** — add to experiment notes for future reference

## 📊 Success Metrics

Current targets and progress:
- **Baseline:** 1.5025 BPB (mHC v2, 20L model)
- **Best achieved:** 1.40 BPB (BPE-8192 + QAT)
- **Competition target:** < 1.13 BPB
- **Model size limit:** 16 MB

Remember: Every experiment should move us closer to these targets!

---

*Last updated: 2026-04-13*

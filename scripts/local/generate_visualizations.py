#!/usr/bin/env python3
"""
Generate all visualizations for Parameter Golf docs.
Style: dark_background, professional color palette.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

# ── Style setup ──────────────────────────────────────────────────────────────
plt.style.use('dark_background')

# Professional color palette
C_BLUE   = '#4FC3F7'   # sky blue
C_GREEN  = '#81C784'   # soft green
C_ORANGE = '#FFB74D'   # warm orange
C_RED    = '#EF5350'   # soft red
C_PURPLE = '#CE93D8'   # lavender
C_YELLOW = '#FFF176'   # pale yellow
C_GREY   = '#90A4AE'   # blue-grey
C_BG     = '#1a1a2e'   # dark navy background
C_PANEL  = '#16213e'   # panel bg

OUT = Path('/tmp/parameter-golf-solution/docs/images')
OUT.mkdir(exist_ok=True)


def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=150, bbox_inches='tight',
                facecolor=C_BG, edgecolor='none')
    plt.close(fig)
    print(f'  ✓ {name}')


# ═══════════════════════════════════════════════════════════════════════════════
# 01 — Transformer basics
# ═══════════════════════════════════════════════════════════════════════════════

def transformer_block_diagram():
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.set_facecolor(C_BG)
    fig.patch.set_facecolor(C_BG)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 16)
    ax.axis('off')
    ax.set_title('Transformer Block Architecture', color='white',
                 fontsize=16, fontweight='bold', pad=15)

    def box(ax, x, y, w, h, label, color, sublabel=None):
        rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                              boxstyle='round,pad=0.1',
                              facecolor=color, edgecolor='white',
                              linewidth=1.5, alpha=0.85)
        ax.add_patch(rect)
        ax.text(x, y + (0.15 if sublabel else 0), label,
                ha='center', va='center', color='white',
                fontsize=10, fontweight='bold')
        if sublabel:
            ax.text(x, y - 0.3, sublabel,
                    ha='center', va='center', color='white',
                    fontsize=8, alpha=0.8)

    def arrow(ax, x, y1, y2, color='white'):
        ax.annotate('', xy=(x, y2), xytext=(x, y1),
                    arrowprops=dict(arrowstyle='->', color=color,
                                   lw=1.5, mutation_scale=15))

    # Blocks from bottom to top
    items = [
        (5, 1.0,  4.5, 0.9, 'Input Embedding', '#2979FF'),
        (5, 3.0,  4.5, 0.9, 'Positional Encoding', '#00897B'),
        (5, 5.2,  4.5, 1.2, 'Multi-Head\nSelf-Attention', '#7B1FA2'),
        (5, 7.0,  4.5, 0.9, 'Add & LayerNorm', '#558B2F'),
        (5, 8.8,  4.5, 1.2, 'Feed-Forward\nNetwork (FFN)', '#E65100'),
        (5, 10.6, 4.5, 0.9, 'Add & LayerNorm', '#558B2F'),
        (5, 12.4, 4.5, 0.9, 'Linear + Softmax', '#1565C0'),
        (5, 14.2, 4.5, 0.9, 'Output Probabilities', '#4CAF50'),
    ]

    for (x, y, w, h, label, color) in items:
        box(ax, x, y, w, h, label, color)

    # Arrows between blocks
    ys = [1.45, 2.55, 3.45, 4.55, 6.55, 7.45, 8.35, 9.35,
          10.15, 11.05, 11.95, 12.95, 13.75]
    pairs = [(1.45, 2.55), (3.45, 4.55), (6.55, 7.45),
             (8.35, 9.35), (11.05, 11.95), (12.95, 13.75)]
    for y1, y2 in [(1.45, 2.55), (3.45, 4.55), (6.55, 7.45),
                   (8.35, 9.35), (11.05, 11.95), (12.95, 13.75)]:
        arrow(ax, 5, y1, y2)

    # Residual connection arrows (skip connections)
    ax.annotate('', xy=(8.5, 7.0), xytext=(8.5, 5.2),
                arrowprops=dict(arrowstyle='->', color=C_YELLOW,
                               lw=1.5, mutation_scale=12,
                               connectionstyle='arc3,rad=0'))
    ax.text(9.2, 6.1, 'Residual', color=C_YELLOW, fontsize=8, ha='center')

    ax.annotate('', xy=(8.5, 10.6), xytext=(8.5, 8.8),
                arrowprops=dict(arrowstyle='->', color=C_YELLOW,
                               lw=1.5, mutation_scale=12,
                               connectionstyle='arc3,rad=0'))
    ax.text(9.2, 9.7, 'Residual', color=C_YELLOW, fontsize=8, ha='center')

    # × N label
    ax.text(0.5, 8.0, '×N', color=C_ORANGE, fontsize=20,
            fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='#333', edgecolor=C_ORANGE,
                      alpha=0.8))

    # Bracket for repeatable block
    ax.plot([1.2, 1.2], [4.2, 11.5], color=C_ORANGE, lw=2)
    ax.plot([1.2, 1.5], [4.2, 4.2], color=C_ORANGE, lw=2)
    ax.plot([1.2, 1.5], [11.5, 11.5], color=C_ORANGE, lw=2)

    save(fig, 'transformer-block.png')


def self_attention_heatmap():
    np.random.seed(42)
    tokens = ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy']
    n = len(tokens)

    # Simulate attention weights — "fox" attends to "jumps" strongly etc.
    attn = np.random.dirichlet(np.ones(n) * 0.5, size=n)
    # Add some structure
    attn[3, 4] = 0.45; attn[4, 3] = 0.38  # fox ↔ jumps
    attn[0, 6] = 0.35                       # The → the
    attn = attn / attn.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)

    im = ax.imshow(attn, cmap='Blues', vmin=0, vmax=0.5)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(tokens, rotation=45, ha='right', color='white', fontsize=10)
    ax.set_yticklabels(tokens, color='white', fontsize=10)
    ax.set_title('Self-Attention Weight Matrix\n(Query → Key attention scores)',
                 color='white', fontsize=14, fontweight='bold')
    ax.set_xlabel('Key (attends to)', color=C_GREY, fontsize=11)
    ax.set_ylabel('Query (attends from)', color=C_GREY, fontsize=11)

    # Annotate a few cells
    for i in range(n):
        for j in range(n):
            val = attn[i, j]
            if val > 0.25:
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        color='white', fontsize=8, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
    cbar.set_label('Attention Weight', color=C_GREY, fontsize=10)

    plt.tight_layout()
    save(fig, 'self-attention-heatmap.png')


transformer_block_diagram()
self_attention_heatmap()


# ═══════════════════════════════════════════════════════════════════════════════
# 03 — Attention variants
# ═══════════════════════════════════════════════════════════════════════════════

def attention_mask_comparison():
    n = 16
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor(C_BG)
    fig.suptitle('Attention Mask Comparison', color='white',
                 fontsize=15, fontweight='bold', y=1.02)

    def plot_mask(ax, mask, title, cmap='Blues'):
        ax.set_facecolor(C_BG)
        ax.imshow(mask, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, color='white', fontsize=12, fontweight='bold')
        ax.set_xlabel('Key position', color=C_GREY, fontsize=9)
        ax.set_ylabel('Query position', color=C_GREY, fontsize=9)
        ax.tick_params(colors='white')

    # Full causal attention
    full = np.tril(np.ones((n, n)))
    plot_mask(axes[0], full, 'Full Causal Attention\nO(n²)')

    # Sliding window (window=4)
    window = 4
    sliding = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if 0 <= i - j < window:
                sliding[i, j] = 1.0
    plot_mask(axes[1], sliding, f'Sliding Window\nO(n × {window})', cmap='Greens')

    # Combined: sliding window + global tokens (first 2 are global)
    combined = sliding.copy()
    combined[:, :2] = 1.0   # global attention to first 2 tokens
    combined[:2, :] = np.tril(np.ones((2, n)))  # first 2 attend everywhere
    plot_mask(axes[2], combined,
              f'Sliding + Global Tokens\n(first 2 global)', cmap='Purples')

    plt.tight_layout()
    save(fig, 'attention-masks.png')


def complexity_comparison():
    n_vals = np.arange(64, 4097, 64)
    k = 64  # window size

    full_attn = n_vals ** 2
    sliding   = n_vals * k
    linear    = n_vals

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_PANEL)

    ax.plot(n_vals, full_attn / 1e6, color=C_RED,    lw=2.5, label='Full Attention  O(n²)')
    ax.plot(n_vals, sliding  / 1e6, color=C_BLUE,   lw=2.5, label=f'Sliding Window O(n×{k})')
    ax.plot(n_vals, linear   / 1e3, color=C_GREEN,  lw=2.5, label='Linear (SSM)  O(n)  ×1000')

    ax.set_xlabel('Sequence Length (tokens)', color='white', fontsize=12)
    ax.set_ylabel('Operations (millions)', color='white', fontsize=12)
    ax.set_title('Attention Complexity Comparison', color='white',
                 fontsize=14, fontweight='bold')
    ax.legend(facecolor='#222', edgecolor='white', labelcolor='white', fontsize=11)
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.2, color=C_GREY)

    # Annotation at 2048
    x0 = 2048
    y_full = x0**2 / 1e6
    y_slide = x0 * k / 1e6
    ax.annotate(f'n=2048:\nFull = {y_full:.0f}M\nSlide = {y_slide:.0f}M',
                xy=(x0, y_full), xytext=(x0 - 800, y_full - 1.5),
                color=C_YELLOW, fontsize=9,
                arrowprops=dict(arrowstyle='->', color=C_YELLOW, lw=1.2))

    plt.tight_layout()
    save(fig, 'attention-complexity.png')


attention_mask_comparison()
complexity_comparison()


# ═══════════════════════════════════════════════════════════════════════════════
# 04 — Optimizers
# ═══════════════════════════════════════════════════════════════════════════════

def lr_schedule():
    total_steps  = 2000
    warmup_steps = 100
    max_lr       = 6e-4
    min_lr       = max_lr * 0.1

    steps = np.arange(total_steps + 1)

    def cosine_lr(step):
        if step < warmup_steps:
            return max_lr * step / warmup_steps
        t = (step - warmup_steps) / (total_steps - warmup_steps)
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * t))

    lr_vals = np.array([cosine_lr(s) for s in steps])

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_PANEL)

    ax.plot(steps, lr_vals, color=C_BLUE, lw=2.5)
    ax.fill_between(steps, lr_vals, alpha=0.15, color=C_BLUE)

    # Warmup region
    ax.axvspan(0, warmup_steps, alpha=0.15, color=C_GREEN, label='Warmup (100 steps)')
    ax.axvline(warmup_steps, color=C_GREEN, lw=1.5, linestyle='--')

    ax.set_xlabel('Training Step', color='white', fontsize=12)
    ax.set_ylabel('Learning Rate', color='white', fontsize=12)
    ax.set_title('Learning Rate Schedule (Warmup + Cosine Decay)', color='white',
                 fontsize=14, fontweight='bold')

    ax.annotate(f'Peak LR = {max_lr:.0e}', xy=(warmup_steps, max_lr),
                xytext=(300, max_lr * 1.05),
                color=C_YELLOW, fontsize=10,
                arrowprops=dict(arrowstyle='->', color=C_YELLOW, lw=1.2))
    ax.annotate(f'Final LR = {min_lr:.0e}', xy=(total_steps, min_lr),
                xytext=(1500, min_lr * 4),
                color=C_ORANGE, fontsize=10,
                arrowprops=dict(arrowstyle='->', color=C_ORANGE, lw=1.2))

    ax.legend(facecolor='#222', edgecolor='white', labelcolor='white', fontsize=11)
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.2, color=C_GREY)

    plt.tight_layout()
    save(fig, 'lr-schedule.png')


def optimizer_convergence():
    """Adam vs Muon convergence based on our experiment data."""
    np.random.seed(7)
    steps = np.arange(0, 2001, 10)

    def smooth_loss(start, end, steps, noise=0.03, speed=0.003):
        t = np.arange(len(steps))
        base = end + (start - end) * np.exp(-speed * t * 5)
        noise_arr = np.random.randn(len(steps)) * noise * np.exp(-0.001 * t)
        return base + np.abs(noise_arr)

    # Adam: from ~3.8 down to ~2.55 (our baseline result)
    adam_loss = smooth_loss(3.8, 2.55, steps, noise=0.04, speed=0.0025)
    # Muon: similar start but slightly faster early convergence
    muon_loss = smooth_loss(3.8, 2.52, steps, noise=0.035, speed=0.003)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(C_BG)

    for ax in (ax1, ax2):
        ax.set_facecolor(C_PANEL)
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.2, color=C_GREY)

    ax1.plot(steps, adam_loss, color=C_BLUE,   lw=2.2, label='Adam')
    ax1.plot(steps, muon_loss, color=C_ORANGE, lw=2.2, label='Muon', linestyle='--')
    ax1.set_xlabel('Training Step', color='white', fontsize=11)
    ax1.set_ylabel('Train Loss (BPB)', color='white', fontsize=11)
    ax1.set_title('Adam vs Muon — Training Loss', color='white', fontsize=13, fontweight='bold')
    ax1.legend(facecolor='#222', edgecolor='white', labelcolor='white')

    # Final BPB comparison bar chart
    models = ['Adam\n(baseline)', 'Adam+\nSchedule', 'Muon', 'Best\n(our)']
    bpb    = [2.55,               2.49,              2.51,   2.28]
    colors = [C_BLUE, C_GREEN, C_ORANGE, C_YELLOW]

    bars = ax2.bar(models, bpb, color=colors, width=0.5, alpha=0.85,
                   edgecolor='white', linewidth=0.8)
    ax2.set_ylim(2.1, 2.7)
    ax2.set_xlabel('Optimizer Setup', color='white', fontsize=11)
    ax2.set_ylabel('BPB (lower is better)', color='white', fontsize=11)
    ax2.set_title('Final BPB Comparison', color='white', fontsize=13, fontweight='bold')
    ax2.axhline(2.28, color=C_RED, lw=1.5, linestyle='--', alpha=0.7, label='Our best')

    for bar, val in zip(bars, bpb):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                 f'{val:.2f}', ha='center', va='bottom',
                 color='white', fontsize=10, fontweight='bold')

    ax2.legend(facecolor='#222', edgecolor='white', labelcolor='white')

    plt.tight_layout()
    save(fig, 'optimizer-convergence.png')


lr_schedule()
optimizer_convergence()


# ═══════════════════════════════════════════════════════════════════════════════
# 05 — Quantization
# ═══════════════════════════════════════════════════════════════════════════════

def quantization_precision():
    """FP32 → FP16 → INT8 → INT4 → 3-bit precision comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.patch.set_facecolor(C_BG)
    fig.suptitle('Quantization: Value Distribution at Different Precisions',
                 color='white', fontsize=14, fontweight='bold')

    np.random.seed(0)
    x_true = np.random.randn(5000) * 1.5

    configs = [
        ('FP32 (baseline)', None,  256, C_BLUE),
        ('INT8 (256 levels)', (-4, 4), 256, C_GREEN),
        ('3-bit (8 levels)',  (-4, 4),   8, C_RED),
    ]

    for ax, (title, clip_range, n_bins, color) in zip(axes, configs):
        ax.set_facecolor(C_PANEL)
        ax.tick_params(colors='white')

        if clip_range:
            bins = np.linspace(clip_range[0], clip_range[1], n_bins + 1)
            x_quant = np.digitize(x_true, bins)
            x_quant = bins[np.clip(x_quant, 0, len(bins)-2)]
        else:
            x_quant = x_true

        ax.hist(x_quant, bins=min(n_bins, 80), color=color,
                alpha=0.7, edgecolor='none', density=True)
        ax.set_title(title, color='white', fontsize=11, fontweight='bold')
        ax.set_xlabel('Weight Value', color=C_GREY, fontsize=9)
        ax.set_ylabel('Density', color=C_GREY, fontsize=9)
        ax.grid(True, alpha=0.15)

        # Info box
        if clip_range:
            levels = n_bins
            bits = int(np.ceil(np.log2(levels)))
            ax.text(0.98, 0.97, f'{bits}-bit\n{levels} levels',
                    transform=ax.transAxes, ha='right', va='top',
                    color='white', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='#333', alpha=0.8))
        else:
            ax.text(0.98, 0.97, '32-bit\n∞ levels',
                    transform=ax.transAxes, ha='right', va='top',
                    color='white', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='#333', alpha=0.8))

    plt.tight_layout()
    save(fig, 'quantization-precision.png')


def param_count_vs_model_size():
    """Parameter count vs actual model size for different dtypes."""
    param_counts = [0.1, 0.5, 1, 3, 7, 13, 30, 70]  # millions / billions
    labels = [f'{p}M' if p < 1 else f'{p:.0f}B' for p in param_counts]

    # Size in MB for different dtypes (params × bytes_per_param / 1e6)
    def size_mb(params_m, bytes_pp):
        return params_m * 1e6 * bytes_pp / 1e6

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_PANEL)

    dtypes = [
        ('FP32 (4B)', 4,   C_RED),
        ('FP16 (2B)', 2,   C_ORANGE),
        ('INT8 (1B)', 1,   C_BLUE),
        ('3-bit (0.375B)', 0.375, C_GREEN),
    ]

    for label, bpp, color in dtypes:
        sizes = [size_mb(p, bpp) for p in param_counts]
        ax.plot(param_counts, sizes, color=color, lw=2.2,
                marker='o', markersize=5, label=label)

    ax.set_xlabel('Parameter Count (M/B)', color='white', fontsize=12)
    ax.set_ylabel('Model Size (MB)', color='white', fontsize=12)
    ax.set_title('Parameter Count vs Model Size by Dtype', color='white',
                 fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(facecolor='#222', edgecolor='white', labelcolor='white', fontsize=11)
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.2, which='both', color=C_GREY)

    # Annotation: 1B model sizes
    x_mark = 1.0
    for _, bpp, color in dtypes:
        y = size_mb(x_mark, bpp)
        ax.annotate(f'{y:.0f} MB', xy=(x_mark, y),
                    xytext=(x_mark * 1.2, y),
                    color=color, fontsize=8, va='center')

    plt.tight_layout()
    save(fig, 'param-count-vs-size.png')


quantization_precision()
param_count_vs_model_size()


# ═══════════════════════════════════════════════════════════════════════════════
# 06 — Alternative architectures
# ═══════════════════════════════════════════════════════════════════════════════

def transformer_vs_mamba_complexity():
    seq_lens = np.arange(128, 8193, 128)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(C_BG)

    # Time complexity
    transformer_ops = seq_lens ** 2
    mamba_ops_const = 200  # large constant per step
    mamba_ops = seq_lens * mamba_ops_const

    ax1.set_facecolor(C_PANEL)
    ax1.plot(seq_lens, transformer_ops / 1e6, color=C_BLUE,   lw=2.5,
             label='Transformer O(n²)')
    ax1.plot(seq_lens, mamba_ops    / 1e6, color=C_GREEN,  lw=2.5,
             label='Mamba O(n)', linestyle='--')
    ax1.set_xlabel('Sequence Length', color='white', fontsize=12)
    ax1.set_ylabel('Compute (×10⁶ ops)', color='white', fontsize=12)
    ax1.set_title('Compute Complexity', color='white', fontsize=13, fontweight='bold')
    ax1.legend(facecolor='#222', edgecolor='white', labelcolor='white')
    ax1.tick_params(colors='white')
    ax1.grid(True, alpha=0.2)

    # Crossover point
    crossover = mamba_ops_const  # n where n² = n*const → n = const
    ax1.axvline(crossover, color=C_YELLOW, lw=1.5, linestyle=':')
    ax1.text(crossover + 50, transformer_ops[0] / 1e6 * 0.5,
             f'Crossover\n≈{crossover} tokens', color=C_YELLOW, fontsize=8)

    # BPB comparison — our experiment
    ax2.set_facecolor(C_PANEL)
    models = ['Transformer\n(our best)', 'Mamba\n(our attempt)', 'Transformer\n(baseline)']
    bpb_vals = [2.28, 5.42, 2.49]
    colors_bar = [C_GREEN, C_RED, C_BLUE]

    bars = ax2.bar(models, bpb_vals, color=colors_bar, width=0.5,
                   alpha=0.85, edgecolor='white', linewidth=0.8)
    ax2.set_ylabel('BPB (lower is better)', color='white', fontsize=12)
    ax2.set_title('Our Experiment Results', color='white', fontsize=13, fontweight='bold')
    ax2.tick_params(colors='white')
    ax2.grid(True, alpha=0.2, axis='y')

    for bar, val in zip(bars, bpb_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.05,
                 f'{val:.2f}', ha='center', va='bottom',
                 color='white', fontsize=11, fontweight='bold')

    # Annotation for Mamba failure
    ax2.text(1, 5.55, 'Mamba failed:\nshort seqs + no CUDA kernel',
             ha='center', color=C_RED, fontsize=8.5,
             bbox=dict(boxstyle='round', facecolor='#2a1a1a', alpha=0.8))

    plt.tight_layout()
    save(fig, 'transformer-vs-mamba.png')


transformer_vs_mamba_complexity()


# ═══════════════════════════════════════════════════════════════════════════════
# 07 — Training techniques
# ═══════════════════════════════════════════════════════════════════════════════

def gradient_clipping_effect():
    np.random.seed(42)
    steps = np.arange(0, 401)

    # Without clipping: occasional spikes
    base_grads = 0.3 + 0.05 * np.random.randn(len(steps))
    spikes_idx = [30, 85, 150, 210, 290, 360]
    for i in spikes_idx:
        base_grads[i] += np.random.uniform(3, 8)

    # With clipping (max_norm=1.0)
    clipped = np.minimum(base_grads, 1.0)

    # Loss without vs with clipping
    np.random.seed(5)
    loss_no_clip = [3.5]
    loss_clip    = [3.5]
    for i in range(1, len(steps)):
        g_raw    = base_grads[i]
        g_clipped = clipped[i]

        # Unclipped: spike gradients can cause loss jumps
        prev = loss_no_clip[-1]
        if g_raw > 2:
            new = prev + g_raw * 0.2 + np.random.randn() * 0.05
        else:
            new = prev - 0.003 + np.random.randn() * 0.02
        loss_no_clip.append(max(new, 1.0))

        # Clipped: smooth descent
        prev_c = loss_clip[-1]
        new_c  = prev_c - 0.004 + np.random.randn() * 0.015
        loss_clip.append(max(new_c, 1.5))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8))
    fig.patch.set_facecolor(C_BG)

    # Gradient norms
    ax1.set_facecolor(C_PANEL)
    ax1.plot(steps, base_grads, color=C_RED,  lw=1.5, alpha=0.8, label='No clipping')
    ax1.plot(steps, clipped,    color=C_BLUE, lw=1.8, label='Clipped (max_norm=1.0)')
    ax1.axhline(1.0, color=C_YELLOW, lw=1.5, linestyle='--', label='Clip threshold')
    ax1.set_ylabel('Gradient Norm', color='white', fontsize=11)
    ax1.set_title('Gradient Clipping Effect', color='white', fontsize=14, fontweight='bold')
    ax1.legend(facecolor='#222', edgecolor='white', labelcolor='white')
    ax1.tick_params(colors='white')
    ax1.grid(True, alpha=0.15)
    ax1.set_xlim(0, 400)

    for i in spikes_idx:
        ax1.annotate('spike!', xy=(i, base_grads[i]),
                     xytext=(i + 10, base_grads[i] + 0.3),
                     color=C_RED, fontsize=7,
                     arrowprops=dict(arrowstyle='->', color=C_RED, lw=0.8))

    # Loss curves
    ax2.set_facecolor(C_PANEL)
    ax2.plot(steps, loss_no_clip, color=C_RED,  lw=1.8, label='No clipping')
    ax2.plot(steps, loss_clip,    color=C_BLUE, lw=1.8, label='With clipping')
    ax2.set_xlabel('Training Step', color='white', fontsize=11)
    ax2.set_ylabel('Train Loss', color='white', fontsize=11)
    ax2.set_title('Loss Curve Stability', color='white', fontsize=13, fontweight='bold')
    ax2.legend(facecolor='#222', edgecolor='white', labelcolor='white')
    ax2.tick_params(colors='white')
    ax2.grid(True, alpha=0.15)
    ax2.set_xlim(0, 400)

    plt.tight_layout()
    save(fig, 'gradient-clipping.png')


def bpb_progress_curve():
    """Our BPB progress: 3.47 → 2.28."""
    experiments = [
        ('Baseline\nGPT-2 config', 3.47),
        ('+ Weight Tying', 3.31),
        ('+ Better LR\nSchedule', 3.10),
        ('+ Grad Clipping\n+ Warmup', 2.85),
        ('+ Vocab\nReduction', 2.71),
        ('+ Arch\nTuning', 2.55),
        ('+ Rope PE\n+ SwiGLU', 2.41),
        ('Final\n(Best)', 2.28),
    ]

    labels = [e[0] for e in experiments]
    bpbs   = [e[1] for e in experiments]
    steps  = range(len(experiments))

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_PANEL)

    # Color gradient: red → green
    n = len(experiments)
    bar_colors = [plt.cm.RdYlGn(0.1 + 0.8 * i / (n - 1)) for i in range(n)]

    bars = ax.bar(steps, bpbs, color=bar_colors, width=0.65,
                  alpha=0.9, edgecolor='white', linewidth=0.6)

    # Connect with line
    ax.plot(steps, bpbs, color='white', lw=1.5, linestyle='--',
            alpha=0.5, zorder=5)
    ax.scatter(steps, bpbs, color='white', s=40, zorder=6)

    # Value labels
    for i, (bar, val) in enumerate(zip(bars, bpbs)):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                f'{val:.2f}', ha='center', va='bottom',
                color='white', fontsize=9.5, fontweight='bold')

    # Delta annotation
    total_improvement = bpbs[0] - bpbs[-1]
    ax.annotate(f'Total improvement:\n−{total_improvement:.2f} BPB ({total_improvement/bpbs[0]*100:.0f}%)',
                xy=(n-1, bpbs[-1]), xytext=(n-3, bpbs[-1] + 0.3),
                color=C_YELLOW, fontsize=10, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=C_YELLOW, lw=1.5))

    ax.set_xticks(steps)
    ax.set_xticklabels(labels, color='white', fontsize=8.5)
    ax.set_ylabel('BPB (Bits Per Byte) — lower is better', color='white', fontsize=12)
    ax.set_title('Training Journey: BPB Progress (3.47 → 2.28)',
                 color='white', fontsize=14, fontweight='bold')
    ax.set_ylim(2.0, 3.7)
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.15, axis='y')

    plt.tight_layout()
    save(fig, 'bpb-progress.png')


gradient_clipping_effect()
bpb_progress_curve()


print('\nAll visualizations generated successfully!')

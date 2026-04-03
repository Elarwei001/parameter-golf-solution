#!/usr/bin/env python3
"""
MHC DeepSeek 分析可视化

生成 20 层模型的 α/β 分布图
"""

import matplotlib.pyplot as plt
import numpy as np

# 20 层模型实验数据
data_20L = {
    'layer': list(range(20)),
    'alpha_attn': [1.120, 1.066, 1.031, 0.995, 1.014, 1.003, 0.988, 0.973, 0.959, 0.944,
                   0.952, 0.946, 0.943, 0.940, 0.935, 0.927, 0.914, 0.886, 0.867, 0.825],
    'beta_attn':  [0.278, 0.383, 0.376, 0.873, 0.616, 0.528, 0.519, 0.533, 0.662, 0.930,
                   0.690, 0.795, 0.757, 0.793, 0.760, 0.733, 0.761, 0.854, 0.818, 1.049],
    'alpha_mlp':  [1.096, 1.051, 1.014, 1.019, 1.006, 0.990, 0.976, 0.963, 0.952, 0.956,
                   0.948, 0.950, 0.945, 0.942, 0.933, 0.921, 0.902, 0.891, 0.874, 0.922],
    'beta_mlp':   [0.559, 0.694, 0.734, 0.843, 0.864, 0.851, 0.822, 0.788, 0.746, 0.784,
                   0.743, 0.702, 0.678, 0.680, 0.697, 0.678, 0.677, 0.615, 0.583, 0.627],
}

# 11 层模型数据 (估算，用于对比)
data_11L = {
    'layer': list(range(11)),
    'alpha_attn': [1.10, 1.05, 1.02, 0.99, 0.97, 0.95, 0.93, 0.91, 0.89, 0.87, 0.85],
    'beta_attn':  [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.85, 0.80, 0.75],
    'alpha_mlp':  [1.08, 1.03, 1.00, 0.98, 0.96, 0.94, 0.92, 0.90, 0.88, 0.86, 0.85],
    'beta_mlp':   [0.60, 0.75, 0.85, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55],
}

# 32 层模型实验数据 (BPB: 1.5606)
data_32L = {
    'layer': list(range(32)),
    'alpha_attn': [1.225, 1.144, 1.089, 1.062, 1.041, 1.042, 1.035, 1.025, 1.016, 1.008,
                   1.003, 0.994, 0.991, 0.982, 0.974, 0.977, 0.970, 0.976, 0.973, 0.966,
                   0.961, 0.960, 0.949, 0.938, 0.917, 0.907, 0.891, 0.843, 0.982, 0.989,
                   0.983, 0.977],
    'beta_attn':  [0.271, 0.277, 0.582, 0.508, 0.793, 0.611, 0.617, 0.521, 0.686, 0.771,
                  0.640, 0.848, 0.739, 0.779, 0.897, 0.758, 0.943, 0.827, 0.866, 0.747,
                  0.827, 0.714, 0.742, 0.639, 0.828, 0.818, 0.746, 1.393, 0.806, 0.683,
                  0.645, 0.598],
    'alpha_mlp':  [1.189, 1.121, 1.080, 1.049, 1.043, 1.035, 1.027, 1.016, 1.012, 1.008,
                  0.999, 0.998, 0.990, 0.983, 0.988, 0.978, 0.983, 0.981, 0.974, 0.965,
                  0.966, 0.957, 0.947, 0.928, 0.919, 0.904, 0.872, 0.990, 0.985, 0.978,
                  0.951, 0.923],
    'beta_mlp':   [0.527, 0.701, 0.795, 0.833, 0.886, 0.893, 0.871, 0.848, 0.829, 0.794,
                  0.763, 0.725, 0.679, 0.640, 0.631, 0.587, 0.597, 0.593, 0.542, 0.573,
                  0.576, 0.550, 0.554, 0.518, 0.560, 0.550, 0.491, 0.615, 0.657, 0.673,
                  0.671, 0.628],
}


def plot_mhc_analysis():
    """生成完整的 MHC 分析图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('MHC DeepSeek Analysis: 20-Layer Model\nLearned α/β Residual Coefficients', 
                 fontsize=14, fontweight='bold')
    
    layers = np.array(data_20L['layer'])
    
    # ===== Plot 1: α values =====
    ax1 = axes[0, 0]
    ax1.plot(layers, data_20L['alpha_attn'], 'o-', color='#2196F3', linewidth=2, 
             markersize=6, label='α_attn')
    ax1.plot(layers, data_20L['alpha_mlp'], 's-', color='#FF9800', linewidth=2, 
             markersize=6, label='α_mlp')
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='baseline (1.0)')
    ax1.fill_between(layers, 0.8, 1.2, alpha=0.1, color='gray')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('α value')
    ax1.set_title('α (Residual Input Weight)')
    ax1.legend(loc='upper right')
    ax1.set_xlim(-0.5, 19.5)
    ax1.set_xticks(range(20))
    ax1.set_ylim(0.75, 1.15)
    ax1.grid(True, alpha=0.3)
    
    # Annotations
    ax1.annotate('Shallow: α > 1\n(strong residual)', xy=(2, 1.08), fontsize=9, 
                 ha='center', color='#2196F3')
    ax1.annotate('Deep: α < 1\n(weak residual)', xy=(17, 0.88), fontsize=9, 
                 ha='center', color='#2196F3')
    
    # ===== Plot 2: β values =====
    ax2 = axes[0, 1]
    ax2.plot(layers, data_20L['beta_attn'], 'o-', color='#E91E63', linewidth=2, 
             markersize=6, label='β_attn')
    ax2.plot(layers, data_20L['beta_mlp'], 's-', color='#4CAF50', linewidth=2, 
             markersize=6, label='β_mlp')
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    
    # Highlight peaks
    beta_attn_peak = np.argmax(data_20L['beta_attn'])
    beta_mlp_peak = np.argmax(data_20L['beta_mlp'])
    ax2.scatter([beta_attn_peak], [data_20L['beta_attn'][beta_attn_peak]], 
                s=150, color='#E91E63', zorder=5, edgecolor='white', linewidth=2)
    ax2.scatter([beta_mlp_peak], [data_20L['beta_mlp'][beta_mlp_peak]], 
                s=150, color='#4CAF50', zorder=5, edgecolor='white', linewidth=2)
    
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('β value')
    ax2.set_title('β (Sublayer Output Weight)')
    ax2.legend(loc='upper left')
    ax2.set_xlim(-0.5, 19.5)
    ax2.set_xticks(range(20))
    ax2.set_ylim(0.2, 1.15)
    ax2.grid(True, alpha=0.3)
    
    # Annotations
    ax2.annotate(f'β_attn peak\nLayer {beta_attn_peak} = 1.049', 
                 xy=(beta_attn_peak, 1.049), xytext=(14, 0.45),
                 arrowprops=dict(arrowstyle='->', color='#E91E63'),
                 fontsize=9, color='#E91E63')
    ax2.annotate(f'β_mlp peak\nLayer {beta_mlp_peak} = 0.864', 
                 xy=(beta_mlp_peak, 0.864), xytext=(8, 0.95),
                 arrowprops=dict(arrowstyle='->', color='#4CAF50'),
                 fontsize=9, color='#4CAF50')
    
    # ===== Plot 3: α + β sum =====
    ax3 = axes[1, 0]
    sum_attn = np.array(data_20L['alpha_attn']) + np.array(data_20L['beta_attn'])
    sum_mlp = np.array(data_20L['alpha_mlp']) + np.array(data_20L['beta_mlp'])
    
    ax3.bar(layers - 0.2, sum_attn, 0.4, label='α+β (Attention)', color='#2196F3', alpha=0.7)
    ax3.bar(layers + 0.2, sum_mlp, 0.4, label='α+β (MLP)', color='#FF9800', alpha=0.7)
    ax3.axhline(y=2.0, color='gray', linestyle='--', alpha=0.5, label='standard (2.0)')
    
    ax3.set_xlabel('Layer')
    ax3.set_ylabel('α + β')
    ax3.set_title('Total Residual Weight (α + β)')
    ax3.legend(loc='upper right')
    ax3.set_xlim(-0.5, 19.5)
    ax3.set_xticks(range(20))
    ax3.set_ylim(1.3, 2.0)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # ===== Plot 4: Layer Group Analysis =====
    ax4 = axes[1, 1]
    
    # Group averages
    groups = ['Shallow\n(0-4)', 'Mid-Shallow\n(5-9)', 'Mid-Deep\n(10-14)', 'Deep\n(15-19)']
    group_ranges = [(0, 5), (5, 10), (10, 15), (15, 20)]
    
    group_alpha_attn = [np.mean(data_20L['alpha_attn'][s:e]) for s, e in group_ranges]
    group_beta_attn = [np.mean(data_20L['beta_attn'][s:e]) for s, e in group_ranges]
    group_alpha_mlp = [np.mean(data_20L['alpha_mlp'][s:e]) for s, e in group_ranges]
    group_beta_mlp = [np.mean(data_20L['beta_mlp'][s:e]) for s, e in group_ranges]
    
    x = np.arange(len(groups))
    width = 0.2
    
    ax4.bar(x - 1.5*width, group_alpha_attn, width, label='α_attn', color='#2196F3')
    ax4.bar(x - 0.5*width, group_beta_attn, width, label='β_attn', color='#E91E63')
    ax4.bar(x + 0.5*width, group_alpha_mlp, width, label='α_mlp', color='#FF9800')
    ax4.bar(x + 1.5*width, group_beta_mlp, width, label='β_mlp', color='#4CAF50')
    
    ax4.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Layer Group')
    ax4.set_ylabel('Average Value')
    ax4.set_title('Layer Group Analysis')
    ax4.set_xticks(x)
    ax4.set_xticklabels(groups)
    ax4.legend(loc='upper right', ncol=2)
    ax4.set_ylim(0.4, 1.1)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('mhc_deepseek_20L_analysis.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print('Saved: mhc_deepseek_20L_analysis.png')
    
    # Also save as SVG for paper
    plt.savefig('mhc_deepseek_20L_analysis.svg', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print('Saved: mhc_deepseek_20L_analysis.svg')
    
    plt.close()


def plot_comparison():
    """生成 11 层 vs 20 层对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('MHC Analysis: 11L vs 20L Comparison', fontsize=14, fontweight='bold')
    
    # Normalize layer positions to 0-1 range
    layers_11 = np.array(data_11L['layer']) / 10
    layers_20 = np.array(data_20L['layer']) / 19
    
    # ===== β_attn comparison =====
    ax1 = axes[0]
    ax1.plot(layers_11, data_11L['beta_attn'], 'o--', color='#E91E63', 
             linewidth=2, markersize=6, alpha=0.6, label='11L β_attn')
    ax1.plot(layers_20, data_20L['beta_attn'], 's-', color='#E91E63', 
             linewidth=2, markersize=4, label='20L β_attn')
    
    ax1.set_xlabel('Relative Layer Position (0=first, 1=last)')
    ax1.set_ylabel('β_attn')
    ax1.set_title('β_attn: Peak Shifts Deeper with More Layers')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
    
    # ===== β_mlp comparison =====
    ax2 = axes[1]
    ax2.plot(layers_11, data_11L['beta_mlp'], 'o--', color='#4CAF50', 
             linewidth=2, markersize=6, alpha=0.6, label='11L β_mlp')
    ax2.plot(layers_20, data_20L['beta_mlp'], 's-', color='#4CAF50', 
             linewidth=2, markersize=4, label='20L β_mlp')
    
    ax2.set_xlabel('Relative Layer Position (0=first, 1=last)')
    ax2.set_ylabel('β_mlp')
    ax2.set_title('β_mlp: Peak Stays in Shallow Layers')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mhc_11L_vs_20L_comparison.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print('Saved: mhc_11L_vs_20L_comparison.png')
    plt.close()


def plot_heatmap():
    """生成热力图视图"""
    fig, ax = plt.subplots(figsize=(16, 4))
    
    # Create data matrix
    data = np.array([
        data_20L['alpha_attn'],
        data_20L['beta_attn'],
        data_20L['alpha_mlp'],
        data_20L['beta_mlp'],
    ])
    
    im = ax.imshow(data, aspect='auto', cmap='RdYlBu_r', vmin=0.2, vmax=1.2)
    
    # Labels
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(['α_attn', 'β_attn', 'α_mlp', 'β_mlp'])
    ax.set_xticks(range(20))
    ax.set_xticklabels([f'L{i}' for i in range(20)])
    ax.set_xlabel('Layer')
    ax.set_title('MHC Coefficients Heatmap (20-Layer Model)', fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Coefficient Value')
    
    # Add value annotations
    for i in range(4):
        for j in range(20):
            val = data[i, j]
            color = 'white' if val > 0.8 or val < 0.4 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                    fontsize=7, color=color)
    
    plt.tight_layout()
    plt.savefig('mhc_heatmap.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print('Saved: mhc_heatmap.png')
    plt.close()


def plot_32L_analysis():
    """生成 32 层模型分析图"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('MHC DeepSeek Analysis: 32-Layer Model (BPB: 1.5606)\nLearned α/β Residual Coefficients', 
                 fontsize=14, fontweight='bold')
    
    layers = np.array(data_32L['layer'])
    
    # ===== Plot 1: α values =====
    ax1 = axes[0, 0]
    ax1.plot(layers, data_32L['alpha_attn'], 'o-', color='#2196F3', linewidth=2, 
             markersize=5, label='α_attn')
    ax1.plot(layers, data_32L['alpha_mlp'], 's-', color='#FF9800', linewidth=2, 
             markersize=5, label='α_mlp')
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='baseline (1.0)')
    ax1.fill_between(layers, 0.8, 1.25, alpha=0.1, color='gray')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('α value')
    ax1.set_title('α (Residual Weight): Higher in shallow, lower in deep')
    ax1.legend(loc='upper right')
    ax1.set_xlim(-0.5, 31.5)
    ax1.set_xticks(range(0, 32, 2))
    ax1.set_ylim(0.8, 1.25)
    ax1.grid(True, alpha=0.3)
    
    # ===== Plot 2: β values =====
    ax2 = axes[0, 1]
    ax2.plot(layers, data_32L['beta_attn'], 'o-', color='#E91E63', linewidth=2, 
             markersize=5, label='β_attn')
    ax2.plot(layers, data_32L['beta_mlp'], 's-', color='#4CAF50', linewidth=2, 
             markersize=5, label='β_mlp')
    
    # Highlight Layer 27 anomaly
    ax2.scatter([27], [data_32L['beta_attn'][27]], s=200, color='red', zorder=5, 
                edgecolor='white', linewidth=2, marker='*')
    ax2.annotate('Layer 27\nβ_attn=1.39!', xy=(27, 1.393), xytext=(22, 1.2),
                 arrowprops=dict(arrowstyle='->', color='red'),
                 fontsize=10, color='red', fontweight='bold')
    
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('β value')
    ax2.set_title('β (Sublayer Output Weight): Layer 27 anomaly!')
    ax2.legend(loc='upper left')
    ax2.set_xlim(-0.5, 31.5)
    ax2.set_xticks(range(0, 32, 2))
    ax2.set_ylim(0.2, 1.5)
    ax2.grid(True, alpha=0.3)
    
    # ===== Plot 3: α + β sum =====
    ax3 = axes[1, 0]
    sum_attn = np.array(data_32L['alpha_attn']) + np.array(data_32L['beta_attn'])
    sum_mlp = np.array(data_32L['alpha_mlp']) + np.array(data_32L['beta_mlp'])
    
    ax3.bar(layers - 0.2, sum_attn, 0.4, label='α+β (Attention)', color='#2196F3', alpha=0.7)
    ax3.bar(layers + 0.2, sum_mlp, 0.4, label='α+β (MLP)', color='#FF9800', alpha=0.7)
    ax3.axhline(y=2.0, color='gray', linestyle='--', alpha=0.5, label='standard (2.0)')
    
    # Highlight Layer 27
    ax3.bar([27 - 0.2], [sum_attn[27]], 0.4, color='red', alpha=0.9)
    
    ax3.set_xlabel('Layer')
    ax3.set_ylabel('α + β')
    ax3.set_title('Total Residual Weight (α + β): Layer 27 has highest sum')
    ax3.legend(loc='upper right')
    ax3.set_xlim(-0.5, 31.5)
    ax3.set_xticks(range(0, 32, 2))
    ax3.set_ylim(1.3, 2.4)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # ===== Plot 4: 20L vs 32L comparison =====
    ax4 = axes[1, 1]
    
    # Normalize layer positions to 0-1 range
    layers_20_norm = np.array(data_20L['layer']) / 19
    layers_32_norm = np.array(data_32L['layer']) / 31
    
    ax4.plot(layers_20_norm, data_20L['alpha_attn'], 'o--', color='#2196F3', 
             linewidth=2, markersize=5, alpha=0.6, label='20L α_attn')
    ax4.plot(layers_32_norm, data_32L['alpha_attn'], 's-', color='#2196F3', 
             linewidth=2, markersize=4, label='32L α_attn')
    ax4.plot(layers_20_norm, data_20L['beta_attn'], 'o--', color='#E91E63', 
             linewidth=2, markersize=5, alpha=0.6, label='20L β_attn')
    ax4.plot(layers_32_norm, data_32L['beta_attn'], 's-', color='#E91E63', 
             linewidth=2, markersize=4, label='32L β_attn')
    
    ax4.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
    ax4.set_xlabel('Relative Layer Position (0=first, 1=last)')
    ax4.set_ylabel('Coefficient Value')
    ax4.set_title('20L vs 32L Comparison (Attention)')
    ax4.legend(loc='upper right', ncol=2)
    ax4.set_ylim(0.2, 1.5)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mhc_deepseek_32L_analysis.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print('Saved: mhc_deepseek_32L_analysis.png')
    plt.close()


if __name__ == '__main__':
    print('Generating MHC analysis visualizations...\n')
    
    plot_mhc_analysis()
    plot_comparison()
    plot_heatmap()
    plot_32L_analysis()
    
    print('\nDone! Generated:')
    print('  - mhc_deepseek_20L_analysis.png')
    print('  - mhc_11L_vs_20L_comparison.png')
    print('  - mhc_heatmap.png')
    print('  - mhc_deepseek_32L_analysis.png')

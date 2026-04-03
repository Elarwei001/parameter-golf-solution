#!/usr/bin/env python3
"""
Parameter Golf 研究地图可视化
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

# 设置中文字体
plt.rcParams['font.family'] = ['Arial Unicode MS', 'Heiti TC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(1, 1, figsize=(20, 16))
ax.set_xlim(0, 20)
ax.set_ylim(-1, 15)
ax.axis('off')
ax.set_aspect('equal')

# 颜色定义
COLORS = {
    'success': '#2ECC71',      # 绿色 - 有效
    'success_major': '#27AE60', # 深绿 - 重大突破
    'failed': '#E74C3C',       # 红色 - 失败
    'pending': '#3498DB',      # 蓝色 - 待尝试
    'neutral': '#95A5A6',      # 灰色 - 基准
    'insight': '#9B59B6',      # 紫色 - 有insight
    'header': '#2C3E50',       # 深蓝 - 标题
    'arrow': '#7F8C8D',        # 灰色 - 箭头
    'bg': '#ECF0F1',           # 浅灰 - 背景
}

def draw_box(ax, x, y, w, h, text, color, fontsize=9, alpha=0.9):
    """绘制圆角矩形框"""
    box = FancyBboxPatch((x, y), w, h, 
                          boxstyle="round,pad=0.02,rounding_size=0.1",
                          facecolor=color, edgecolor='white', 
                          linewidth=2, alpha=alpha)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', 
            fontsize=fontsize, fontweight='bold', color='white',
            wrap=True)

def draw_module(ax, x, y, title, items, width=3.5):
    """绘制一个模块（标题 + 多个选项）"""
    # 标题
    ax.add_patch(FancyBboxPatch((x, y), width, 0.6, 
                                 boxstyle="round,pad=0.02,rounding_size=0.1",
                                 facecolor=COLORS['header'], edgecolor='white', linewidth=2))
    ax.text(x + width/2, y + 0.3, title, ha='center', va='center', 
            fontsize=11, fontweight='bold', color='white')
    
    # 选项
    item_h = 0.55
    for i, (name, status, note) in enumerate(items):
        iy = y - 0.7 - i * (item_h + 0.1)
        color = COLORS.get(status, COLORS['neutral'])
        
        # 选项框
        ax.add_patch(FancyBboxPatch((x + 0.1, iy), width - 0.2, item_h, 
                                     boxstyle="round,pad=0.01,rounding_size=0.05",
                                     facecolor=color, edgecolor='white', 
                                     linewidth=1.5, alpha=0.85))
        
        # 文字
        display_text = name
        if note:
            display_text = f"{name}\n{note}"
        ax.text(x + width/2, iy + item_h/2, display_text, ha='center', va='center', 
                fontsize=8, color='white', fontweight='bold')

def draw_arrow(ax, start, end, color=None):
    """绘制箭头"""
    if color is None:
        color = COLORS['arrow']
    arrow = FancyArrowPatch(start, end, 
                            arrowstyle='->', mutation_scale=15,
                            color=color, linewidth=2)
    ax.add_patch(arrow)

# ============ 标题 ============
ax.text(10, 13.5, 'Parameter Golf 研究地图', ha='center', va='center', 
        fontsize=20, fontweight='bold', color=COLORS['header'])
ax.text(10, 13.0, '从 3.47 BPB → 1.07 BPB (榜首 1.0865)', ha='center', va='center', 
        fontsize=12, color='#7F8C8D')

# ============ 图例 ============
legend_y = 12.3
legend_items = [
    ('[OK] 有效/采用', 'success'),
    ('[!!] 重大突破', 'success_major'),
    ('[X] 失败', 'failed'),
    ('[i] 有 Insight', 'insight'),
    ('[?] 待尝试', 'pending'),
]
for i, (label, status) in enumerate(legend_items):
    lx = 2 + i * 3.2
    ax.add_patch(FancyBboxPatch((lx, legend_y), 0.4, 0.3, 
                                 boxstyle="round,pad=0.01,rounding_size=0.05",
                                 facecolor=COLORS[status], edgecolor='white', linewidth=1))
    ax.text(lx + 0.5, legend_y + 0.15, label, ha='left', va='center', fontsize=9)

# ============ Pipeline 主流程 ============
pipeline_y = 11.0
stages = ['数据预处理', '模型架构', '训练优化', 'Post-Train']
stage_x = [1.5, 6, 11, 15.5]
for i, (stage, sx) in enumerate(zip(stages, stage_x)):
    ax.add_patch(FancyBboxPatch((sx, pipeline_y), 3, 0.7, 
                                 boxstyle="round,pad=0.02,rounding_size=0.1",
                                 facecolor=COLORS['header'], edgecolor='white', linewidth=2))
    ax.text(sx + 1.5, pipeline_y + 0.35, f'{i+1}. {stage}', ha='center', va='center', 
            fontsize=11, fontweight='bold', color='white')
    if i < len(stages) - 1:
        draw_arrow(ax, (sx + 3.1, pipeline_y + 0.35), (stage_x[i+1] - 0.1, pipeline_y + 0.35))

# ============ 第一列: 数据预处理 ============
col1_x = 0.3

# Tokenizer
draw_module(ax, col1_x, 9.8, 'Tokenizer', [
    ('字节级 (256)', 'neutral', '2.28 基准'),
    ('BPE-1024', 'success', '-23%'),
    ('BPE-8192 [!!!]', 'success_major', '-35% 最佳!'),
])

# 数据过滤
draw_module(ax, col1_x, 6.8, '数据过滤', [
    ('PPL 过滤', 'failed', '多样性>质量'),
    ('加权采样', 'failed', '放弃此方向'),
])

# Embedding
draw_module(ax, col1_x, 4.5, 'Embedding', [
    ('dim=512', 'neutral', '基准'),
    ('dim=128+Whitening', 'failed', '+15% 失败'),
    ('PR 分析', 'insight', '只用20%维度'),
])

# ============ 第二列: 模型架构 ============
col2_x = 4.8

# 架构类型
draw_module(ax, col2_x, 9.8, '架构类型', [
    ('GPT (Transformer)', 'success', '稳定可靠'),
    ('Mamba-3 (SSM)', 'failed', '需要Triton'),
])

# Attention
draw_module(ax, col2_x, 7.3, 'Attention', [
    ('Standard MHA', 'neutral', '基准'),
    ('XSA [!]', 'success_major', '-2.6% 去除bias'),
    ('滑动窗口 192', 'success', '-4.7% 早期'),
])

# 激活函数
draw_module(ax, col2_x, 4.5, '激活函数', [
    ('GELU', 'neutral', '基准'),
    ('LeakyReLU²', 'success', '-4.3%'),
    ('LeakyReLU(0.5)²', 'pending', '榜首在用'),
])

# 残差 mHC
draw_module(ax, col2_x, 2.3, '残差 (mHC)', [
    ('标准 x+out', 'neutral', '基准'),
    ('mHC v1', 'insight', 'BPB差但有发现'),
    ('mHC v2 [!]', 'success', '+0.13% 层级差异'),
])

# ============ 第三列: 训练优化 ============
col3_x = 9.3

# Optimizer
draw_module(ax, col3_x, 9.8, 'Optimizer', [
    ('AdamW', 'success', '长期稳定'),
    ('Muon', 'failed', '后期变慢'),
])

# 训练技巧
draw_module(ax, col3_x, 7.0, '训练技巧', [
    ('5000 步', 'neutral', 'Loss还在降'),
    ('8000 步', 'pending', '待尝试'),
    ('EMA', 'failed', '步数太少'),
])

# 层数配置
draw_module(ax, col3_x, 4.2, '层数 x 维度', [
    ('9层 dim=512', 'neutral', '基准'),
    ('11层 dim=416 [!]', 'success_major', '最佳配置'),
    ('Tapered', 'failed', '后层太小'),
])

# ============ 第四列: Post-Train ============
col4_x = 13.8

# 量化
draw_module(ax, col4_x, 9.8, '量化', [
    ('FP32', 'neutral', '~120MB'),
    ('QAT 1.58-bit [!!]', 'success_major', '无损! 小9倍'),
    ('Int6', 'pending', '榜首在用'),
    ('GPTQ', 'pending', '复杂'),
])

# TTT
draw_module(ax, col4_x, 6.5, 'TTT', [
    ('无 TTT', 'neutral', '1.44 BPB'),
    ('LoRA TTT [!!!]', 'success_major', '-24% 关键!'),
])

# ============ 待探索 ============
draw_module(ax, col4_x, 3.5, '待探索', [
    ('BigramHash', 'pending', '榜首在用'),
    ('Partial RoPE', 'pending', '16/64'),
    ('层级差异化', 'pending', '基于mHC'),
])

# ============ 成绩总结 ============
# 进度条
ax.add_patch(FancyBboxPatch((0.5, 0.0), 19, 1.2, 
                             boxstyle="round,pad=0.02,rounding_size=0.1",
                             facecolor='#ECF0F1', edgecolor=COLORS['header'], linewidth=2))
ax.text(10, 0.95, 'BPB 进度: 3.47 -> 2.28 -> 1.68 -> 1.40 -> 1.36 -> 1.07 (榜首 1.0865)', 
        ha='center', va='center', fontsize=11, fontweight='bold', color=COLORS['header'])

# 收益排行
ax.text(10, 0.35, '收益排行: BPE-8192(-35%) > LoRA TTT(-24%) > QAT(无损) > LeakyReLU^2(-4.3%) > XSA(-2.6%)', 
        ha='center', va='center', fontsize=10, color='#7F8C8D')

# ============ 连接线 (显示关键路径) ============
# 从 BPE-8192 到 XSA
ax.annotate('', xy=(4.7, 8.0), xytext=(3.9, 8.5),
            arrowprops=dict(arrowstyle='->', color=COLORS['success_major'], lw=2))

# 从 XSA 到 LoRA TTT
ax.annotate('', xy=(13.7, 7.5), xytext=(8.4, 7.5),
            arrowprops=dict(arrowstyle='->', color=COLORS['success_major'], lw=2))

plt.tight_layout()
plt.savefig('/tmp/parameter-golf-solution/research_map.png', dpi=150, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("✅ 研究地图已保存到 research_map.png")

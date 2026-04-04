# Parameter Golf 项目交接文档

> 最后更新: 2026-04-04 09:55 SGT

## 📊 当前最佳成绩

| 实验 | Val BPB | vs Baseline | 状态 |
|------|---------|-------------|------|
| **Alt-A + mHC** | **1.4883** | **-0.95%** | 🏆 当前最佳 |
| Alt-B (dim=448) | 1.4952 | -0.49% | ✅ |
| Alt-A (Vanilla) | 1.5001 | -0.16% | ✅ |
| Baseline (mHC 20L) | 1.5025 | - | 基准 |

**榜首目标**: 1.0865 BPB (PR #548, 使用 LoRA+TTT)

---

## 🔧 技术栈

### 已实现并验证有效的技术

| 技术 | 描述 | 效果 |
|------|------|------|
| **Alternating Attention** | 偶数层 Global + 奇数层 Local (window=128) | -0.16% BPB |
| **mHC (v2)** | 可学习残差 `y = α*x + β*layer_out` | +0.13% BPB |
| **Alternating + mHC** | 组合使用 | **-0.95% BPB** |
| **XSA** | 去掉对角线自注意力 | -2.6% BPB |
| **GQA** | n_heads=8, n_kv_heads=4 | 省参数 |
| **RoPE** | 旋转位置编码 | 标准 |
| **LeakyReLU²** | 激活函数 | -4.3% BPB |
| **BPE-8192** | 分词器 | -35% BPB |

### 已实现但未集成到最新架构

| 技术 | 脚本 | 说明 |
|------|------|------|
| **LoRA + TTT** | `scripts/modal/modal_lora_ttt.py` | 榜首技术，能达到 1.08 BPB |
| **XSA + TTT** | `scripts/modal/modal_xsa_ttt.py` | TTT 变体 |

### 已实验但效果不好

| 技术 | 结果 | 原因 |
|------|------|------|
| PLE (Per-Layer Embedding) | BPB +274% | 干扰残差流 |
| 数据质量过滤 | BPB -0.2% | 多样性更重要 |
| EMA 权重平均 | BPB -0.8% | 训练步数太少 |

---

## 📁 代码结构

```
parameter-golf-solution/
├── scripts/
│   ├── modal/                    # Modal 云端训练
│   │   ├── modal_alternating_attn.py  # ⭐ 最新架构
│   │   ├── modal_mhc_v2_deep.py       # Baseline (mHC)
│   │   ├── modal_lora_ttt.py          # LoRA + TTT
│   │   └── ...
│   └── local/                    # 本地测试
├── docs/
│   ├── experiments/
│   │   ├── EXPERIMENTS.md        # 实验汇总
│   │   └── reports/              # 详细报告
│   └── analysis/
├── skills/
│   └── experiment-analysis/      # 实验分析模板
├── BASELINE_CONFIG.md            # 基准配置
├── AGENTS.md                     # AI 协作规则
└── README.md
```

---

## 🔑 关键配置

### Baseline 配置 (BASELINE_CONFIG.md)

```yaml
Model:
  dim: 384
  n_layers: 20
  n_heads: 8
  n_kv_heads: 4
  vocab_size: 8192

Training:
  seq_len: 256
  batch_size: 64
  steps: 5000
  lr: 1e-3
  warmup: 200 steps
  min_lr_ratio: 0.1

Data:
  path: /data/datasets/fineweb10B_sp8192
  train_shards: 5 (500M tokens)
  BYTES_PER_TOKEN: 3.67
```

### BPB 计算公式

```python
BYTES_PER_TOKEN = 3.67
bpb = (val_loss / math.log(2)) * (1.0 / BYTES_PER_TOKEN)
```

⚠️ **重要**: 之前脚本漏了 `BYTES_PER_TOKEN` 转换，导致报告数值错误。已修复。

---

## 📋 待办事项

### 高优先级

1. **集成 LoRA + TTT 到 Alternating Attention**
   - 当前最佳 (Alt-A + mHC) 只有 1.4883 BPB
   - 加入 TTT 预期能提升到 ~1.1 BPB
   - 参考: `scripts/modal/modal_lora_ttt.py`

2. **训练 Alternating 专用 mHC**
   - 当前用的是全 Global 模型的 mHC 参数
   - 用 Alternating 架构重新训练 mHC，可能更优

### 中优先级

3. **Alt-C: FlashAttention + seq_len=1024**
   - 真正的稀疏计算
   - 长上下文训练
   - 需要 `pip install flash-attn`

4. **层级差异化架构**
   - 基于 mHC 发现：浅层 MLP 重要，深层 Attention 重要
   - 浅层用更小的 attention，深层用更大的

### 低优先级

5. **更长训练** (5000 → 10000 steps)
6. **调整 Local/Global 比例** (尝试 2:1 或 3:1)

---

## 🔬 关键发现

### mHC α/β 分布规律

| 阶段 | 层 | 特点 |
|------|-----|------|
| 浅层 | 0-3 | MLP 主导 (β_mlp 高)，Attention 弱 |
| 过渡 | 4-5 | 两者都下降 |
| 深层 | 6+ | Attention 主导 (β_attn 高) |

**~90% 深度异常规律**：在 11L/20L/32L 模型中，约 85-95% 深度的层会出现 β_attn 峰值。

### Alternating Attention

- **Local 层**: 学局部模式 (n-gram 级别)
- **Global 层**: 学全局语义
- **层级分工有效**: 不需要每层都做全局 attention

---

## 🚀 运行命令

### 最佳配置 (Alt-A + mHC)

```bash
cd /tmp/parameter-golf-solution
modal run --detach scripts/modal/modal_alternating_attn.py --mhc
```

### 其他实验

```bash
# Alt-A (Vanilla)
modal run --detach scripts/modal/modal_alternating_attn.py

# Alt-B (dim=448)
modal run --detach scripts/modal/modal_alternating_attn.py --dim 448

# LoRA + TTT (独立脚本)
modal run --detach scripts/modal/modal_lora_ttt.py

# 继续训练 (从 checkpoint)
modal run --detach scripts/modal/modal_alternating_attn.py --steps 10000 --resume /data/checkpoints/alternating_attn/alt_mHC_dim384_L20_step5000.pt
```

---

## 📊 实验报告位置

- `docs/experiments/EXPERIMENTS.md` — 汇总
- `docs/experiments/reports/2026-04-03_alt-a-vanilla.md`
- `docs/experiments/reports/2026-04-03_alt-a-mhc.md`
- `docs/experiments/reports/2026-04-04_alt-b.md`

---

## ⚠️ 注意事项

1. **BPB 计算必须包含 BYTES_PER_TOKEN**
2. **Modal 脚本用 `--detach` 防止断连**
3. **代码和日志全用英文**（比赛要求）
4. **不要在代码里用 emoji**
5. **每次实验后出分析报告**（使用 `skills/experiment-analysis/SKILL.md` 模板）

---

## 🔗 相关链接

- **GitHub Repo**: https://github.com/Elarwei001/parameter-golf-solution
- **比赛**: https://github.com/openai/parameter-golf
- **榜首 PR**: https://github.com/openai/parameter-golf/pull/548 (BPB 1.0865)
- **我们的 PR**: https://github.com/openai/parameter-golf/pull/1254 (BPB 1.1070)

---

*交接完成后，下一步建议：将 LoRA+TTT 集成到 Alt-A+mHC 架构，目标 BPB < 1.1*

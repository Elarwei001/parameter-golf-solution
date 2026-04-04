# 实验航海日志

> 从 2.28 BPB 到 1.44 BPB，记录每次探索

## 总进度

| 日期 | 最佳 BPB | 关键里程碑 |
|------|---------|-----------|
| 2026-03-31 | 2.28 | 起航，字节级 baseline |
| 2026-03-31 | 1.40 | BPE-8192 + QAT，大跳跃 |
| 2026-04-01 | 0.986 | XSA + LoRA TTT 组合 |
| 2026-04-03 | 1.4777 | mHC v2 + Alternating Attention + scratch |
| 2026-04-04 | 1.5321 | Sandwich QAT Adaptive（QAT 仍有退化）|

---

## 2026-04-01

### XSA + LoRA TTT 组合
- **改动**: XSA (Exclusive Self Attention) + Test-Time Training with LoRA
- **结果**: BPB **0.986** 🏆 (vs 1.40 baseline, -27.6%)
- **发现**: TTT 是巨大的跳跃，但推理开销高
- **详见**: archive/

### XSA (Exclusive Self Attention)
- **改动**: 去除 self-similarity bias — `z = y - proj(y, v_self)`
- **结果**: BPB **1.441** (-2.6% vs 1.48 standard attention)
- **发现**: 2 行代码，简单有效
- **详见**: records/（无独立 record，集成于后续实验）

---

## 2026-04-03

### mHC v2 (11层 baseline)
- **改动**: 双参数可学习残差 `y = α*x + β*layer_out`，对比 v1 的单参数
- **配置**: dim=384, 11L, BPE-8192
- **结果**: BPB **1.5167** (+0.13% vs 1.5187 standard residual) ✅
- **发现**: 浅层 MLP 主导，深层 Attention 主导；Layer 4-5 是过渡区；α+β 约为 1.3-1.9，低于标准残差的 2.0
- **详见**: records/2026-04-03_mhc-v2/

### mHC v2 20层 / 32层深度实验
- **改动**: 验证 mHC α/β 规律在更深模型中是否一致
- **配置**: 20L BPB=1.4978，32L BPB=1.4651
- **发现**: **~90% 深度异常规律** — β_attn 峰值稳定出现在 85-95% 深度！11L→Layer 9, 20L→Layer 18-19, 32L→Layer 27
- **详见**: records/2026-04-03_mhc-v2-20l/

### Alternating Attention (Alt-A Vanilla + Alt-A mHC 迁移)
- **改动**: 偶数层 Global attention，奇数层 Local (window=128)
- **配置**: dim=384, 20L, 11 Global + 9 Local
- **结果**: Vanilla BPB=1.5001 (-0.16%)，mHC 迁移 BPB=1.4883 (-0.95%)
- **发现**: Alternating 几乎不掉点；mHC 参数可跨架构迁移
- **详见**: records/2026-04-03_alt-a-mhc-vanilla/

---

## 2026-04-04

### Alt-A mHC-scratch (Uniform MLP) ⭐ 当前最佳 FP32
- **改动**: Alternating Attention + 从头学习 mHC 参数（初始化为 1.0）
- **配置**: dim=384, 20L, mHC scratch
- **结果**: BPB **1.4777** (-1.65% vs baseline 1.5025)
- **发现**: 从头学习 > 迁移参数；架构特异性比预训练初始化更重要
- **详见**: records/2026-04-04_alt-a-mhc-scratch/

### Alt-B (dim=448)
- **改动**: 加大 dim，Alternating Attention
- **配置**: dim=448, 20L, 39.82M params
- **结果**: BPB **1.4952** (-0.49% vs baseline)
- **发现**: 参数 +21%，BPB 提升仅 0.49%，效率低于 mHC scratch
- **详见**: records/2026-04-04_alt-b/

### Sandwich MLP
- **改动**: 浅层 3x MLP，中段 1.2x，深层 3x
- **配置**: Layers 0-3: 3x, 4-16: 1.2x, 17-19: 3x，Alternating + mHC
- **结果**: BPB **1.4833** (-1.28%)，参数 22.79M (-23%)
- **发现**: 省 23% 参数，仅劣化 0.38%；自洽性悖论：设计依据与 Sandwich 自身学到的 β_mlp 不一致
- **详见**: records/2026-04-04_alt-a-mhc-sandwich/

### Front-loaded MLP
- **改动**: 只有浅层 3x，中层和深层都用 1.2x
- **配置**: Layers 0-3: 3x, 4-19: 1.2x，21.20M params
- **结果**: BPB **1.4918** (+0.95% vs Uniform)
- **发现**: 深层 MLP 仍然重要；Front 中段 β_attn 升高（补偿机制）；Sandwich 是最优 MLP 配置
- **详见**: records/2026-04-04_alt-a-mhc-front/

### Sandwich QAT (from step 0)
- **改动**: Sandwich MLP + Ternary QAT (1.58-bit) 从 step 0 开始
- **配置**: dim=384, 20L, 量化大小 16.52 MB
- **结果**: BPB **1.5350** (+3.5% vs Sandwich FP32)
- **发现**: Embedding 是瓶颈（12.60MB = 78.8% 预算）；从 step 0 开始 QAT 太激进
- **详见**: records/2026-04-04_sandwich-qat-from-step0/

### Sandwich QAT Adaptive ⭐
- **改动**: 自适应 QAT 切换（先 FP32 → EMA loss 收敛 → 切 QAT）
- **配置**: dim=384, 20L, Sandwich MLP, 实际在 step 500 切换
- **结果**: BPB **1.5321** (vs baseline 1.5025, +1.97%)
- **发现**: Adaptive 比 from-step-0 好 0.19%，但 QAT 仍有退化；切换时机 step 500 可能还太早
- **详见**: records/2026-04-04_sandwich-qat-adaptive/

### FP16 Full Model 转换
- **改动**: 训练好的 QAT 模型全体转 FP16
- **结果**: BPB **1.5365** (+0.20%)，模型大小 91.18MB → 45.59MB
- **发现**: FP16 转换整个模型只损失 0.20% BPB，模型减半
- **详见**: records/2026-04-04_fp16-full-model/

### Embedding FP16 转换 ⭐
- **改动**: 仅 embedding FP32→FP16，其余不变
- **结果**: BPB **1.5336** (+0.16%)，量化大小 16.52MB → 10.23MB，省 **6.29MB**
- **发现**: Embedding 精度对模型影响极小，可以安全用 FP16；为升级到更大 dim 留出空间
- **详见**: records/2026-04-04_embedding-fp16/

### Sandwich QAT 30L ⭐ 当前最佳 QAT
- **改动**: 层数 20→30，其余不变（dim=384, Sandwich MLP, adaptive QAT）
- **配置**: 31.82M params (ternary 28.65M + FP32 3.17M), 16 Global + 14 Local
- **结果**: BPB **1.5215** (vs 20L QAT 1.5321, **-0.69%**)
- **发现**: 加深层数有效！30L 比 20L 奮 0.0106 BPB；但 QAT 退化仍是主要瓶颈（vs FP32 baseline +1.27%）；训练时间 2439s (+46%)
- **详见**: records/2026-04-04_sandwich-qat-30l/

---

## 历史实验 (2026-03-31 / 04-01)

### 激活函数对比
- LeakyReLU2 BPB=2.18 最佳，GELU=2.28 baseline

### Sliding Window Attention
- window=192 最佳 BPB=2.18

### Tokenizer 对比 ⭐⭐
- 字节级: 2.17 → BPE-1024: 1.68 → **BPE-8192: 1.40**
- Tokenizer 影响远超模型架构，是最大的单一改进

### QAT 量化 ⭐⭐
- FP32 1.402 → QAT 1.58-bit 1.403，**几乎无损**
- 模型从 120MB 压缩到 13.5MB（9 倍）

### Embedding 空间效率分析
- 训练后有效维度从 482→104（-78%），但直接用小 dim 效果差
- dim=128 + whitening: BPB=1.704（+15% 负面结果）

### Progressive Attention (渐进式 Attention)
- 3 层 MLP-only + 8 层 full，参数减 6.5%，BPB 差 0.25%
- 浅层 Attention 确实不太重要，验证了 mHC 的发现

### PLE (Per-Layer Embedding)
- BPB=1.5315 (+1.93% vs baseline)，不采用

### Data Quality Filtering
- Gaussian weighted sampling 反而更差，数据多样性 > 质量过滤

### EMA Weights
- 5000 步训练中 EMA 更差（decay=0.999）

---

*最后更新: 2026-04-04*

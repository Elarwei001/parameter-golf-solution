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

### Sandwich Late QAT at Step 3000
- **改动**: 固定延后 QAT，在前 3000 steps 保持 FP32，再切入 ternary QAT
- **配置**: dim=384, 20L, Sandwich MLP + mHC, `qat_start_step=3000`, 总步数 5000
- **结果**: BPB **1.5268**，Val loss **3.8839**，量化大小 **16.52MB**
- **对比**:
  - 比 prior late QAT step 4000 (`1.5412`) **更好 0.0144 BPB**
  - 比 adaptive QAT 20L (`1.5321`) **更好 0.0053 BPB**
  - 但仍比 Sandwich FP32 (`1.4833`) **差 2.93%**
- **发现**: 把 QAT 从 step 4000 提前到 step 3000 确实有帮助，说明 1000 QAT steps 太短；但这条线依然没有解决 QAT 退化，量化仍是主瓶颈
- **详见**: records/2026-04-17_sandwich-late-qat-3000/

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

*最后更新: 2026-04-06*

---

## 2026-04-05

### dim=464 Sandwich QAT 30L (BPE 8192)
- **改动**: 加大 dim=464, 30层 Sandwich (6×3.0 + 21×1.2 + 3×3.0)
- **配置**: dim=464, n_heads=8, n_kv=4, head_dim=58, 40k steps, A100-40GB
- **结果**: Val BPB **1.3546**, LoRA+TTT BPB **1.0715** (-34.0%)
- **发现**: LoRA+TTT 效果惊人；但模型 18.09MB 超 16MB 限制
- **详见**: records/2026-04-05_dim464-sandwich-qat/

### dim=448 Sandwich QAT 30L (BPE 8192) ⭐
- **改动**: dim 降到 448, MLP 缩减为 2×3.0 + 28×1.2，确保 fit 16MB
- **配置**: dim=448, n_heads=8, n_kv=4, head_dim=56, 40k steps, A100-40GB
- **结果**: Val BPB **1.3805**, 旧版 LoRA+TTT 记录 **1.0885**
- **模型大小**: ~14.36MB ✅
- **后续复核**: 改写为更接近 leaderboard 的 legal score-first TTT 后，A100 复跑只有 **~1.524x BPB**，说明原先 1.0885 不能直接视为当前 leaderboard 口径下的 legal TTT 结果
- **发现**: dim=448 基座仍可用，但此前强 TTT 提升主要来自非当前 legal score-first 设定
- **详见**: records/2026-04-05_dim448-sandwich-qat/

### 11L Depth Compare (baseline vs recurrence vs sandwich+mHC)
- **改动**: 做一个干净的三组基座对照，只比较深度利用方式，不加 TTT
- **配置**: SP8192, dim=448, 11 物理层, 5k steps, A100-40GB
- **结果**:
  - baseline: **1.4936** (`23.55M` params)
  - sandwich+mHC: **1.4887** (`19.93M` params)
  - recurrence: **1.4957** (`23.55M` params, 由 step 4000 checkpoint 续跑完成)
- **发现**: 在这个小预算短跑设定下，**sandwich+mHC 以更少参数胜出**；plain baseline 第二；简单 replay 中层的 recurrence 第三，说明这套 recurrence 在当前设定下没有带来收益
- **详见**: records/2026-04-13_depth-compare-11l/

---

## 2026-04-06

### dim=544 Sandwich QAT 30L + Scylla Tokenizer (998 vocab) 🔄
- **改动**: 切换到 Scylla tokenizer (vocab 998), dim 提升到 544, MLP 恢复 9×3.0 + 21×1.2
- **配置**: dim=544, n_heads=8, n_kv=4, head_dim=68, 40k steps, A100-40GB
- **预期优势**:
  - Embedding 从 7.34MB → 0.90MB，省 6.4MB
  - 参数 58.09M (比 dim=448 多 41%)
  - Scylla bytes/token ~4.13 (vs BPE ~3.67)
  - 模型大小 ~14.75MB ✅
- **状态**: 训练中，预计 2026-04-06 晚完成
- **详见**: records/2026-04-06_dim544-scylla/

---

## 技术对比 (vs 榜首 PR #1405, 1.0856 BPB)

| Technique | Us | PR #1405 | Priority |
|-----------|-----|----------|----------|
| Tokenizer | BPE 8192 → Scylla 998 ✅ | Scylla 998 | Done |
| Quantization | Ternary (1.58-bit) | Full Hessian GPTQ int6 | Future |
| Compression | None | LZMA-9 | Future |
| Optimizer | AdamW | Parallel Muon | Future |
| EMA/SWA | None | Yes | Easy win |
| LoRA+TTT | ✅ | No | Our advantage |
| BigramHash | No | 3072×112 | Future |
| QK-Gain | No | 4.0 | Easy win |

## TODO

- [ ] **Alt-attn 消融实验**: 全 global attention 去掉 alt-attn，验证 mHC 参数波浪是否由 global/local 交替导致
- [ ] **dim=448 + Scylla 消融**: 单独验证 Scylla tokenizer 的 BPB 改善（控制 dim 不变）

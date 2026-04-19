# PR #1493 榜首方案分析（1.0810 BPB）

> 分析日期：2026-04-19
> 源：https://github.com/openai/parameter-golf/pull/1493
> 作者：bigbag (Pavel Liashkov)

## 1. 榜首完整技术栈

### 1.1 模型架构（11 物理层 → 17 虚拟层）

| 组件 | 配置 |
|------|------|
| Tokenizer | SP8192（SentencePiece）|
| 模型维度 | `dim=512, embed_dim=512, num_heads=8, num_kv_heads=4, head_dim=64` |
| 物理层数 | `num_layers=11` |
| 深度循环 | L3-5 循环 2 次（`frac≥0.35` 时启用）→ 17 虚拟层 |
| U-Net skip | encoder/decoder 对称 skip，带 `skip_weights` + sigmoid `skip_gates` |
| Parallel Residual | `layer_idx >= 7` 使用并行残差（attn 和 MLP 并行，GPT-J 风格）|
| Block 内残差 | `resid_mix`：每层用可学习权重混合 x 和 x₀（原始 embedding）|
| 层级标量 | `attn_scale`、`mlp_scale`（shape=[dim]，每层每维可学习）|
| XSA | 后 N 层启用（默认 N=11 全部）|
| QK-Gain | 5.25（每 head 一个标量，初始化为该值）|
| Partial RoPE | 只对前 16 个 dim 做 RoPE |
| LN Scale | `ln_scale_factor = 1/√(layer_idx+1)` |
| 激活 | `LeakyReLU(slope=0.5)` 后平方 |
| Logit Softcap | `30 × tanh(logits / 30)` |
| MLP 倍率 | `mlp_mult=4.0`（均匀）|
| 权重共享 | `tie_embeddings=True`（input/output 权重共享）|
| Init | tied embedding std=0.005；其他正交初始化；proj 层零初始化 |

### 1.2 训练配置

| 项目 | 配置 |
|------|------|
| Optimizer（矩阵）| **Muon + Newton-Schulz（5 步）**，row_normalize |
| Optimizer（embed）| AdamW，lr=0.03（tied），wd=0.085 |
| Optimizer（head）| Adam，lr=0.008 |
| Optimizer（scalar/gate）| AdamW，lr=0.02，wd=0.02 |
| matrix LR | 0.022 |
| Muon momentum | 0.99（从 0.92 经 1500 步 warmup）|
| WD | muon=0.095, embed=0.085, adam=0.02 |
| EMA | decay=0.9965 |
| Warmdown | 后 72% 线性下降到 0 |
| Grad clip | 0.3 |
| Iterations | 20000, `max_wallclock=600s` |
| Batch | 786432 tokens, seq_len=2048, 8xH100 |

### 1.3 推理与压缩

| 项目 | 配置 |
|------|------|
| 量化 | **GPTQ** int6 matrices + int8 embeddings，SDClip（matrix 12.85σ，embed 20σ）|
| 压缩 | byte shuffle（stride=2）+ Brotli（q=11）|
| Sliding eval | stride=64，每窗口仅计算新增部分的 loss |
| Legal Score-First TTT | SGD lr=0.005，mom=0.9，3 epochs，chunk 32768 tokens，cosine LR 衰减 |

## 2. 我们当前方案 vs PR #1493

| 技术 | 我们 | PR #1493 | 差距 |
|------|------|----------|------|
| Tokenizer | BPE 8192 | SP8192 | 相似 |
| 量化 | Ternary (1.58-bit) | GPTQ int6 + brotli | **巨大差距** |
| Optimizer | AdamW | Muon + Newton-Schulz | **大差距** |
| 深度 | 30 物理层 | 11 物理 + 循环 → 17 虚拟 | 不同思路 |
| 残差 | mHC（每层 α/β 2 参数）| resid_mix + attn_scale + mlp_scale（每维）| 他们更细粒度 |
| Parallel Residual | ❌ | L7+ | **缺失** |
| U-Net Skip | ❌ | 对称 skip+gates | **缺失** |
| Partial RoPE | ❌ | 16/64 dims | 小收益 |
| LeakyReLU² | ✅ | ✅ | 同 |
| XSA | ❌（只有脚本）| ✅ 全部层 | **缺失** |
| QK-Gain | 4.0（Phase 1.1 加）| 5.25 | 需调参 |
| Legal Score-First TTT | ❌ | ✅ | **关键缺失** |
| EMA | ❌（Phase 1.1 加）| ✅ 0.9965 | 即将加 |
| Logit Softcap | ❌ | 30 | **缺失** |
| LN Scale | ❌ | 1/√(l+1) | **缺失** |

## 3. 在榜首之上的可能优化空间

### 3.1 高把握改动（我们有先验）

**1) Sandwich MLP × 11L 循环架构**
- PR #1493 用 `mlp_mult=4.0` 均匀，我们验证过 sandwich（首尾 3.0x，中间 1.2x）能省 23% 参数
- 省下的参数给 dim（512→544）或 embed_dim
- 预期：**-0.005 ~ -0.010 BPB**

**2) QK-Gain 分层扫描**
- PR #1493 用 per-head 5.25（所有层一样初始化）
- 深层 attention 更重要（mHC 实验证实），深层用 6.0，浅层 4.0
- 预期：**-0.001 ~ -0.003 BPB**

### 3.2 中把握改动（需小规模实验）

**3) 循环范围/次数**
- PR #1493：L3-5 × 2 循环（17 虚拟）
- 尝试：L2-6 × 2（21 虚拟）或 L3-5 × 3（20 虚拟）
- 预期：未知，需实验

**4) Logit softcap 值**
- 固定 30。扫 {20, 25, 30, 40, 50}
- 预期：**-0.001 ~ -0.002 BPB**

**5) `enable_looping_at` 时机**
- 固定 0.35（训练 35% 进度开启循环）
- 尝试 0.25 / 0.5
- 预期：未知

### 3.3 低把握改动（榜首可能已扫过）

**6) Embed bits / clip_sigmas**
- int8 @ 20σ；试 int7 @ 15σ
- 预期：**-0.001 BPB**

**7) Byte-shuffle stride**
- 固定 stride=2。试 3/4
- 预期：极小

## 4. 关键障碍

1. **硬件**：我们没有 8xH100 SXM。Modal A100 跑出的结果**不合法**（不能提交官方榜）。必须用 Runpod 或 OpenAI compute grant。
2. **工程复杂度**：Muon、GPTQ SDClip、Legal TTT、byte shuffle、brotli 都要移植；或直接 fork 他们的代码。
3. **时间**：距 4/30 截止只剩 11 天。

## 5. 建议路径

**最现实的方案：直接 fork PR #1493 代码 + 叠加 1 个差异化**，而不是从零集成。

优先级：
1. 在 Runpod 1×H100 跑通原始代码（验证环境，~1 天）
2. 单 seed 复现 ~1.08（~1-2 天）
3. 叠加 Sandwich MLP（最有把握的差异化）
4. 如果有余力再试循环参数扫 / QK-Gain 分层
5. 3 seed 提交（~1 天）

## 6. 当前 Phase 1.1 实验的价值

我们在 Modal 上跑的 40k steps EMA+SWA+QK-Gain 实验：

- 不能直接进榜（硬件不合规 + baseline 1.38 离榜门槛太远）
- 但能验证：EMA / SWA / QK-Gain 在我们架构上的收益是否真实
- 结果可指导后续 fork #1493 后的差异化实验（例如 QK-Gain 是否确实有用）

可以视作"**Phase 0: 我们自有技术栈收益验证**"。

## 参考

- 原始代码（LZMA 压缩）：`research/pr1493/train_gpt_pr1493_diff.py`
- 解压后代码：`research/pr1493/train_gpt_decoded.py`
- PR 描述：`research/pr1493/README.md`

# 榜首技术全解（面向 AI 新手）

> 目标：把 PR #1493（1.0810 BPB）用到的每一个技巧拆成独立知识点，配直觉、伪代码、"为什么有效"。
> 读完后你应该能：看到代码里某个花哨写法，能说出它解决什么问题、大致怎么工作。

## 先备知识：Transformer 一个 Block 在做什么

一个标准的 Transformer Block 做两件事：

```python
# 输入 x shape = [batch, seq_len, dim]
x = x + Attention(LayerNorm(x))   # 注意力子层
x = x + MLP(LayerNorm(x))          # 前馈子层
# 输出 x shape 不变
```

- **Attention**：让每个 token "看"其他 token，汇总信息（"他说的他和这里这个代词有关"）
- **MLP**：对每个 token 独立做非线性变换（"把信息压缩/提取特征"）
- **Residual（残差连接）**：`x = x + f(x)` 而不是 `x = f(x)`，保证信息不丢失、梯度好传
- **LayerNorm / RMSNorm**：数值归一化，让训练稳定

几乎所有优化都是在这三件事上打补丁。

---

## 第 1 部分：Tokenizer

### 1.1 SP8192（SentencePiece BPE，词表 8192）

**作用**：把一段文本切成 token id 序列。

**为什么重要**：评分用的是 **BPB（bits per byte）**，不是 bits per token。
```
BPB = cross_entropy_loss / ln(2) / (bytes_per_token)
```
同样的模型 loss，如果 **token 覆盖的字节更多**，BPB 就更低（更好）。

- 字节级 tokenizer：`bytes_per_token = 1`，BPB = loss
- BPE 8192：`bytes_per_token ≈ 3.67`，BPB ≈ loss/2.56
- SP8192（我们实验叫 Scylla 998 是同族）：针对 FineWeb 优化，`bytes_per_token ≈ 4.13`

**结论**：tokenizer 是最大杠杆之一。词表大一点、切得聪明一点，BPB 直接掉一档。

---

## 第 2 部分：Embedding 层相关

### 2.1 Tied Embeddings（绑定输入/输出 embedding）

**做法**：输入 embedding `tok_emb` 和输出投影 `lm_head` 共用同一个权重矩阵。

```python
logits = x @ tok_emb.weight.T   # 不再有独立 lm_head
```

**为什么**：
- 省一半 embedding 参数（`vocab_size × dim`）
- 在参数严格受限的 16MB 挑战里特别关键

**副作用**：input 和 output 被迫耦合，通常需要更小的初始化标准差（PR #1493 用 `std=0.005`）来平衡。

### 2.2 Partial RoPE（只对部分维度做旋转位置编码）

**背景**：RoPE（Rotary Position Embedding）是给 Q、K 注入位置信息的标准做法，对每个 head_dim 做旋转。

**Partial RoPE**：只对前 N 个 dim 做旋转，后面的 dim **保留原始内容信息**，不被位置干扰。

```python
# head_dim = 64, rope_dims = 16
x_with_pos = rotate(x[:16])
x_content = x[16:]            # 这部分不动
out = concat([x_with_pos, x_content])
```

**为什么有效**：位置信息只需要一部分维度表达；剩下的维度留给纯语义，能学得更干净。

---

## 第 3 部分：Attention 层的优化

### 3.1 RMSNorm on Q / K

**做法**：在计算 attention 之前，对 Q 和 K 各自做 RMSNorm。

```python
q = rms_norm(q); k = rms_norm(k)
attn = softmax(q @ k.T / sqrt(d))
```

**为什么**：训练初期 Q/K 的范数可能爆炸或消失，导致 softmax 饱和。先归一化能稳定训练。这是 modern 配方里的标配。

### 3.2 QK-Gain（可学习的 attention 温度）

**做法**：给 q 乘一个可学习标量（每 head 一个）。

```python
q = q * q_gain                          # q_gain shape=[num_heads], 初始化=5.25
attn = softmax(q @ k.T / sqrt(d))
```

**效果**：相当于让模型自己学一个"锐度"——值越大，softmax 越尖锐（attention 更集中到少数 token）。

**为什么初始化 5.25 比 1 好**：相当于 attention 温度 = 1/5.25 ≈ 0.19，默认比标准更锐利，实验经验发现这对语言建模友好。值是 hyperparameter 扫出来的。

### 3.3 GQA（Grouped-Query Attention，我们已在用）

**做法**：num_heads=8 但 num_kv_heads=4，每 2 个 head 共享一组 KV。

**为什么**：Q 通常需要很多个 head 看不同角度；但 K/V 可以共享，省内存、省参数。

### 3.4 XSA（Exclusive Self Attention）

**做法**：attention 输出里去掉"自己看自己"的那部分。

```python
y = attention(q, k, v)                       # 标准 attention 输出
v_self = v[token_i]                          # 当前 token 自己的 value
projection = (y · v_self_normalized) * v_self_normalized
z = y - projection                           # 扣掉这部分
```

**为什么**：在 causal attention 下，每个 token 至少会 attend 到自己，这部分信息其实已经在残差连接里有了。把它从 attention 输出里减掉，能让 attention 更专注于"其他 token 传来的新信息"。

**效果**：我们实验里 -2.6% BPB（几行代码，白给）。

---

## 第 4 部分：残差连接的进化

这是 PR #1493 最精细的一块。标准做法 `x = x + f(x)`，他们做了一连串改进。

### 4.1 resid_mix（混合 x₀ 回来）

**做法**：每层不只加 `f(x)`，还可以把**最初的 embedding** `x₀` 混进来。

```python
# 每层有个 [2, dim] 的可学习矩阵 resid_mix
x_in = resid_mix[0] * x + resid_mix[1] * x0   # 初始化 [1, 0]
# 然后对 x_in 做 attn + mlp
```

**为什么**：深层 x 已经被反复变换，容易偏离原始 token 信息。允许每层"回看原始 embedding"，能让模型在深层仍保留词本身的信息。初始化 [1, 0] 保证不加入噪声，但允许学习到底要不要混。

### 4.2 attn_scale / mlp_scale（每维可学习门）

**做法**：attention 输出和 MLP 输出各自有一个 `[dim]` 大小的可学习缩放。

```python
x = x + attn_scale * attn_out    # attn_scale shape=[dim], 初始化全 1
x = x + mlp_scale * mlp_out
```

**和我们的 mHC 对比**：
- 我们 mHC：每层 4 个标量（α_attn, β_attn, α_mlp, β_mlp）
- PR #1493：每层 2×dim 个标量（每个维度独立）

他们**粒度更细**，能让不同通道用不同强度。这是一个潜在的改进方向。

### 4.3 Parallel Residuals（GPT-J 风格）

**标准串联**：
```python
x = x + attn(norm(x))
x = x + mlp(norm(x))       # mlp 的输入依赖 attn 的输出
```

**并行**：
```python
normed = norm(x)
x = x + attn(normed) + mlp(normed)  # attn 和 mlp 同时计算，互不依赖
```

**为什么**：
- 计算上可以并行（速度快）
- 少一层 norm，结构更简洁
- 实验上 loss 几乎不损失

**PR #1493**：前 7 层用串联（浅层需要顺序处理），L7+ 用并行。

### 4.4 LN Scale（层深归一化缩放）

**做法**：每层的 LN 输出乘一个 `1/√(layer_idx+1)`。

```python
x = norm(x) * (1 / sqrt(layer_idx + 1))
```

**为什么**：深层的残差流累积了很多层的输出，范数越来越大。除以 √layer 能抵消这种累积，让每层看到的输入保持类似的 scale，训练更稳定。类似 "µP"（muP, maximal update parametrization）的思想。

### 4.5 U-Net Skip Connections

**做法**：把 11 层分成"encoder 前半 + decoder 后半"，在 decoder 里把对称 encoder 层的输出跳接过来。

```
Encoder: L0 → L1 → L2 → L3 → L4 → L5(中)
Decoder: L5 → L6 → L7 → L8 → L9 → L10
Skip:      └────────────↑ L6 从 L4 拿 skip
           └──────────────────↑ L7 从 L3 拿 skip
           ...
```

```python
skip = encoder_output[i]
scaled = skip_weights[i] * skip                         # 可学习缩放
gate = sigmoid(skip_gates[i])                           # 可学习门
x = lerp(scaled, x, gate)                               # 插值：gate=1→只用 x, gate=0→只用 skip
```

**为什么**：
- 受 U-Net 在图像分割里的启发
- 让深层能"直接看到"浅层的特征，不用全靠残差流传递
- `skip_gates` 初始化为 0（sigmoid(0)=0.5），均衡起步，模型自己学要不要开

---

## 第 5 部分：激活函数与输出

### 5.1 LeakyReLU²（平方的 LeakyReLU）

**公式**：`f(x) = max(x, 0.5*x)²`

```python
y = F.leaky_relu(x, negative_slope=0.5) ** 2
```

**为什么**：
- 比 GELU/SiLU 更便宜（没 exp）
- 平方让激活是"二次的"，表达力更强
- negative_slope=0.5 让负半轴不死掉（避免 dying ReLU）
- 经验上在小模型里效果好

### 5.2 Logit Softcap

**做法**：输出 logits 前用 tanh 压一下。

```python
logits = 30 * tanh(logits / 30)
```

**为什么**：限制 logits 绝对值上限为 30，防止训练中某些 token 的 logit 爆炸。类似一个"软性 clip"，不像硬截断那样打断梯度。这个技巧来自 Gemma2。

---

## 第 6 部分：深度循环（Depth Recurrence）

### 6.1 概念

**做法**：训好的某些层重复使用多次，相当于"深度循环"。

```
物理层: L0 L1 L2 [L3 L4 L5] L6 L7 L8 L9 L10
执行顺序（2 次循环）:
         L0 L1 L2 [L3 L4 L5] [L3 L4 L5] [L3 L4 L5] L6 L7 L8 L9 L10
                                                   ↑ 17 次前向
```

**为什么有效**：
- **省参数**：17 个"虚拟层"只用 11 个物理层的参数
- 深度对大模型重要，但参数预算不够时，让某几层重复走几次，近似实现"更深"
- 灵感来自 Universal Transformer

**关键细节 `enable_looping_at=0.35`**：前 35% 训练时不循环（让层先学基础功能），之后才开启循环（让层学"可复用"的转换）。

**我们实验的教训**：单独复现 recurrence 时（L4-5 × 5）反而更差；但在完整 stack（skip + resid_mix + parallel 等）加持下是榜首技术。说明它需要配合别的组件才能发挥。

---

## 第 7 部分：Optimizer——Muon

### 7.1 AdamW（对照）

**标准做法**：对每个参数维护动量和二阶矩，自适应步长。

```python
m = β₁·m + (1-β₁)·g        # 一阶动量
v = β₂·v + (1-β₂)·g²       # 二阶矩
update = m / (√v + ε)
```

### 7.2 Muon：针对矩阵参数的优化器

**关键思想**：对**矩阵参数**，梯度 G 的最优更新方向不一定是 G 本身，而是 G 的"正交化"版本——这样更新能充分利用矩阵的所有方向。

**步骤**：
1. 像 AdamW 一样算动量：`buf = momentum·buf + g`
2. 对 `g` 做 Newton-Schulz 迭代，把它"正交化"（把 SVD 的奇异值都压成 1）

```python
# 简化版 Newton-Schulz（5 步）
X = G / ‖G‖
for _ in range(5):
    A = X @ X.T
    B = b·A + c·A @ A
    X = a·X + B @ X        # a,b,c = 3.4445, -4.775, 2.0315
# 现在 X 是 G 的"正交化"版本
update = X * lr
```

**为什么有效**：标准 SGD 的更新方向受 G 的大奇异值主导，小方向基本不动。Muon 把所有方向都放大到同样的 scale，学得更均匀。

**Row Normalize**：额外在 Newton-Schulz 前对每行做归一化，进一步稳定训练。

**Momentum Warmup**：Muon momentum 从 0.92 经 1500 步涨到 0.99。高动量能让收敛更稳，但起步就 0.99 会"粘住"，所以先低后高。

**代价**：只能用于矩阵（ndim=2），且不能太小。标量参数、embedding 还是用 AdamW。

---

## 第 8 部分：训练技巧

### 8.1 多 Optimizer / 分组 LR

PR #1493 同时跑 3-4 个 optimizer：
- Muon：所有 block 内的矩阵参数（attention QKV、MLP），lr=0.022, wd=0.095
- AdamW：token embedding（特殊，lr=0.03, wd=0.085）
- Adam：lm_head（如果不 tied，lr=0.008）
- AdamW：所有 scalar / gate 参数（`attn_scale`, `q_gain` 等，lr=0.02, wd=0.02）

**为什么**：不同类型的参数有非常不同的学习动态。embedding 要学分布式表示，变化慢且敏感；矩阵需要正交化更新；标量门需要精细微调。**分别调参收益很大**。

### 8.2 EMA（Exponential Moving Average）

**做法**：训练时在主权重之外维护一份"滑动平均版本"。

```python
for name, p in model.state_dict().items():
    ema[name] = ema[name] * 0.9965 + p * 0.0035
# 训练结束后，用 ema 作为最终权重
```

**为什么**：训练末期权重在最优解附近震荡，取滑动平均相当于"自动集成"，更接近真正的极小值。几乎零额外代价。

### 8.3 Warmdown（学习率收尾）

**做法**：训练总共 20000 步，后 72%（即 step 5600 之后）线性把 LR 降到 0。

```python
if frac >= 1 - 0.72:
    lr_mul = (1 - frac) / 0.72
```

**为什么**：不像 cosine schedule 缓慢下降，线性 warmdown 结尾降得更猛，有研究表明这在**固定预算训练**里更优（让最后几步做精细收尾）。

### 8.4 梯度裁剪 `grad_clip=0.3`

**为什么这么小**：Muon 本身就相当于做了全局归一化，只需要一个小 clip 防爆。普通 AdamW 通常用 1.0。

---

## 第 9 部分：量化（训练完 → 16MB）

### 9.1 GPTQ（Gradient-based Post-Training Quantization）

**问题**：训好的 FP32 / BF16 模型太大，要压缩到低精度（int6、int8）存。

**朴素做法**：`w_quant = round(w / scale) * scale`，每个权重独立量化。问题：误差累积。

**GPTQ 做法**：
1. 用校准数据跑一遍模型，收集每层输入的 **Hessian**（`H = X.T @ X`，其中 X 是该层输入）
2. Hessian 告诉你"如果这个权重改一点，输出变多少"
3. 逐列量化权重：量化当前列后，把产生的误差**反向传播到还没量化的列**，让它们补偿。

```python
# 伪代码
for col in range(n_cols):
    q_col = round(w[:, col] / scale) * scale      # 量化这列
    error = w[:, col] - q_col                      # 误差
    w[:, col+1:] -= error * H_inverse[col, col+1:] # 用 Hessian 引导，让后面列吸收误差
```

**效果**：比朴素量化 loss 低很多，几乎无损。

### 9.2 SDClip（Std-based clipping）

**做法**：量化前先把权重 clip 到 `±k·σ`（σ 是该行标准差）。

```python
scale = clip_sigmas * row_std / (2^(bits-1) - 1)
w_clipped = clip(w, -clip_sigmas*row_std, clip_sigmas*row_std)
```

**为什么**：权重分布通常是长尾的，有极少数大值。如果量化范围让给它们，其他 99% 的权重就没分辨率了。**砍掉极端值**换取普通值的精度，总误差反而更小。

PR #1493 用 `matrix_clip_sigmas=12.85`（矩阵）、`embed_clip_sigmas=20.0`（embedding）。

### 9.3 Mixed Precision：int6 matrices + int8 embeddings

**为什么分开**：
- Matrix（attention/MLP 权重）：数量多，int6 够用
- Embedding：token 表示的核心，更敏感，int8 保精度

### 9.4 Byte Shuffle + Brotli

**Byte Shuffle**：量化后的权重是 int8 数组。直接存会有很多重复的"高字节"（int8 大部分值是小的，高字节接近 0）。把所有高字节放一起、低字节放一起，后续压缩能吃到更高的冗余。

```python
# stride=2
原始: [b0_hi b0_lo b1_hi b1_lo b2_hi b2_lo ...]
shuffled: [b0_hi b1_hi b2_hi ... b0_lo b1_lo b2_lo ...]
```

**Brotli**：比 zstd/lzma 更猛的通用压缩，针对 web 数据设计。对 shuffle 过的整数数组特别有效。

---

## 第 10 部分：评估技巧

### 10.1 Sliding Window Eval

**朴素评估**：把验证文本切成不重叠的 chunks，对每个 chunk 预测所有 token。

**问题**：每个 chunk 的**前几个 token 没有上下文**，loss 偏高。

**Sliding window**：
```
窗口长度 2048，stride 64
窗口1: tokens [0:2048],  只算 [0:2048] 的 loss
窗口2: tokens [64:2112], 只算 [2048:2112] 的 loss
窗口3: tokens [128:2176],只算 [2112:2176] 的 loss
...
```

每个 token 在被预测时都有 2048-64=1984 token 的前文。**同一个 token 只被打分一次**（合法要求），总 loss 更低但公平。

### 10.2 Legal Score-First TTT（Test-Time Training）

**普通 TTT**（**违规**）：边看边改模型权重，用新看到的内容调参后再评分 → 作弊。

**Legal Score-First TTT**（合规）：
1. 把验证集按 chunk 切块（32768 tokens 一块）
2. 对第 i 块：**先**用**当前权重 + 无梯度** 打分 → 累加 loss
3. **然后**在这一块上做 SGD 更新（3 epochs，lr=0.005）
4. 用更新后的权重打分第 i+1 块……

规则关键：**每块都是先评分再更新**，保证每个 token 的评分没用到未来信息。等效于"在线自适应"，是**合法的**。

**为什么有效**：验证集和训练集分布不同，模型在验证阶段边学边评，能适应验证集的风格。

---

## 第 11 部分：编排——这些东西怎么拼在一起

榜首的核心观察是：**单个技术的收益都很小（0.001~0.01 BPB），但叠加起来就 1.1147 → 1.0810**。

工程角度看，这个赛道像：
1. **基座架构**（dim/layer/heads/MLP）：决定参数预算怎么分
2. **残差与注意力细节**（resid_mix, attn_scale, parallel, ln_scale, XSA, QK-gain, softcap, partial RoPE）：十来个小改动各贡献一点
3. **深度循环 + U-Net Skip**：在固定物理层数下提升"有效深度"
4. **Optimizer（Muon）**：让相同架构收敛到更低 loss
5. **EMA + 多组 LR/WD**：训练末期的压榨
6. **GPTQ + SDClip + Brotli**：不损失性能地压到 16MB
7. **Sliding Window + Legal TTT**：在评估阶段再榨一道

**理解了这些模块，就能判断"我加这个改动是替换了哪个模块/和谁配合"**，避免盲目叠加。

---

## 推荐阅读顺序（如果你想深挖某个方向）

1. **Attention 基础** → "Attention Is All You Need"
2. **RoPE** → "RoFormer"
3. **GQA** → Meta "GQA" paper
4. **Muon** → Keller Jordan 的 modded-nanogpt repo
5. **GPTQ** → "GPTQ: Accurate Post-Training Quantization"
6. **Universal Transformer / Depth Recurrence** → DeepMind 论文
7. **TTT** → 原始 TTT 论文（Sun et al.）

---

*最后更新：2026-04-19*

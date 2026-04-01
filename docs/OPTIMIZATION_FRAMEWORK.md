# Parameter Golf 优化框架

> 系统性提升 BPB 的两个维度

---

## 🎯 核心目标

**预测准下一个 token** = 最小化 BPB

这取决于两个独立但互补的维度：

---

## 维度 1：模型的表达能力与学习效果

### 核心问题
> 训练好的模型，能否学到信息间的通用关系？

### 评估指标

| 指标 | 测什么 | 如何测量 |
|------|--------|----------|
| **BPB / Perplexity** | 整体预测准度 | 标准验证集评估 |
| **Error Analysis** | 哪些 token 预测差 | 按 token 类型分组分析 loss |
| **Long-range Dependency** | 能记住多远 | 变长上下文测试 |
| **Calibration** | 概率是否可靠 | Expected Calibration Error |
| **Generalization** | 分布外表现 | 不同领域测试集 |

### 优化思路（可自动迭代）

```
循环 {
    1. 训练模型
    2. 错误分析：哪些 token 预测差？
       ├─ 罕见词 → 调整 tokenizer / embedding
       ├─ 长距离依赖 → 增加 context / 架构改进
       ├─ 特定模式 → 数据增强 / 架构改进
       └─ 数值/日期 → 特殊处理
    3. 针对性改进
    4. 回到 1
}
```

### 具体优化方向

#### A. 架构改进
- [x] **XSA** (Exclusive Self Attention): 去除 self-similarity bias → -2.6%
- [ ] **更深层数**: 用满 16MB 限制
- [ ] **U-Net skip connections**: 跨层信息流
- [ ] **Mixture of Experts**: 稀疏激活增加容量

#### B. 训练技巧
- [x] **QAT 量化**: 训练时感知量化，无损压缩
- [ ] **Muon Optimizer**: 专为 transformer 设计
- [ ] **更长训练**: 5000 → 10000 步
- [ ] **EMA**: 权重平均，更稳定

#### C. 数据/Tokenizer
- [x] **BPE-8192**: 比字节级好 35%
- [ ] **BigramHash**: 输入增强，显式 bigram 信息
- [ ] **数据清洗**: 去噪、去重复

#### D. 初始化
- [ ] **正交初始化**: 更好的训练起点
- [ ] **Spectral embedding init**: 谱初始化

---

## 维度 2：推理时动态学习 (TTT)

### 核心问题
> 如何利用测试时的"第一手信息"？

### 关键洞察

```
标准评估（闭卷考试）：
  模型只能用训练时学到的"通用知识"
  
TTT（开卷考试）：
  模型可以临时学习当前文档的：
  - 词频分布
  - 写作风格
  - 主题术语
  - 局部模式
```

### TTT 方法对比

| 方法 | 原理 | BPB 改进 | 复杂度 |
|------|------|----------|--------|
| **Full Fine-tuning** | 更新所有权重 | 最高 | 最慢 |
| **LoRA TTT** | 只更新低秩适配器 | ~7% | 较快 |
| **Sliding Window** | 只看最近 context | 小 | 最快 |

### LoRA TTT 优化空间

| 参数 | 影响 | 权衡 |
|------|------|------|
| **LoRA rank** | rank 越大，适应能力越强 | 过拟合风险增加 |
| **TTT epochs** | 多轮学习，效果更好 | 时间成本增加 |
| **学习哪些层** | Q/V/LM head？更多？| 需要实验 |
| **学习率** | 高 LR 快速适应 | 可能不稳定 |
| **学习率调度** | 开头猛，后面稳 | 更合理 |
| **chunk 大小** | 多少 tokens 一组 | batch 效率 |

### 高级 TTT 技术

- [ ] **Reptile meta-TTT**: 元学习初始化
- [ ] **Causal TTT**: 只用前文训练，更严格
- [ ] **Multi-Pass TTT**: 多轮扫描（可能不合规）

---

## 两个维度的平衡

```
        ▲ TTT 效果
        │
   高   │    ┌──────────────────┐
        │    │  最优区域：       │
        │    │  好的基础模型      │
        │    │  + 有效的 TTT     │
        │    └──────────────────┘
        │   ╱                    ╲
   低   │  ╱  弱模型+强TTT      强模型+无TTT ╲
        │ ╱   (起点太差)         (浪费TTT机会) ╲
        └─────────────────────────────────────────►
              基础模型质量

最优策略：
  1. 在 16MB 限制内训练最好的基础模型
  2. 用 LoRA TTT 做局部适应
```

---

## 实验 Checklist

### 已完成 ✅
- [x] BPE-8192 tokenizer
- [x] QAT 量化 (1.58-bit)
- [x] XSA (Exclusive Self Attention)
- [x] Embedding 空间分析

### 进行中 🔄
- [ ] LoRA TTT 实现

### 待尝试 📋
- [ ] Muon Optimizer
- [ ] 更长训练 (10000 步)
- [ ] BigramHash 输入增强
- [ ] U-Net skip connections
- [ ] EMA 权重平均
- [ ] 组合最佳技术

---

## 自动化迭代流程（未来）

```python
def auto_optimize():
    while bpb > target:
        # 1. 训练
        model = train(config)
        
        # 2. 错误分析
        errors = analyze_errors(model, val_data)
        
        # 3. 生成改进假设
        hypotheses = generate_hypotheses(errors)
        
        # 4. 选择最有希望的改进
        next_change = select_best(hypotheses)
        
        # 5. 更新配置
        config = apply_change(config, next_change)
        
        # 6. 记录实验
        log_experiment(config, bpb)
```

---

## 当前进度

| 指标 | 值 |
|------|-----|
| 当前最佳 BPB | **1.44** (XSA) |
| 榜首 BPB | **1.08** |
| 差距 | ~25% |
| 主要差距来源 | TTT (~7%) + 其他技术 (~18%) |

---

*创建: 2026-04-01*
*最后更新: 2026-04-01*

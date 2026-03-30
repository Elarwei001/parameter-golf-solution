# 训练技巧

> 同样的模型和数据，训练技巧能让结果差几个点。这些是我们踩坑后的经验。

## 一句话定义

**训练技巧 = 让模型更好地学习的工程细节**

架构决定上限，训练技巧决定能否接近上限。

---

## Learning Rate 调参

### 太大 vs 太小

```
lr 太大:
Loss ─╱╲──╱╲──╱╲─→ 震荡不收敛

lr 太小:
Loss ────────────→ 收敛太慢

lr 合适:
Loss ─╲
       ╲
        ╲────→ 平稳下降
```

### 我们的发现

| lr | 效果 |
|-----|------|
| 1e-3 | 开始震荡 |
| **6e-4** | 最佳 🏆 |
| 3e-4 | 收敛慢 |
| 1e-4 | 太慢 |

**建议**：从 3e-4 开始，每次乘/除 2 来调。

---

## Warmup

### 为什么需要

训练开始时：
- 权重随机初始化
- 梯度方向不稳定
- 大学习率容易"走偏"

### 实现

```python
def get_lr(step, warmup_steps, max_lr):
    if step < warmup_steps:
        # 线性 warmup
        return max_lr * step / warmup_steps
    else:
        # 正常学习率
        return max_lr
```

### 我们的配置

```
total_steps = 2000
warmup_steps = 100 (5%)
```

---

## Gradient Clipping

### 为什么需要

某些 batch 的梯度特别大 → 权重剧烈更新 → 训练崩溃

```python
# 限制梯度范数
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 经验值

| 场景 | max_norm |
|------|----------|
| 稳定训练 | 1.0 |
| 激进训练 | 0.5 |
| 不确定 | 从 1.0 开始 |

---

## Checkpoint 和恢复训练

### 为什么重要

- GPU 可能中断
- 需要增量实验
- 成本控制

### 我们的实现

```python
def save_checkpoint(model, optimizer, step, path):
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': step,
    }, path)

def load_checkpoint(model, optimizer, path):
    ckpt = torch.load(path)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    return ckpt['step']
```

### Modal 上的用法

```bash
# 首次训练
modal run experiment.py --experiment-name "v1" --max-seconds 600

# 继续训练
modal run experiment.py --experiment-name "v1" --resume-from "v1" --max-seconds 1200
```

---

## Batch Size 选择

### Trade-off

| Batch Size | 优点 | 缺点 |
|------------|------|------|
| 大 | 梯度稳定、GPU 利用率高 | 内存大、泛化可能差 |
| 小 | 内存小、正则化效果 | 梯度噪声、慢 |

### 我们的经验

```
A100 80GB:
- batch_size=32, seq_len=1024 ✓
- batch_size=64, seq_len=1024 → OOM

T4 16GB:
- batch_size=8, seq_len=512 ✓
```

### 有效 Batch Size

如果内存不够，用梯度累积：

```python
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## 数据加载优化

### DataLoader 参数

```python
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,      # 并行加载
    pin_memory=True,    # GPU 传输更快
    prefetch_factor=2,  # 预加载
)
```

### 我们踩的坑

```python
# ❌ 错误：每次都从头读文件
for epoch in range(100):
    data = load_all_data()  # 慢！
    train(data)

# ✅ 正确：预加载 + memory map
data = np.memmap('data.bin', dtype='uint16', mode='r')
dataloader = DataLoader(MemMapDataset(data), ...)
```

---

## 混合精度训练

### 为什么用

- FP16 计算更快
- 内存减半
- 大多数场景精度足够

### PyTorch 实现

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    with autocast():  # FP16 前向
        loss = model(batch)
    
    scaler.scale(loss).backward()  # 缩放梯度
    scaler.step(optimizer)
    scaler.update()
```

---

## 我们的完整训练配置

```python
config = {
    # 模型
    "dim": 512,
    "n_layers": 9,
    "n_heads": 8,
    
    # 优化器
    "optimizer": "adamw",
    "lr": 6e-4,
    "weight_decay": 0.1,
    "betas": (0.9, 0.95),
    
    # 训练
    "batch_size": 32,
    "seq_len": 1024,
    "warmup_steps": 100,
    "max_steps": 2000,
    "grad_clip": 1.0,
    
    # 效率
    "mixed_precision": True,
    "compile": True,  # torch.compile
}
```

---

## 常见问题排查

| 症状 | 可能原因 | 解决方案 |
|------|----------|----------|
| Loss 不降 | lr 太小/太大 | 调整 lr |
| Loss 爆炸 | 梯度爆炸 | 加 gradient clipping |
| OOM | batch 太大 | 减小 batch 或用梯度累积 |
| 训练太慢 | DataLoader 瓶颈 | 增加 num_workers |
| 过拟合 | 正则化不足 | 加 weight decay/dropout |

---

## 🚨 成本控制（血泪教训）

### 我们的 $184 事故

```
问题：用 process poll 频繁监控训练
结果：7 分钟花了 $184

原因：
- 每次 poll 返回几千行训练日志
- 日志全部存入会话历史
- 会话膨胀到 58MB
- Claude Opus input $15/M tokens
```

### 解决方案

1. **不要 poll 长任务**
2. **日志写文件，只发摘要**
3. **用 isolated cron 监控**

```python
# ❌ 每 50 步打印
if step % 50 == 0:
    print(f"Step {step}, Loss {loss}")  # 太多！

# ✅ 每 500 步打印
if step % 500 == 0:
    print(f"Step {step}, Loss {loss}")
```

---

*上一篇：[非 Transformer 架构](06-alternative-architectures.md)*

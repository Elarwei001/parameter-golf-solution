# BASELINE_CONFIG.md - 标准实验配置

**所有实验必须使用这套配置，只改模型架构部分。**

## 数据配置 (固定)

| 参数 | 值 | 说明 |
|------|-----|------|
| Dataset | fineweb10B_sp8192 | FineWeb 10B, 8192 vocab |
| Train files | 前 5 个 shard | ~500M tokens |
| Val files | 全部 val shard | ~40M tokens |
| Vocab size | 8192 | |
| Header size | 256 * 4 bytes | 跳过文件头 |

## 训练配置 (固定)

| 参数 | 值 |
|------|-----|
| seq_len | 256 |
| batch_size | 64 |
| steps | 5000 |
| lr | 1e-3 |
| lr_schedule | cosine (warmup=200) |
| weight_decay | 0.1 |
| grad_clip | 1.0 |
| seed | 42 |

## 模型配置 (Baseline)

| 参数 | 值 |
|------|-----|
| dim | 384 |
| n_layers | 20 |
| n_heads | 8 |
| n_kv_heads | 4 |
| 参数量 | ~33M |

## Baseline 结果 (mHC v2, 20层)

| 指标 | 值 |
|------|-----|
| Val Loss | 3.8222 |
| Val BPB | **1.5025** |
| 脚本 | `modal_mhc_v2_deep.py` |
| 日志 | `train_mhc_v2_20layers.log` |

---

## 实验规则

1. **只改模型架构** — 数据、训练超参保持不变
2. **记录所有配置差异** — 如果必须改配置，在实验记录中说明原因
3. **用相同 seed** — 保证可复现
4. **对比 BPB** — 主要指标是 Val BPB

## 代码模板

新实验应该从 `modal_mhc_v2_deep.py` 复制以下部分：
- Image 定义
- Volume 配置
- 数据加载代码
- `get_batch()` 函数
- 训练循环
- LR schedule

只修改模型类定义。

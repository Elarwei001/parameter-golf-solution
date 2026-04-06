# TODO - 待验证思路

---

## 高优先级

- [ ] **dim=544 Scylla + LoRA+TTT 评估** — 等训练完成后跑 LoRA+TTT 评估脚本
- [ ] **TTT 超参调优** — 尝试 rank=16, epochs=3, 更多 docs，看能否进一步降低 BPB

## 中优先级

- [ ] **Alt-attn 消融实验** — 跑一个全 global attention（去掉 alt-attn）的 dim=544 Scylla，对比 mHC 参数曲线是否还有波浪起伏。验证波浪是否由 global/local 交替导致。
- [ ] **dim=448 + Scylla 消融** — 单独验证 Scylla tokenizer 的 BPB 改善（控制 dim 不变，只换 tokenizer）
- [ ] **EMA + SWA** — 加 EMA(0.997) + SWA(every 50)，几行代码，可能直接降 0.5-1% loss
- [ ] **QK-Gain** — 给 Q/K 加可学习标量乘数，初始值 4.0，一个参数 per layer
- [ ] **Late QAT (LR-based)** — 改用 LR schedule 触发 QAT（LR < 0.15 × base_lr 时启用），比当前的 adaptive 更可靠

## 低优先级 / 探索性

- [ ] **GPTQ int6** — 替换 ternary quantization 为 Full Hessian GPTQ int6，需要较大改动
- [ ] **LZMA/Brotli 压缩** — 导出时压缩模型，腾出更多参数空间
- [ ] **Parallel Muon** — 替换 AdamW 为 Muon optimizer
- [ ] **BigramHash** — 添加 n-gram 特征输入
- [ ] **Partial RoPE** — 只对部分 head dims 做 RoPE，释放更多内容维度

---

*最后更新: 2026-04-06*

# Proposal: add-stageb-3step-reflection

## Goal
Adopt the math-style three段式反思（summary → critique → batch update）和严格JSON协议，
让 Stage‑B 指导库更新可控、可审计，避免样本摘要污染，同时保留训练‑free快速迭代能力。

## Why now
- 现 run 输出显示 guidance 被候选 Reason/critic 文本污染，规则不可读。
- 当前反思/critic 宽松解析和兜底策略导致无格式漂移也会落库。
- 参考 `references/youtu-agent/training_free_grpo/math` 的三步模板，可在不改模型权重下获得“以小博大”的稳定规则演进。

## Scope
- 设计并落地三段式反思流程（轨迹摘要→问题批判→经验合并）。
- 全链路严格 JSON 协议；无 JSON 直接视为错误，不落库。
- 反思 eligibility 收紧：仅处理冲突/部分正确样本；其余直接跳过。
- 反思结束对经验去重、压缩、重新编号（G0..Gn）。
- 各阶段结果落盘并可复用（幂等、可断点续跑）。
- Prompt 简洁化：单块输入、最小上下文，便于解析和 token 预算。

## Out of scope
- 不调整底模或训练流程；保持 training‑free。
- 不修改 Stage‑A 摘要或数据契约。

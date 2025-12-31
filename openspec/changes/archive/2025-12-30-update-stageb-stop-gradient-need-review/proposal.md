# Change Proposal — update-stageb-stop-gradient-need-review

## Why

Stage‑B 的 `need_review` 需要被严格定义为 **stop‑gradient 队列**：用于隔离“给定 gt_label 仍无法产出可学习梯度”的样本，以避免污染 guidance 学习闭环。

在该语义下，`need_review` 不再尝试区分“GT 噪声 vs Stage‑A 摘要噪声 vs 推理能力不足 vs 温度/采样不足 vs 起始 prompt 误导”等根因；它们都属于“本轮无法学习（unlearnable）”并统一进入 `need_review`，供后续人工逐一排查。

当前实现已具备：
- reflection 输出 `no_evidence_group_ids`（group_id 级）表达“拿到 gt_label 仍不可学/想不明白”的能力；
- runner 在反思 flush 边界路由 `need_review_queue.jsonl`。

但仍存在与 stop‑gradient 哲学不完全一致的风险点：
1) **梯度候选定义不完整**：仅在 label mismatch 时触发反思，可能漏掉 “rollout 不一致（候选 verdict 混合/低一致性）但最终判断正确” 的样本；这类样本仍可能提供“降低不确定性/增强稳定性”的梯度。
2) **stop‑gradient 样本可能影响 guidance 更新**：反思目前是 bundle 级混合上下文，stop‑gradient case 即便不作为 evidence，仍可能在同一上下文中影响 ops 归纳方向。
3) **梯度样本可能被静默丢弃**：若某个梯度候选既未被判为 stop‑gradient，也未对 `operations` 有任何可审计贡献，则该样本既不进入复核队列，也不产生可用梯度，训练自由优化链路不闭环。

## What Changes

1) **梯度候选（eligible for reflection）扩展**
   - 除 `label_match=False` 外，将 “rollout 不一致（候选 verdict 存在 pass/fail 混合）/低一致性” 以及 `conflict_flag/needs_manual_review` 等不确定性信号视为存在梯度机会，进入 reflection（不等价于 stop‑gradient）。

2) **严格的 group_id 级 stop‑gradient（非粘性）**
   - `need_review` 等价于 stop‑gradient：每个 epoch 都重新判定；历史进入 need‑review 不构成后续 epoch 的黑名单。
   - `need_review` 是唯一的 stop‑gradient 人工复核队列；低一致性/不确定性本身不应被当作 stop‑gradient（只作为学习入口信号与可观测字段）。

3) **两段式反思（Two-pass Reflection）**
   - pass‑1（decision）：仅做 group_id 级 stop‑gradient 判定，输出 `no_evidence_group_ids`。
   - pass‑2（ops）：仅使用 learnable groups（排除 stop‑gradient）生成 `operations`，从机制上保证 stop‑gradient 不会进入 ops 上下文影响规则归纳。

4) **强制闭环：每个梯度候选必须二选一**
   - 对每个进入 reflection 的梯度候选 group：
     - 要么进入 `no_evidence_group_ids`（stop‑gradient）；
     - 要么被至少一条 `operation.evidence` 引用（贡献梯度）。
   - 对于未被覆盖的 group，系统 MUST 进行**下一批次复判/重试**（robust），而不是静默丢弃或立即一刀切写入 need‑review。
   - 为保证终止性与可控成本，引入 per-group per-epoch 的 bounded retry budget；默认：每个 `group_id` 每个 epoch **最多重试 2 次**。重试仅指“再次执行 decision+ops 两段式反思”，不得仅为闭环而重跑 rollout。

5) **规则生命周期反馈隔离 + 缓冲**
   - stop‑gradient 工单 MUST NOT 参与 rules 的 hit/miss 反馈归因；
   - 因 stop‑gradient 判定发生在 reflection 之后，系统需要对反馈做缓冲/延迟提交，避免时序导致的污染。

6) **Prompt 与 initial guidance 作为优化面**
   - reflection prompts 使用两个独立 prompt 文件（decision/ops），明确 two-pass 职责分离，并显式提示 “learnability closure + 未覆盖复判”，以减少 `L\\E`；
   - ops pass 输出可选 `coverage`（learnable/covered/uncovered）用于可观测性；正确性仍由系统计算的集合为准；
   - initial guidance 作为 bootstrap，侧重 mission 定义 + 证据取舍 + 安全默认，避免预先硬编码应由 reflection 学出的潜在规律（例如明确数量阈值）。

7) **严格 evidence + 成本上界**
   - operations 必须提供有效 `evidence`（非空且属于 learnable 集合）；缺失/非法 evidence 的 op 必须拒绝应用，且不得有“默认全量 bundle”回退；
   - 引入 mission-level 的 reflection 调用上限与确定性复判 batch 缩小策略，避免重试导致成本失控；超限样本进入 need-review（reason_code=budget_exhausted）。

## Scope

- 能力：`stage-b-training-free`（runner / reflection prompt&engine / guidance lifecycle / need-review artifacts）。
- 文档：`openspec/specs/stage-b-training-free/spec.md` 对应语义更新；运行时文档在后续实现阶段同步更新。

## Non‑Goals

- 不改变 Stage‑A JSONL 输入/输出契约。
- 不引入外部检索/知识库。
- 不重写 Stage‑B rollout 两行协议或 selection 策略（除非为满足 stop‑gradient 闭环必需）。

## Success Criteria

- 反思输入仅包含“梯度候选”（`label_match=False` 或 rollout 不一致/低一致性），稳定正确样本不进入反思上下文。
- 每个梯度候选 group 在每个 epoch 都被强制判为：
  - stop‑gradient（进入 need‑review），或
  - learnable（进入 `operations[*].evidence` 并驱动 guidance 更新）。
- stop‑gradient group_id 不出现在任何已应用 operation 的 evidence 中，且 stop‑gradient case 不进入 ops pass 上下文。
- need‑review 为非粘性（每个 epoch 重新判），可在后续 epoch 重新变为 learnable。
- stop‑gradient/hard-fail 不参与 hit/miss 归因（通过缓冲/延迟提交实现）。

## Risks & Mitigations

- 风险：两段式反思增加 LLM 调用次数与时延。
  - 缓解：仅对梯度候选触发；并对 decision/ops 分别缓存与复用；支持 batch_size/token_budget 调参。
- 风险：模型输出未严格满足 “覆盖/互斥/证据合法” 不变量（漏写 group_id / evidence 乱填）。
  - 缓解：引入 deterministic validator：对未覆盖 group 进入下一批次复判；对非法 evidence 的 op 拒绝应用并重试。
- 风险：rollout 不一致样本增多导致反思调用增加（成本上升）。
  - 缓解：仅对矛盾/低一致性样本触发反思，并支持 batch_size 与 token_budget 的可配置优化。

# Spec Delta — stage-b-training-free

## MODIFIED Requirements

### Requirement: Stage‑B SHALL 将 need‑review 统一为“人工复核：候选池无任何候选支持 GT”，并同时产出 JSONL 队列与汇总 JSON。
need‑review 的语义 MUST 从“反思前置隔离/噪声治理（quarantine）”调整为“人工复核入口”。系统 MUST 在给定 `gt_label` 后，将“rollout 候选池中 **没有任何候选支持 GT**（`label_match=True` 的候选数为 0）”的 ticket 记录到 need‑review。

need‑review MUST 包含两种形态：
- 流式：`need_review_queue.jsonl`（每条 ticket 一行）
- 汇总：`need_review.json`（run 结束时聚合生成，便于直接阅读与统计）

#### Scenario: 候选池无任何候选支持 GT → 入 need_review_queue
- **GIVEN** 一个 group ticket（Stage‑A per‑image summaries + gt‑label）
- **AND** rollout 产生的候选中没有任何候选能够支持 `gt-label`（所有候选 `label_match != True`）
- **WHEN** Stage‑B 完成该 ticket 的候选生成与路由判定
- **THEN** 系统 MUST 将该 ticket 追加写入 `need_review_queue.jsonl`
- **AND** 记录中 MUST 包含 `ticket_key/group_id/mission/gt_label/pred_verdict/reason_code/reflection_id/reflection_cycle` 等可审计字段
- **AND** need‑review 记录 MUST NOT 影响 inference 的二分类输出协议（最终输出仍为两行二分类且无第三状态词面）。

#### Scenario: 候选池存在支持 GT 的候选 → 不入 need_review_queue
- **GIVEN** 一个 group ticket
- **WHEN** rollout 候选中存在至少一个候选支持 `gt-label`（存在 `label_match=True`）
- **THEN** 系统 MUST NOT 将该 ticket 写入 `need_review_queue.jsonl`。

#### Scenario: Run 结束生成 need_review.json
- **GIVEN** 某个 mission/run 目录下存在 `need_review_queue.jsonl`
- **WHEN** Stage‑B 结束该 mission 的运行（run end）
- **THEN** 系统 MUST 生成 `need_review.json`，其内容为对 `need_review_queue.jsonl` 的确定性聚合（顺序与计数可复现）
- **AND** `need_review.json` MUST 可直接被人工复核工具读取（无需依赖额外上下文文件）。

### Requirement: Stage‑B MUST NOT 将“硬故障”写入 need‑review（人工复核）队列；硬故障 MUST 仅进入 failure/debug 工件与日志。
硬故障包括但不限于：两行协议解析失败、无候选、无有效候选、selection 报错等。它们属于 pipeline/debug 类问题，MUST NOT 污染人工复核队列。

#### Scenario: 解析失败只写 failure，不写 need_review
- **GIVEN** rollout 输出无法解析为合法两行二分类（缺失 Verdict/Reason 或格式错误）
- **WHEN** Stage‑B 处理该 group
- **THEN** 系统 MUST 将原始输出与失败原因写入 `failure_malformed.jsonl`（或等价 failure/debug 工件）
- **AND** 系统 MUST NOT 将该 group 写入 `need_review_queue.jsonl`（因为该问题不属于“反思后仍不可学”的人工复核语义）。

### Requirement: Reflection SHALL 鼓励“无监督式规则学习”，并且 MUST NOT 仅以 `G0` 为唯一标准阻断学习。
当 `pred` 与 `gt-label` 冲突时，reflection MUST 在看到 `gt-label` 后尝试总结可泛化规则（允许“可能解释/假设”），以推动每个 mission 的关注点在运行中逐步学习出来。

约束：
- `gt=fail & pred=pass`：reflection 应优先总结 fail 场景或“关键通过证据缺失/无法确认”的必要条件规则；
- `gt=pass & pred=fail`：reflection 应优先总结通过的例外/特殊情况，或指出哪些负项更可能属于其他 mission、不得干扰本 mission；
- reflection MUST NOT 因为“当前 `G0` 未命中明确负项/正项证据”就直接阻断学习；相反应允许提出可淘汰的假设规则，并交由 rule lifecycle 快速试错。

#### Scenario: gt=fail 且 pred=pass 时产生 fail-side 可学习假设
- **GIVEN** `gt-label=fail` 且 selection 最终 `pred=pass`
- **WHEN** reflection 生成规则变更 proposal
- **THEN** 它 MUST 优先尝试输出“可泛化的 fail-side 假设规则”（例如关键证据缺失/覆盖不足的必要条件）
- **AND** 它 MUST NOT 仅因为“G0 相关明确负项缺失”而强制转为 quarantine 并跳过学习。

#### Scenario: gt=pass 且 pred=fail 时产生 pass-side 例外/特殊情况
- **GIVEN** `gt-label=pass` 且 selection 最终 `pred=fail`
- **WHEN** reflection 生成规则变更 proposal
- **THEN** 它 MUST 优先尝试输出“可泛化的通过例外/特殊情况”或“mission 独立性澄清”的规则
- **AND** 规则 MUST NOT 写成对单个样本的复述，必须满足可泛化与可验证要求。

## ADDED Requirements

### Requirement: Stage‑B SHALL 提供 rule hit/miss 反馈闭环，并支持快速淘汰低质量规则。
当 reflection 应用了一组规则变更后，系统 MUST 在后续运行中对“最近一次变更涉及的规则 keys”进行 hit/miss 反馈，并结合置信度阈值自动清理，以快速剔除“瞎猜规则”。

#### Scenario: 规则应用后，后续正确/错误反馈到最近变更 keys
- **GIVEN** reflection 在某次周期应用了规则变更并产出 `last_applied_rule_keys`
- **WHEN** 随后的 group verdict 与 `gt-label` 一致
- **THEN** 系统 MUST 对 `last_applied_rule_keys` 记一次 hit（提高其置信度）
- **AND WHEN** 随后的 group verdict 与 `gt-label` 不一致
- **THEN** 系统 MUST 对 `last_applied_rule_keys` 记一次 miss（降低其置信度）。

#### Scenario: 低置信度且累计 miss 达阈值的规则被自动清理
- **GIVEN** 某条非 `G0` 的规则满足：
  - `confidence < confidence_drop_threshold` 且 `miss_count >= min_miss_before_drop`
- **WHEN** Stage‑B 执行周期性清理（例如 epoch_end）
- **THEN** 系统 MUST 从 guidance 中移除该规则，并保留可审计快照以支持回滚。

# Spec Delta — stage-b-training-free

## MODIFIED Requirements

### Requirement: Stage‑B SHALL 以 winning candidate 为准进行 need‑review 路由，并在 run 结束生成包含 latest+all 的 need_review.json。
need‑review 的语义保持为“候选支持性驱动的人工复核入口”：系统 MUST 在给定 `gt_label` 后，对“没有支持 GT 的候选（按 has_support 判定）”的 ticket 写入 need‑review。

本变更将 “是否存在支持 GT 的候选” 的判定从 **any-candidate** 调整为 **winning-candidate**：
- 仅当 winning candidate 存在且 `label_match == True` 时，视为“有支持”并避免 need‑review；
- 其它情况视为“无支持”，并路由入队。

同时，`need_review.json` MUST 同时包含：
- `latest_by_ticket`（每个 ticket 最新一条）
- `all_history`（保留历史全量）

#### Scenario: winning 不支持 GT → 入 need_review_queue
- GIVEN 一个 ticket 的 rollout 候选与 selection 结果
- WHEN winning candidate 不存在，或其 `label_match != True`
- THEN 系统 MUST 将该 ticket 追加写入 `need_review_queue.jsonl`
- AND 记录 MUST 包含 `ticket_key/group_id/mission/gt_label/pred_verdict/pred_reason/reflection_id/reflection_cycle/global_step` 等可审计字段。

#### Scenario: winning 支持 GT → 不入 need_review_queue
- GIVEN 一个 ticket
- WHEN winning candidate 的 `label_match == True`
- THEN 系统 MUST NOT 将该 ticket 写入 `need_review_queue.jsonl`。

#### Scenario: run 结束生成 need_review.json（latest+all）
- GIVEN 某 mission/run 目录存在 `need_review_queue.jsonl`
- WHEN Stage‑B 结束该 mission 的运行
- THEN 系统 MUST 生成 `need_review.json`
- AND `need_review.json` MUST 同时包含 `latest_by_ticket` 与 `all_history`
- AND 其聚合与排序 MUST 可复现（同一输入与 seed 下输出稳定）。

### Requirement: Stage‑B MUST NOT 将硬故障或 reflection 解析失败写入 need‑review；它们 MUST 进入独立 debug 工件并可审计。
系统 MUST 将硬故障与 reflection 解析失败分别写入独立 debug 工件（例如 `failure_malformed.jsonl` 与 `reflection_malformed.jsonl`），并 MUST NOT 将这些事件写入 `need_review_queue.jsonl`。

硬故障包括但不限于：两行协议解析失败、无候选、无有效候选、selection 报错等。reflection 解析失败包括：reflection JSON 解析/生成失败、截断 JSON、schema 不满足等。

#### Scenario: rollout/selection 硬故障只写 failure_malformed
- GIVEN 某 ticket 发生 `no_candidates|format_error|no_valid_candidates|selection_error`
- WHEN Stage‑B 处理该 ticket
- THEN 系统 MUST 将失败原因写入 `failure_malformed.jsonl`
- AND 系统 MUST NOT 将该 ticket 写入 `need_review_queue.jsonl`。

#### Scenario: reflection JSON 解析失败只写 reflection_malformed
- GIVEN 某次 reflection 的 JSON 解析/生成失败
- WHEN 系统记录该失败
- THEN 系统 MUST 将失败写入 `reflection_malformed.jsonl`（含 mission/异常摘要/原始片段截断等）
- AND 系统 MUST NOT 将该失败写入 `need_review_queue.jsonl`。

## ADDED Requirements

### Requirement: Stage‑B SHALL 引入基于 global_step 的反馈归因窗口（256 steps），仅对困难票进行 hit/miss 反馈，并以 reflection_id 聚合后批量写回。
系统 MUST 支持 `feedback_window_steps = 256` 的归因窗口：仅在窗口内、且仅对困难票/冲突票，记录 hit/miss 反馈。

credit MUST 先按 `reflection_id` 聚合，再回流到该次 reflection 更新涉及的 rule keys；hit/miss MUST 在内存累积，并在每次 reflection flush（或窗口过期）时批量写回到 guidance repo。

#### Scenario: 仅窗口内的困难票参与反馈
- GIVEN 已存在一个 applied 的 `reflection_id` 与其涉及的 rule keys
- WHEN 处理某 ticket 的 `global_step` 落在 `[applied_step, applied_step + 256]` 且该 ticket 被判定为困难票
- THEN 系统 MUST 记录该 ticket 的 hit/miss 到该 `reflection_id` 的聚合计数
- AND WHEN `global_step` 超出窗口或发生下一次 reflection flush
- THEN 系统 MUST 将聚合计数批量写回到 guidance repo。

### Requirement: Stage‑B SHALL 采用规则生命周期路线 B：新增规则不自动 +1 hit，仅更新/merge 既有规则才视为 reinforcement。
系统 MUST 避免对新增规则施加“写入即命中”的先验偏置；新增规则的 hit 仅来自后续窗口反馈。

#### Scenario: 新增 rule key 不自动 hit+1
- GIVEN reflection applied 新增了一个此前不存在的 rule key
- WHEN guidance repo 写入该 key 的 metadata
- THEN 系统 MUST NOT 因“新增写入”而对该 key 执行 `hit_count +1`。

#### Scenario: update/merge 既有 key 允许 hit+1
- GIVEN reflection applied 对既有 key 执行 update 或 merge
- WHEN guidance repo 更新该 key 的 metadata
- THEN 系统 MAY 对该 key 执行一次 reinforcement（`hit_count +1`），并重新计算 confidence。

### Requirement: Stage‑B SHALL 提供 fail-first 例外短语护栏并记录审计信息。
系统 MUST 支持 `fail_first_exception_phrases`（子串匹配），允许少量显式例外绕过 fail-first 覆盖；触发例外时 MUST 记录审计字段以便回溯。

#### Scenario: 命中例外短语时绕过 fail-first
- GIVEN selection 原本将触发 deterministic fail-first 覆盖
- WHEN reason 文本命中 `fail_first_exception_phrases` 任一子串
- THEN 系统 MUST 允许绕过 fail-first 覆盖
- AND MUST 在 selection/trajectory/日志中记录“例外触发”的审计信息（至少含命中的短语与触发原因）。

### Requirement: Stage‑B SHALL 产出可审计的复核/排除原因桶，并使 metrics.jsonl 的 include/exclude 口径可追溯到该原因桶。
系统 MUST 为每个 ticket 产出明确的原因桶（例如 `review_bucket`），用于区分：
- 真正的人工复核入口（need-review）
- pipeline 硬故障（failure_malformed）
- reflection 解析失败（reflection_malformed）
- 低一致性/建议复核但仍可学习的票（low_agreement / selection_manual_review）

系统 MUST 同时提供一个可直接用于指标计算的布尔值（例如 `exclude_from_metrics`），其取值 MUST 由原因桶可复现地派生。

#### Scenario: metrics exclude/include 由原因桶派生
- GIVEN 一批 ticket 的 outcome 记录包含 `review_bucket` 与 `exclude_from_metrics`
- WHEN 生成 `metrics.jsonl`
- THEN 系统 MUST 使用 `exclude_from_metrics` 决定“exclude_*”口径中的样本集合
- AND 每条 metrics 记录 MUST 能够反查到各原因桶的数量统计（例如按 bucket 计数）。

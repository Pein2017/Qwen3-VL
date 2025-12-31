# Design — update-stageb-feedback-window-and-observability

## 1) 目标语义回顾

本变更的中心目标是把 Stage‑B 的“人工复核入口”与“学习闭环反馈”统一到可审计、低噪声的路径上：
- **need-review**：只包含“候选池无任何候选支持 `gt_label`（`label_match=True` 为 0）”的票；宁可多一些，但不得混入硬故障。
- **feedback（hit/miss）**：只对“困难票/冲突票”在有限窗口内反馈，并以 `reflection_id` 聚合，降低 credit assignment 噪声，帮助找到更好的规则。

推理输出协议保持不变：严格两行二分类（`Verdict/Reason`），无第三状态词面。

## 2) 关键对象与粒度

- `ticket`：`GroupTicket`（一组 Stage‑A summaries + mission + gt_label），业务上等价于“工单”。
- `case/record`：`ExperienceRecord`（ticket + 候选 + signals + winning_candidate + global_step）。
- `reflection update`：一次 reflection 的 applied 结果，唯一标识为 `reflection_id`，会变更一组 rule keys。

本变更刻意将“归因单元”提升为 `reflection_id`，避免把所有后续效果都归到“最后一次触碰的 rule-key”。

## 3) need-review 路由：只看 winning candidate 的支持性

### 3.1 动机

现状是“候选池任一候选支持 GT 即视为可解释”，门槛偏硬：只要采样里偶然出现一个支持 GT 的候选，即使 winning 很不稳定、reflection 也无法提出可学规则，也不会进入 need-review，导致人工覆盖不足。

### 3.2 新策略

- 将 “has_support” 的判定从 **any-candidate** 改为 **winning-candidate**：
  - 仅当 `winning_candidate` 存在且其 `signals.label_match == True` 时，视为“有支持”；
  - 其它情况视为“无支持”，并入 need-review（保持语义为“候选池无任何候选支持 `gt_label`”，但把判定口径切换为 winning-candidate）。

### 3.3 兼容性

- 仍保留 `need_review_queue.jsonl` / `need_review.json` 文件名与主字段（`ticket_key/group_id/mission/gt_label/pred_*`）。
- `manual_review_queue.jsonl` 继续仅作为兼容占位；复核入口以 need-review 为准。

## 4) need_review.json：latest + all

### 4.1 需求

人工复核有两类常见视图：
- **latest**：每个 ticket 只看最新一条（方便“当下处理”）。
- **all**：保留历史全量（方便回溯“为什么反复进队/是否已被修复”）。

### 4.2 建议结构（单 mission 文件）

保留现有聚合字段（如 `items`/`by_reason_code`），并新增两个字段（最小破坏）：
- `latest_by_ticket`: `{ticket_key: entry}`（entry 取该 ticket 的最新一条，按 `global_step/epoch_step/reflection_cycle` 的确定性排序规则选取）
- `all_history`: `[{...}, ...]`（等价于原 `items`，但字段名更明确；可与 `items` 共存一段时间）

排序必须可复现：同一输入、同一 seed、同一 artifacts 生成顺序应稳定。

## 5) hit/miss 反馈降噪：归因窗口 + 困难票反馈 + reflection_id 聚合

### 5.1 当前噪声来源

runner 侧用 `last_applied_rule_keys` 做 last-touch 归因会引入三类噪声：
- 规则更新频繁，导致“后续效果”被错误归因到最近一次更新；
- 一次 reflection 更新多个 keys，导致 credit 在多个 key 之间难以分配；
- 将全部票纳入反馈会让“易票”淹没“困难票”的信号，且低一致性票会放大噪声。

### 5.2 最小实现：窗口化的 last-touch（但按 reflection_id 聚合）

确认参数：`feedback_window_steps = 256`，以 `global_step` 为计数单位。

建议做法：
1) 每次 reflection **applied** 时记录一个 `ActiveFeedbackContext`：
   - `reflection_id`
   - `applied_rule_keys`
   - `start_global_step`（以 applied 时刻为起点）
   - `end_global_step = start + feedback_window_steps`
2) 在处理每个 ticket 的 selection 后，如果满足：
   - 当前存在 active context；
   - `global_step` 落在 `[start, end]`；
   - ticket 被判定为“困难/冲突票”（见 5.3）；
   - 则把该票的 `label_match` 计入该 `reflection_id` 的 hit/miss 计数（内存累积，不落盘）。
3) 在下一次 reflection flush、或 active context 过期时，批量把该 `reflection_id` 的累计 hit/miss 回写到 guidance repo（映射到该 reflection 涉及的 rule keys）。

该策略仍是近似，但窗口化与困难票过滤可显著降噪，并保持实现复杂度可控。

### 5.3 “困难票/冲突票”定义（降噪版）

为降低噪声与实现难度，本变更采用保守定义（满足任一即视为困难票）：
- `selection.label_match == False`（pred 与 gt 不一致）
- selection 触发 deterministic fail-first 覆盖（或等价审计标记）
- `conflict_flag == True`
- `low_agreement == True` 且（错例或 fail-first 相关）

注：不把“低一致性但最终正确”的票默认纳入困难票，避免把采样噪声当作学习信号。

## 6) 规则生命周期（路线 B）

原则：允许 reflection 大胆提出假设，但必须“快进快出”；同时避免新增规则初始置信度偏高。

路线 B 规则：
- 更新/merge 既有 key：允许 `hit_count +1`（视为 reinforcement）。
- 新增 key：不自动 +1 hit；仅在后续窗口反馈中获得 hit。

## 7) 工件与标记的可审计性

### 7.1 `reflection_malformed.jsonl`

reflection JSON 解析/生成失败属于“debug 事件”，需要独立工件记录以便提示词/解析器迭代，但不得混入 need-review。

建议最小字段：
- `mission`, `ticket_key`（若可定位）、`reflection_cycle`, `reflection_id`（若有）
- `error_type`, `error_message`, `raw_snippet`（截断）

### 7.2 `in_manual_review` 的语义拆清

建议将 “是否排除/是否复核” 从单一布尔值拆成可审计原因桶（可同时保留旧字段兼容）：
- `review_bucket`: `none|need_review|failure_malformed|reflection_malformed|low_agreement|selection_manual_review|...`
- `exclude_from_metrics`: bool（由 bucket 派生，避免歧义）

## 8) fail-first 例外护栏（护栏里外）

增加 `fail_first_exception_phrases`（子串匹配）作为最小开关：
- 若 fail-first 将覆盖 verdict，但 reason 命中例外短语，则允许绕过覆盖（并在 artifacts/日志中留下审计字段）。

该机制用于少量已知例外快速止血；长期应通过 reflection 学习可泛化“例外规则”来替代手工短语。

## 9) 风险与后续演进

- 归因窗口仍是近似：如果噪声仍然较大，下一步再考虑把 credit 升级到“multi-touch/分桶 uplift”或更强的反事实评估（非本次目标）。
- need-review 覆盖率上升：通过 bucket 化统计监控来源，必要时收紧困难票定义或增加采样预算。

# Change Proposal — update-stageb-feedback-window-and-observability

## Why
- 当前 `need_review_queue.jsonl` 的语义是“候选池无任何候选支持 `gt_label`（`label_match=True` 为 0）”，而 `metrics.jsonl` 的 `fp/fn` 是判定正确性统计，两者天然不一致：容易出现“错例很多但 need‑review 为空/或错例不多但 need‑review 偏多”，运营侧难以定位“哪些票真的需要人工复核”。
- hit/miss 反馈目前以 `last_applied_rule_keys` 的 last-touch 方式逐票归因，存在明显 credit assignment 噪声：规则更新频繁或一次 reflection 修改多个 key 时，容易出现“好规则被坏票拖累 / 坏规则被好票抬高”的震荡。
- `in_manual_review` 当前同时承载“硬故障/低一致性/需要人工复核”等多种语义，导致指标与工单分流难以审计；同时缺少 `reflection` 解析失败的独立工件（需要与 need-review 区分）。
- deterministic fail-first 对少量已知例外缺少最小化“护栏里外”开关，导致误拒难以快速止血。

## What Changes
- **need-review 路由更保守（宁可多）**：从“候选池任一候选支持 GT 则不入队”调整为“仅当 **winning candidate** 支持 GT 才不入队”；即只看 `selection.label_match`（或等价字段）。
- **need-review 汇总增强**：`need_review.json` 同时输出 `latest_by_ticket` + `all_history`（仍保持确定性排序），便于“看最新”与“追溯历史”两种工作流并存。
- **反馈降噪（最小实现优先）**：
  - 引入 `global_step` 归因窗口（确认：`feedback_window_steps = 256`）。
  - 仅对“困难票/冲突票”反馈（降噪版，减少实现难度）。
  - 将 credit 的聚合单元提升为 `reflection_id`（先按 reflection update 聚合评估，再回流到该次更新涉及的 rule keys）。
  - hit/miss 在内存累积，每次 reflection flush 时批量写回 guidance repo（减少 I/O 与抖动）。
- **规则生命周期（路线 B，更保守）**：`apply_reflection` 时仅“更新/merge 既有规则”视为 reinforcement（`hit_count +1`）；新增规则不自动 +1 hit。
- **可观测性工件补齐**：
  - 新增 `reflection_malformed.jsonl`：记录 reflection JSON 解析/生成失败（debug 工件，不进入 need-review）。
  - 规范化 `in_manual_review`/复核相关标记与原因桶，使 `metrics.jsonl` 可审计（need-review / failure_malformed / low_agreement / selection_manual_review 等）。
- **fail-first 例外（最小护栏）**：新增可配置 `fail_first_exception_phrases`（子串匹配），允许少量显式例外绕过 fail-first 覆盖。

## Impact
- 影响能力：`stage-b-training-free`（need-review 路由与汇总、guidance 生命周期、metrics/工件可观测性、fail-first 配置护栏）。
- 影响代码面：`src/stage_b/runner.py`、`src/stage_b/reflection/engine.py`（工件写入/错误分桶）、`src/stage_b/io/guidance.py`、`src/stage_b/scoring/selection.py`、相关测试与文档。
- 影响工件（每 mission/run 目录）：
  - `need_review_queue.jsonl`（语义更保守，覆盖率提升）
  - `need_review.json`（新增 latest+all 结构，保留确定性）
  - `reflection_malformed.jsonl`（新增）

## 目标与约束（必须满足）
1) **只把“真的需要人工复核”的票写进 need-review**：need-review 仍限定为“候选池无任何候选支持 `gt_label`（`label_match=True` 为 0）”，不得混入 rollout/解析/selection 等硬故障。
2) **推理输出协议不变**：最终输出仍为严格两行二分类（`Verdict/Reason`），不得引入任何第三状态词面。
3) **口径统一**：`pass` 视为 positive、`fail` 视为 negative，用于 fp/fn 统计与审计字段解释。
4) **最小改动优先**：先交付“归因窗口 + 困难票反馈 + 批量写回 + 新工件/字段清理”的最小闭环，再考虑更复杂的归因模型。

## Non-goals
- 不改变 Stage‑A 契约与摘要规范。
- 不引入外部检索/知识库。
- 不把 Stage‑B 改造成训练/微调流程（保持 training‑free）。

## Risks & Mitigations
- 风险：need-review 覆盖率上升造成人工压力。
  - 缓解：仅对“候选池无任何候选支持 `gt_label`”入队；并通过 `need_review.json` 的按原因桶统计监控增长来源。
- 风险：归因窗口仍然是近似（非因果）。
  - 缓解：窗口限制 + 困难票过滤 + 以 `reflection_id` 聚合，显著降低 last-touch 噪声；后续如仍不稳再升级到更强的 credit 机制（非本次目标）。

## Success Criteria
- `need_review.json` 同时包含 `latest_by_ticket` 与 `all_history`，且聚合顺序/计数可复现。
- `need_review_queue.jsonl` 覆盖率按预期上升，但不混入硬故障；硬故障仅进入 `failure_malformed.jsonl` / `reflection_malformed.jsonl`。
- hit/miss 反馈仅在 `global_step` 窗口内、且只对困难票生效；写回为批量、可审计。
- `in_manual_review`/原因桶的语义可解释，`metrics.jsonl` 的 include/exclude 口径可追溯到具体桶。

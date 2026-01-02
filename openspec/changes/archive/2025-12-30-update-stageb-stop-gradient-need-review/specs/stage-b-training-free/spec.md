# Spec Delta — stage-b-training-free

## MODIFIED Requirements

### Requirement: Stage‑B SHALL perform two-pass LLM-based reflection to update mission guidance from gradient candidates.
Stage‑B 的 reflection MUST 仅观察 **梯度候选**（gradient candidates），并采用 two-pass 结构：
1) decision pass：对每个梯度候选 group 做 stop‑gradient 判定（输出 `no_evidence_group_ids`）；
2) ops pass：仅使用 learnable groups（排除 stop‑gradient）生成 strict JSON operations（用于更新 guidance）。

梯度候选 MUST 至少包含以下两类（group_id 级）：
- `label_match=False` 的错例；
- rollout 不一致/低一致性：候选 verdict 同时包含 `pass` 与 `fail`，或 selection 信号标记为 `low_agreement`（阈值来源于任务配置，例如 `manual_review.min_verdict_agreement`）。

此外，以下一致性/冲突信号 MUST 作为梯度候选入口条件之一（等价触发）：
- `conflict_flag=true`，或
- `needs_manual_review=true`（无论是否最终会保留 manual_review 队列工件）。

信号定义（为避免“名义一致、语义漂移”）：
- `vote_strength`（group-level）：基于 format_ok 候选的多数票比例，范围 `[0.0, 1.0]`。
- `low_agreement`（group-level）：当 `vote_strength < manual_review.min_verdict_agreement` 时为 true；阈值来源 MUST 仅来自该 mission 的 Stage‑B 配置。
- `label_match`（group-level）：最终选中 verdict（含任何 deterministic override 后）是否等于 `gt_label`。
- `conflict_flag`（group-level）：当 `label_match=false` 时为 true（等价于“该 group 当前为错例/冲突样本”）。
- `needs_manual_review`（group-level）：用于表达“即便 label_match=true 也存在显著不确定性/风险”的信号；可由 selection policy 或确定性规则触发，但必须在 artifacts 中可观测（并不得隐式改变 `label_match/conflict_flag` 的含义）。

#### Scenario: Final verdict correct but rollouts contradict → still enters decision pass
- **WHEN** 一个 ticket 的最终 selection 与 `gt_label` 一致（label_match=True），但 rollout 候选 verdict 同时包含 `pass` 与 `fail`。
- **THEN** 该 ticket MUST 被视为梯度候选并进入 reflection decision pass。

#### Scenario: Stable-correct ticket is excluded from reflection
- **WHEN** 一个 ticket 的最终 selection 与 `gt_label` 一致（label_match=True）。
- **AND WHEN** 该 ticket 不存在 rollout 矛盾（不存在 `pass` 与 `fail` 混合）且未触发 `low_agreement` 等一致性告警。
- **THEN** 该 ticket MUST NOT 出现在 reflection decision pass 的输入 CASES 中。

#### Scenario: Conflict/manual-review flags trigger decision pass eligibility
- **WHEN** 一个 ticket 在候选信号中出现 `conflict_flag=true` 或 `needs_manual_review=true`。
- **THEN** 该 ticket MUST 被视为梯度候选并进入 reflection decision pass。

### Requirement: Stage‑B SHALL route `need_review` as stop-gradient-only after reflection.
`need_review` MUST 等价于 stop‑gradient：表示“给定 `gt_label` 后仍不可学习/无法提出可验证假设”的 tickets。

This change explicitly migrates `need_review` away from the legacy “label-suspect-only” interpretation. Under stop-gradient semantics, `need_review` intentionally does not distinguish root causes (GT noise vs Stage‑A noise vs prompt/model limits vs sampling/temperature); it only encodes “unlearnable after seeing GT” for later manual investigation.

need‑review MUST 满足：
- **严格 group_id 级**：以 reflection decision pass 的 `no_evidence_group_ids` 为 per-ticket stop‑gradient 判定来源；
- **非粘性**：每个 epoch 都重新判定；历史进入 need‑review 不得构成后续 epoch 的黑名单；
- **断梯度**：stop‑gradient ticket MUST NOT 驱动 guidance 修改（不得进入 ops pass 输入，也不得作为已应用操作的 evidence）。

#### Scenario: Non-sticky stop-gradient — can re-evaluate in later epochs
- **WHEN** 某 group 在 epoch=1 被判定为 stop‑gradient 并进入 need‑review。
- **THEN** 在 epoch=2 的反思中，系统 MUST 允许该 group 被重新判定为 learnable（不因历史 need‑review 而强制继续 stop‑gradient）。

### Requirement: Stage‑B MUST treat `need_review` as the only stop-gradient human review queue (manual review is signal-only).
Stage‑B MUST treat `need_review_queue.jsonl` / `need_review.json` as the only human-review queue that implies **stop-gradient** and exclusion from guidance learning.

Stage‑B MUST NOT route “低一致性/不确定性” cases (e.g., `low_agreement`, `conflict_flag`, `needs_manual_review`) to a separate queue in a way that implies stop-gradient. Instead, these cases MUST:
- remain eligible as gradient candidates for reflection (for learning), and
- be surfaced via deterministic signals/fields in exported artifacts (e.g., selection/trajectory warnings), not by stopping gradients.

#### Scenario: Low-agreement cases are learnable and not stop-gradient by default
- **WHEN** a ticket is flagged as `low_agreement` / `needs_manual_review` but decision pass does not classify it as stop-gradient.
- **THEN** the ticket MUST NOT be added to `need_review_queue.jsonl`.
- **AND THEN** the ticket remains eligible to contribute to ops evidence and guidance updates.

## ADDED Requirements

### Requirement: Stage‑B MUST isolate stop-gradient cases from guidance updates via two-pass reflection.
Stage‑B MUST ensure stop‑gradient tickets cannot influence guidance updates by construction:
- decision pass MAY see all gradient candidates;
- ops pass MUST only include learnable groups (`G \\ S`) and MUST NOT include any stop‑gradient group.

#### Scenario: Stop-gradient group never appears in ops pass evidence
- **WHEN** decision pass outputs `no_evidence_group_ids` containing `g_stop` for a batch.
- **AND WHEN** Stage‑B invokes ops pass to generate guidance operations.
- **THEN** the ops pass input MUST NOT include `g_stop`.
- **AND THEN** any applied operation MUST NOT contain `g_stop` in its `evidence`.

### Requirement: Stage‑B MUST enforce learnability closure with robust re-evaluation and bounded retries.
Stage‑B MUST enforce learnability closure per decision/ops cycle, using the following sets:
- `G`: decision pass input gradient candidates
- `S`: stop‑gradient group_ids from decision pass (`no_evidence_group_ids`)
- `L = G \\ S`: learnable candidates (ops pass input)
- `E`: union of `operations[*].evidence` after validation (from ops pass)

Stage‑B MUST enforce:
- **Disjointness**: `S ∩ E == ∅`.
- **Closure**: for each epoch, every gradient candidate MUST end up as either:
  - stop‑gradient (in `need_review`), or
  - learnable contributor (in `E`, i.e. contributes to some validated operation evidence).

If some learnable candidates are not covered (`L \\ E`), Stage‑B MUST re-queue them for a later (typically smaller) decision/ops cycle within the same epoch, with a deterministic bounded retry budget. Once the retry budget is exhausted, the group MUST be routed to `need_review` for that epoch.

Retry budget semantics:
- A “retry” is an additional **decision+ops two-pass reflection cycle** for a `group_id` within the same epoch.
- Retries MUST reuse existing rollout/selection artifacts for that group in the epoch, and MUST NOT re-run rollout solely to satisfy closure.
- Default retry budget MUST be `2` retries per `group_id` per epoch, unless overridden by configuration.

Cost caps (to prevent cost runaway):
- Stage‑B MUST implement a mission-level cap on total reflection calls per epoch (counting both decision and ops calls). If the cap is exceeded, remaining pending grad-candidates MUST be routed to `need_review` with an auditable reason_code indicating budget exhaustion.
- Stage‑B MUST implement a deterministic “retry batch shrink” policy for uncovered `L\\E` groups. Default: on retry attempt `k` (1-based), retry batch size MUST be `max(1, floor(reflection.batch_size / (2**k)))`, and grouping order MUST be stable (e.g., by `group_id` sort).

#### Scenario: Uncovered learnable group is retried and eventually routed
- **WHEN** ops pass returns operations whose validated evidence covers only a subset of learnable candidates.
- **THEN** the uncovered groups MUST be scheduled for re-evaluation in the next reflection cycle.
- **AND THEN** if a group exceeds the configured retry budget within the epoch (default `2`), it MUST be routed to `need_review`.

### Requirement: Stage‑B MUST treat missing/invalid `operations[*].evidence` as invalid operations (no default evidence fallback).
To preserve `S∩E==∅` and `L==E` invariants, Stage‑B MUST apply strict validation to `operations[*].evidence`:
- For any operation that contains an `evidence` field, it MUST be a non-empty list of `group_id` strings, and MUST be a subset of the ops pass input learnable set `L`.
- For any operation that omits `evidence` or provides an empty/invalid evidence list, the operation MUST be considered invalid and MUST NOT be applied.
- Stage‑B MUST NOT implement any fallback behavior like “missing evidence ⇒ default to all bundle groups”.

Invalid operations MUST trigger robust behavior:
- the invalid operation is rejected (not applied);
- the affected learnable groups remain uncovered and MUST be retried via the closure mechanism (bounded by retry budget / cost caps).

#### Scenario: Evidence missing does not default to bundle
- **WHEN** ops pass emits an `update` operation without `evidence`.
- **THEN** Stage‑B MUST treat the op as invalid and MUST NOT apply it.
- **AND THEN** uncovered groups MUST be scheduled for retry (or routed to need_review if budgets exhausted).

### Requirement: Stage‑B MUST refine reflection prompts to maximize hypothesis search while enforcing evidence coverage.
Stage‑B MUST treat reflection prompts as part of the optimization surface and ensure the prompts explicitly support two-pass reflection.

Stage‑B MUST provide **decision-pass** and **ops-pass** reflection prompt templates as **two separate prompt files**, with the following intent:
- decision pass prompt MUST instruct the model to treat `gt_label` as the reference answer and output per-`group_id` stop-gradient decisions via `no_evidence_group_ids` (and MUST NOT require guidance operations);
- ops pass prompt MUST instruct the model to produce only guidance operations from learnable groups and include `operations[*].evidence` as group_id lists.

To reduce “未覆盖 learnable(L\\E)” probability, the ops pass prompt MUST explicitly state that the system enforces learnability closure and will re-evaluate uncovered groups in later cycles; therefore, the model SHOULD ensure every learnable group_id in the ops pass input is referenced by at least one validated `operations[*].evidence`.

To encourage discovering latent rules (e.g., quantity/frequency patterns) without hard-coding mission-specific heuristics, the ops pass prompt MUST instruct the model to explore hypotheses along multiple axes, including:
- 全局/局部视角取舍；
- 子判断拆解与组合（AND/OR 结构）；
- 频次/数量一致性与多图互证；
- “关键证据缺失 ⇒ 更保守判定”的可淘汰假设。

The prompts MUST also include guardrails to prevent overfitting/copying:
- output rules MUST be generalizable “条件→结论” logic, not sample paraphrases;
- MUST forbid copying Stage‑A object-chain formats or embedding group_id/image indices in rule text.
- Only `G0` is read-only; all `G1+` experiences MUST be considered mutable by reflection (add/update/delete/merge allowed).

#### Scenario: Ops pass prompt drives full evidence coverage
- **WHEN** ops pass receives learnable groups `L` as input CASES.
- **AND WHEN** the prompt explicitly requires evidence coverage for every learnable group (or it will be retried).
- **THEN** the model’s operations SHOULD reference all `L` in `operations[*].evidence`, reducing `L \\ E`.

### Requirement: Stage‑B MUST support an extended ops-pass JSON schema for explicit coverage reporting.
Stage‑B MUST allow ops pass to return an extended strict JSON object that includes an optional `coverage` object for observability and robustness.

If present, `coverage` SHOULD contain:
- `learnable_group_ids`: echo of the ops pass input group_ids (`L`)
- `covered_group_ids`: union of group_ids referenced by `operations[*].evidence`
- `uncovered_group_ids`: `learnable_group_ids \\ covered_group_ids`

The system MUST treat `coverage` as advisory:
- if `coverage` is missing, the system MUST compute `L`, `E`, and uncovered sets itself;
- if `coverage` is present but inconsistent with computed sets, the system MUST log a validation warning and still use computed sets as source of truth.

#### Scenario: Coverage object is used for observability without changing correctness
- **WHEN** ops pass returns operations and an optional `coverage` object.
- **THEN** Stage‑B MUST validate correctness using computed sets (`L`, `E`) and not rely on `coverage` alone.
- **AND THEN** Stage‑B SHOULD log `coverage` to aid diagnosis of `L\\E` and retry behavior.

### Requirement: Stage‑B MUST provide learnability-friendly mission seed guidance via `initial_guidance`.
Stage‑B MUST define a mission seed guidance snapshot (“initial guidance”) that bootstraps rollout and reflection without overfitting.

For each mission in `initial_guidance`, the guidance MUST include:
- `G0`: the mission definition / hard constraints, treated as read-only by reflection.
- At least one additional experience that describes evidence selection policy (e.g., global-vs-local usage, multi-image corroboration, conservative fallback when key evidence is missing), expressed as generalizable logic.

To preserve learnability, `initial_guidance` SHOULD avoid encoding mission-specific latent thresholds/rules that are better learned by reflection (e.g., explicit quantity cutoffs); instead, keep the seed guidance focused on evidence discipline and safe defaults.

#### Scenario: Initial guidance is minimal and does not pre-encode latent quantity rules
- **WHEN** a mission starts with `initial_guidance` and encounters a learnable quantity/frequency pattern.
- **THEN** the seed guidance SHOULD not hard-code the pattern as a fixed threshold.
- **AND THEN** the pattern is expected to emerge through reflection operations backed by evidence.

### Requirement: Stage‑B MUST buffer rule hit/miss feedback until stop-gradient decisions are known.
Stage‑B MUST NOT let stop‑gradient tickets affect rule hit/miss attribution. Because stop‑gradient is decided after reflection, Stage‑B MUST buffer pending hit/miss feedback and only commit feedback for tickets that are confirmed learnable contributors (non-stop-gradient and contributing to `E`). Hard-fail tickets MUST also be excluded.

#### Scenario: Stop-gradient ticket does not contribute to rule feedback
- **WHEN** a ticket is processed and would normally contribute a hit/miss update to recently-applied rule keys.
- **AND WHEN** the ticket is later classified as stop‑gradient in decision pass for that epoch.
- **THEN** the buffered feedback for that ticket MUST be dropped and MUST NOT update rule hit/miss counts.

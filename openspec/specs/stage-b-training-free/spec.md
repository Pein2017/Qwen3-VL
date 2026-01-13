# stage-b-training-free Specification

## Purpose
Define the training-free Stage-B pipeline for group/ticket verdicts in **rule-search mode only**: prompt-only rollouts, rule proposer + metric-gated updates, and deterministic artifact logging.
## Requirements
### Requirement: Stage-B SHALL expose a fail-fast `run_all` entry point
Stage-B tooling SHALL provide a Python entrypoint `run_all(config, ...)` that orchestrates ingest → rollout → rule-search and raises validation errors before inference when configuration or inputs are invalid.

#### Scenario: run_all rejects invalid log level
- **GIVEN** a log_level outside {debug, logging, warning}
- **WHEN** `run_all` is invoked
- **THEN** it raises ValueError before launching inference

### Requirement: Stage-B SHALL generate group verdict candidates via prompt-only rollouts under a strict two-line binary contract.
Stage-B MUST generate one or more candidates per ticket using mission guidance plus Stage-A summaries (without exposing GT labels to the rollout prompt). The decoded output MUST be strictly two lines:
- `Verdict: 通过|不通过`
- `Reason: <single-line Chinese rationale>`
The final output MUST NOT contain any third-state wording (e.g., review/待定/证据不足).

#### Scenario: Rollout produces parseable candidates per ticket
- **WHEN** Stage-B runs rollout for a mission with a configured decode grid and `samples_per_decode`.
- **THEN** each candidate response MUST be parseable into `(verdict, reason)` while preserving the raw text for auditing.
- **THEN** candidate-level metadata MUST include decode parameters, `candidate_index`, and `ticket_key="{group_id}::{label}"`.

### Requirement: Stage-B SHALL run rule-search as the only supported execution mode.
Stage-B MUST execute the rule-search loop by default: baseline rollout on a train pool → proposer emits 1-N rule candidates → A/B gate on train pool → optional eval-pool auditing → apply only gated candidates.

Stage-B MUST ALSO support a baseline-only audit execution that skips proposer/reflection and gating when explicitly requested (e.g., CLI `--jump-reflection` or YAML `jump_reflection: true`).

#### Scenario: Baseline-only audit skips proposer/reflection
- **WHEN** Stage-B runs with `--jump-reflection` (or `jump_reflection: true` in config).
- **THEN** Stage-B MUST run a baseline rollout and export baseline audit artifacts.
- **THEN** Stage-B MUST NOT call proposer/reflection or apply any guidance updates.

#### Scenario: Default rule-search behavior is unchanged
- **WHEN** Stage-B runs without `--jump-reflection`.
- **THEN** Stage-B MUST execute the full rule-search loop with proposer + gate + optional eval audit.

### Requirement: Stage-B SHALL export deterministic rule-search artifacts.
Stage-B MUST write rule-search artifacts under `{output.root}/{run_name}/{mission}/` including:
- `rule_candidates.jsonl` (proposed ops with gate metrics)
- `benchmarks.jsonl` (accepted ops with baseline/after metrics)
- `rule_search_hard_cases.jsonl` and `rule_search_candidate_regressions.jsonl` (audit trails)

#### Scenario: Rule-search artifacts are written by rank 0 only
- **WHEN** Stage-B runs with multiple GPUs.
- **THEN** rollout MUST be ticket-parallel while rank 0 writes artifacts exclusively.

### Requirement: Stage-B SHALL optionally export rule-search distillation chat logs after convergence.
When `stage_b_distillation.enabled=true` and rule-search reaches early-stop, Stage-B MUST sample `distill_size` tickets from the input pool, run low-temperature rollouts, and export ChatML conversations to `distill_chatml.jsonl` (or `log_chatml_path` when provided).

#### Scenario: Distill export after rule-search early stop
- **WHEN** rule-search triggers early-stop and distillation is enabled.
- **THEN** the system MUST sample `distill_size` tickets (seeded for reproducibility), run low-temperature rollouts, and export ChatML with system/user prompts plus a two-line assistant verdict.

### Requirement: Stage-B reflection SHALL run a three-stage JSON-only pipeline (summary → critique → batch update) with cached artifacts.
Stage-B SHALL split reflection into summary、critique、batch-update 三个有序阶段，每阶段只接受/产生严格 JSON，并通过缓存避免重复生成。
#### Scenario: Reflection executes for an eligible bundle
- WHEN a reflection bundle is built, THEN Stage-B MUST execute three ordered stages:
  1) **Summary stage**: compress each bundle into a JSON artifact (one file per bundle) containing group_ids, guidance_step, selected candidate verdict/reason, critic summary/critique, and deterministic signals.
  2) **Critique stage**: feed summaries + current experiences into a minimal prompt that returns a JSON array of operations with `option` in {"add","modify","merge"}, `experience` text, and optional source ids.
  3) **Batch update stage**: merge the operations, emit a JSON plan (with provenance) and a preview guidance file, then apply the plan only after successful parse.
- Each stage MUST write artifacts under `{run}/{mission}/reflection_cache/` and SHALL reuse existing artifacts on rerun if present and well-formed, avoiding recomputation.
- Any non-JSON, truncated, or unparsable output SHALL mark the stage as failed, record `ineligible_reason=generation_error`, and MUST NOT mutate guidance.

### Requirement: Stage-B reflection SHALL restrict updates to conflict/partial-correct bundles and skip others without mutation.
反思 SHALL 只处理存在冲突或“部分正确”信号的批次，其余批次 MUST 直接 noop，不得改写 guidance。
#### Scenario: Bundle eligibility is evaluated
- WHEN computing reflection eligibility, THEN the bundle is **eligible** only if at least one condition holds: (a) selected candidate label_mismatch, (b) mixed label_match values within the bundle (partial-correct), or (c) conflict_flag/needs_manual_review is true for any candidate.
- WHEN none of the conditions hold, THEN reflection MUST short-circuit to `action=noop` with `ineligible_reason=non_conflict_bundle`, no LLM calls, and no guidance change.
- WHEN eligibility is true but the critique stage returns zero operations or invalid JSON, THEN reflection MUST log `ineligible_reason=generation_error` and leave guidance unchanged.

### Requirement: Stage-B guidance updates SHALL deduplicate, compact, and renumber experiences after each batch update.
每次批量更新后 SHALL 对经验文本去重、压缩并重新编号，保证指导库短小、无重复且键连续。
#### Scenario: Applying a batch update plan
- WHEN a batch update plan is accepted, THEN Stage-B SHALL:
  - Normalize experience text (trim, collapse whitespace), drop duplicates, and merge any `merge` references before persisting.
  - Reindex experiences densely as `G0..G{n-1}` in stable order (existing order preserved for kept entries; new additions appended), updating metadata accordingly.
  - Reject any plan that would leave the experiences map empty or contains text that matches Stage-A style image summaries (e.g., count patterns like "×1", "标签/").
  - Persist both the compacted guidance and a snapshot before mutation.

### Requirement: Stage-B reflection prompts SHALL remain minimal, single-block, and stop-token bounded.
反思阶段的 prompt SHALL 保持单块、简洁，并通过 stop token 约束输出格式。
#### Scenario: Building prompts for summary and critique stages
- WHEN constructing reflection prompts, THEN the system MUST avoid chat-style multi-turn wrappers and instead send a single text block per stage, containing only the required structured inputs (experiences block + summaries block).
- Prompts MUST specify stop tokens to prevent multiline/JSON bleed-through (e.g., `\n\n`, ``` , `{`, `[`), and SHALL forbid the model from emitting markdown fences.
- The prompt instructions MUST explicitly require JSON arrays with fixed keys and MUST disallow free-text commentary or sample-level copies of Stage-A summaries.

### Requirement: Reflection stages SHALL persist and reuse per-stage artifacts for audit and idempotency.
每个反思阶段的产物都 SHALL 落盘且可复用，以支撑审计与幂等 rerun。
#### Scenario: Rerunning reflection for the same run directory
- WHEN reflection is invoked again for a mission/run where `reflection_cache` already contains valid summary/critique/batch_update artifacts, THEN the pipeline SHALL reuse them (read-only) and skip regeneration.
- If any cached artifact is missing or invalid, ONLY that stage SHALL regenerate; earlier valid artifacts remain intact.
- All artifacts SHALL be written with deterministic filenames (include epoch, reflection_cycle) and SHALL be referenced from `reflection.jsonl` so auditors can trace which files fed each guidance change.

### Requirement: Stage-B SHALL export baseline audit artifacts in jump_reflection mode.
When jump_reflection is enabled (`--jump-reflection` or config `jump_reflection: true`), Stage-B MUST write baseline audit artifacts under `{output.root}/{mission_name}/{output.run_name}/`:
- `baseline_metrics.json`
- `baseline_ticket_stats.jsonl`
- `baseline_wrong_cases.jsonl`

#### Scenario: Baseline audit artifacts are written by rank 0 only
- **WHEN** Stage-B runs with multiple GPUs in jump_reflection mode.
- **THEN** rollout MUST be ticket-parallel while rank 0 writes baseline artifacts exclusively.

### Requirement: Stage-B runner SHALL support a Teacher distillation mode (on by default) that logs chatml conversations without changing training-free semantics.
Stage-B runner SHALL expose a configuration flag for a Teacher distillation mode (default **enabled** in canonical configs) that can switch to an alternative checkpoint（例如 Qwen3-32B 文本模型）做 rollout，同时仍遵守 training-free 约束（不在该流程中更新任何模型权重），并在收敛后的最后一轮 rollout 额外写出 chatml 对话日志供后续 SFT 使用。

#### Scenario: Distillation mode uses a Teacher checkpoint, runs until guidance stops changing, and preserves training-free guarantees
- GIVEN a Stage-B config with `stage_b_distillation.enabled: true` (default) and `model.model_name_or_path` pointing to a Teacher checkpoint（例如 `model_cache/models/Qwen/Qwen3-32`）
- WHEN the Stage-B runner continues epochs until the mission guidance no longer receives any updates (reflection applies zero operations in an epoch)
- THEN it SHALL treat that guidance-stable epoch as the **final distill epoch**, use the Teacher checkpoint only for forward passes（rollout/selection），without performing any optimizer steps or checkpoint updates
- AND it SHALL continue to emit standard training-free artifacts（`trajectories.jsonl`, `selections.jsonl`, `guidance.json`, `reflection.jsonl`）under `{output.root}/{run_name}/{mission}/`
- AND it SHALL additionally emit a distillation log file（例如 `distill_chatml.jsonl`）containing the selected verdict per group from that final epoch.

### Requirement: Distillation logs SHALL capture complete chatml-style conversations aligned with existing chat datasets.
When Stage-B distillation mode is enabled, the runner SHALL record conversations in a shape that can be consumed directly as `dataset: chat` with `template: chatml` (consistent with coig_cqia) so subsequent SFT only needs to point fusion configs at the generated JSONL files.

#### Scenario: Logging the selected Teacher verdict from the convergence epoch as a single-turn chatml conversation
- GIVEN Stage-A summaries for a group (`per_image` dict), mission name, the **converged** guidance text (the epoch where no guidance updates occurred), and the Teacher-selected `{verdict, reason}` pair for that epoch
- WHEN distillation logging is enabled
- THEN the runner SHALL construct a `messages` array containing exactly:
  - a `system` message that encodes Stage-B instructions（Verdict/Reason 两行协议、mission focus、guidance 片段）;
  - a `user` message that summarizes the ticket context（mission、历史 label、按图片编号串联的 Stage-A 摘要文本，以及必要的 task description）;
  - a single `assistant` message whose content corresponds to the **selected** Teacher Verdict/Reason output for that ticket（保持两行协议形式）。
- AND the runner SHALL write a JSONL record for each logged verdict containing `{group_id, mission, label, messages}` only (no extra decode/epoch metadata), where `messages` is compatible with the existing `chatml` template（roles限定在 `system|user|assistant`，content 为字符串），无需后处理即可用于 `dataset: chat`。

### Requirement: Distillation logging SHALL not alter the semantics or schema of existing Stage-B artifacts.
Enabling distillation mode SHALL be additive-only: standard training-free Stage-B artifacts must remain backward compatible so that downstream evaluation and reflection workflows continue to function unchanged.

#### Scenario: Distillation mode coexists with standard trajectories and selections
- GIVEN a Stage-B run with `stage_b_distillation.enabled: true`
- WHEN the run completes
- THEN the system SHALL:
  - write `trajectories.jsonl`, `selections.jsonl`, `guidance.json` and `reflection.jsonl` under the mission directory with the same schema and semantics as before;
  - write a fresh `distill_chatml.jsonl` for the convergence epoch, **overwriting any existing file** in that mission directory on rerun;
  - avoid introducing new mandatory fields into existing artifacts that would break current readers; any distillation-specific metadata SHALL live in the new log file(s) or optional fields that existing tooling can ignore safely.
- AND if distillation logging fails（e.g., due to disk full or schema bug），the runner MAY log a warning and continue producing standard Stage-B artifacts, but MUST NOT corrupt or partially overwrite existing guidance/selections.

### Requirement: Stage-B inference SHALL inject domain knowledge for BBU vs RRU as a read-only block.
Stage-B MUST include a domain knowledge block in the system prompt for inference. The block SHALL be derived from the domain pack dataclasses (BBU/RRU), MUST be treated as read-only scaffolding, and MUST NOT be modified by reflection or written back into guidance files.

#### Scenario: Domain block is appended to system prompt
- **GIVEN** Stage-B runs with domain resolved to `bbu`
- **WHEN** the system prompt is built for rollout
- **THEN** the prompt contains the BBU domain knowledge block
- **AND THEN** the block is not part of `guidance.experiences` and is excluded from reflection updates

---

### Requirement: Stage-B SHALL resolve domain deterministically with validation.
Stage-B MUST resolve the domain using config only, with deterministic precedence:
1) `config.domain_map[mission]` (if configured)
2) `config.default_domain` fallback
Unknown or missing domains MUST raise a validation error before prompt construction.

#### Scenario: Domain resolved from config mapping
- **GIVEN** a config mapping for the ticket’s mission
- **WHEN** Stage-B resolves the domain
- **THEN** it selects the mapped domain and proceeds with the corresponding pack

#### Scenario: Missing domain fails fast
- **GIVEN** no matching mapping and no default domain configured
- **WHEN** Stage-B resolves the domain
- **THEN** it fails with a clear error stating the required config sources

### Requirement: Stage-B prompts tolerate single-line irrelevant summaries
Stage-B prompt assembly SHALL describe irrelevant summaries without imposing a line-count requirement. Any summary containing the literal `无关图片` SHALL be treated as irrelevant whether it appears as a single-line summary or as the second line of a two-line summary.

#### Scenario: Single-line irrelevant summary
- **GIVEN** Stage-A summaries include a value that is exactly `无关图片`
- **WHEN** Stage-B prompt text is assembled
- **THEN** the prompt only states that `无关图片` marks an irrelevant image and does not require a two-line format.

#### Scenario: Two-line summary with irrelevant second line
- **GIVEN** a summary rendered as two lines where line 2 is `无关图片`
- **WHEN** Stage-B prompt text is assembled
- **THEN** the prompt treats the image as irrelevant and does not imply a format violation.

### Requirement: Stage-B SHALL drop summary headers and pass through payloads
Stage-B summary ingestion SHALL ignore any `<DOMAIN=...>, <TASK=SUMMARY>` header line and forward the remaining payload as-is. No schema validation is enforced; JSON and non-JSON payloads are both allowed.

#### Scenario: Header dropped for summary parsing
- **GIVEN** a Stage-A summary with a two-line header + JSON payload
- **WHEN** Stage-B ingests the summary
- **THEN** the header line is discarded and only the payload line is forwarded downstream unchanged

#### Scenario: Dataset key tolerated
- **GIVEN** a summary payload containing a `dataset` key
- **WHEN** Stage-B ingests the summary
- **THEN** Stage-B passes it through unchanged without error

#### Scenario: Non-JSON payload tolerated
- **GIVEN** a summary payload that is not valid JSON
- **WHEN** Stage-B ingests the summary
- **THEN** Stage-B passes it through unchanged without error

### Requirement: Stage-B SHALL fail fast on review-state inputs
Stage-B SHALL treat any third-state wording as invalid input rather than sanitize or downgrade it; only binary verdicts remain allowed end-to-end.
#### Scenario: Reject review markers in Stage-A summaries or rollout candidates
- **WHEN** Stage-B ingestion, rollout parsing, or reflection sees any review marker (e.g., legacy third-state wording such as 待定、证据不足) in Stage-A summaries, verdict lines, or reasons
- **THEN** Stage-B MUST treat the ticket as invalid input and raise a validation error before decoding candidates
- **AND** Stage-B MUST NOT attempt to sanitize or rewrite the review marker; it simply rejects the sample
- **AND** Stage-B smoketests and unit tests MUST cover this rejection path to ensure no third-state plumbing remains


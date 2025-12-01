# stage-b-training-free — Delta for add-stageb-3step-reflection

## ADDED Requirements
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

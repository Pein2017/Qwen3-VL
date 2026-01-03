# Proposal: Adopt Training-Free Stage-B LLM Reflection Loop

> Status & Phases (Unified): This change (2025-11-03) now subsumes 2025-11-10-tighten-stage-b-reflection as Phase 2 of the same specification. Phase 1 is the baseline training-free loop; Phase 2 adds hardening (merge op, prompt evidence completeness, atomic persistence fix, edit budgets, eligibility policy, export metadata). See specs/stage-b-training-free/spec.md for the unified requirements and the “Youtu-Agent alignment and gaps” section.

- Phase 1 (baseline, origin 2025-11-03): Training-free reflection loop with numbered experiences, deterministic signals, eligibility on selected-mismatch or all-wrong, strict JSON parsing, and single-process orchestration.
- Phase 2 (hardening, origin 2025-11-10, merged here):
  - Add `merge` operation with `merged_from` provenance and stable `G<number>` IDs; forbid empty removals.
  - Fix atomic write ordering: temp → atomic rename → snapshot copy → prune.
  - Introduce CriticEngine (LLM-based, strict JSON) for per-candidate evaluations; embed `summary`/`critique` (and optional `root_cause`/`issues`/`uncertainty_note`) in trajectories; pool/deduplicate `candidate_ops` as suggestions for reflection.
  - Enforce budgets: `reflection.max_operations`, `reflection.change_cap_per_epoch` with telemetry/logging.
  - Add `reflection.eligibility_policy` (default `selected_mismatch_or_all_wrong`, optional `contradictions_only`) and optional Verifier hook.
  - Enrich exports with `eligible`, `ineligible_reason`, and `warnings: [str]`.
  - Add guardrail toggles: `reflection.apply_if_delta` (uplift gating) is configurable and `reflection.rapid_mode: true` may disable gating for fast iteration (production defaults keep guardrails on).

References: `/references/youtu-agent` (README, train.py, main.py, math/web experience builders). Migration: the 2025-11-10 directory will be archived/removed after tasks complete and references are cross-linked here.


## Problem Statement
- Stage-B currently assumes GRPO-style gradient updates, yet our deployment case lacks reliable per-image ground truth and has noisy group verdicts.
- We now understand that guidance refreshes must play the role of `optimizer.step()` for the frozen model, but today the pipeline only snapshots raw rules and never promotes new cues systematically.
- Without a disciplined, data-driven guidance promotion loop the system drifts: good cues are missed, noisy labels can poison guidance, and operators have no consistent rubric to review changes.
- There is no mechanism to quarantine unsalvageable tickets (Stage-A gaps or suspect GT labels), so repeated runs waste compute and silently export low-trust verdicts.
- Post-integration review surfaced critical regressions: label signals regress when verdicts are already normalized, reflection gating ignores the winning candidate and skips all-wrong bundles, guidance updates can delete the last cue, and snapshot writes are not atomic.

## Current Behavior & Gaps
- `src/stage_b/` prepares GRPO datasets (`load_stage_a_for_grpo`), prompts, and reward hooks intended for gradient-based updates in ms-swift.
- No persistent experience base exists; Stage-B rollouts are ephemeral and we cannot reuse high-quality verdicts or failure cases.
- Judging only compares model outputs to noisy human labels; there is no programmable judge, verifier plugin, or confidence calibration.
- Retrieval or few-shot priming from prior runs is unsupported, limiting inference-time guidance.
- Mission guidance updates are entirely manual; there is no cue mining, evidence aggregation, or label-trust weighting to treat guidance changes as deliberate optimization steps.
- Operators cannot distinguish between “model needs better guidance” and “input/label is bad” because there is no fail-fast validation or exception queue.
- Deterministic signals treat any verdict that is not the Chinese string `"通过"` as a failure, so English `"pass"` outputs silently degrade label-match metrics.
- Reflection eligibility looks at bundle-wide diversity, never the selected candidate, and cannot escalate “all predictions wrong” situations; reflection therefore stalls exactly when guidance is needed most.
- Applying a `remove` operation can delete the last remaining experience entry, leaving guidance empty and breaking prompt construction on the next batch.
- Snapshot rotation uses second-resolution timestamps and in-place `replace`, risking clobbered history and partially-written guidance files when multiple updates arrive in the same second.
- Reflection still accepts truncated JSON blocks because stop-string cropping cuts off closing braces; the parser quietly falls back to heuristic text parsing instead of failing fast.

- Replace the current Stage-B codepath with a **training-free, LLM-centred loop** modelled after Youtu-Agent’s training-free GRPO pattern but hardened for noisy Stage-A summaries and GT labels. The frozen checkpoint stays fixed; all improvement comes from repeatedly sampling candidates, judging against ground-truth labels, and letting an LLM synthesise guidance updates (our “optimizer step”). Legacy GRPO scaffolding is retired or shimmed so only one Stage-B implementation remains.
  1. **Fail-fast Ingestion & Run Directory**: ingest Stage-A outputs, prompt guidance JSON, and mission config into `{output.root}/{run_name}/{mission}/` with deterministic `stats.json`, shuffled batches (batch size 32 or 64, configurable in YAML), and per-step rollout logs. Missing files, unwritable directories, or inconsistent guidance steps abort the run immediately. Shuffle data each epoch, skip last incomplete batch.
  2. **Experiences Injection (Global per Mission)**: `guidance.json` stores `step`, `updated_at`, and an `experiences` dict rendered as numbered snippets. Writes must be atomic, snapshot rotation must use microsecond-resolution filenames, and the repository enforces “never empty” invariants so removing the last cue is rejected.
  3. **Rollout Sampler**: run the tuned model with a configurable decode grid (temperature, top_p, etc.) and capture K three-line candidates per group. Any malformed responses or prompts missing the expected context fail the step so bad data never propagates.
  4. **Deterministic Signals (Minimal Set)**: compute `label_match`, `confidence`, and optionally `self_consistency` while normalising verdict strings (`"pass"`/`"fail"`, `"通过"`/`"不通过"`). The signals `candidate_agreement` and `label_trust` are deprecated and always set to `None` per training-free Stage-B design (LLM-first philosophy). Signals log warnings on missing metrics but never misclassify due to localisation.
  5. **Three-Stage Reflection (CriticEngine → Screener → Consolidator)**: after each batch, run (a) a **CriticEngine** that emits strict-JSON per-candidate fields `{summary, critique, root_cause?, issues?, candidate_ops?, uncertainty_note?}` (template validated; low temperature; top_p ~0.9; bounded max_new_tokens; per-field length caps; at most 6 candidates per group), (b) a **CritiqueScreener** that checks eligibility using the selected candidate plus an “all wrong” override, and (c) a **GuidanceConsolidator** that consumes critic fields and turns them into structured `{action, summary, critique, operations[...]}` JSON. Reflection calls must return syntactically valid JSON; stop-string truncation is fatal. Applied operations merge into guidance, capture provenance, and maintain at least one experience entry.
  6. **Selection & Export**: select final verdicts with the current experiences, prioritising GT-aligned candidates. Log warnings for problematic tickets but continue processing. Export trajectories, selections, and CriticEngine fields (`summary`, `critique`, optional `root_cause`/`issues`/`uncertainty_note`), plus reflection logs, so downstream QC can identify tickets that may need review.
- Update configs/docs to reflect the reflection-centric workflow, including the single `StageBPipeline.run_all()` entry point, run-directory hygiene expectations, and the new summary/critique artifacts.

- Fully refactor `src/stage_b/`: remove legacy GRPO-only modules, rebuild the package around the training-free runner (dataset ingestion reuse, sampler, LLM reflection engine, pipeline class), and ensure every step validates prerequisites before touching the model. Preserve public API surface (`__init__`, `run_all`) by pointing to new implementations.
- Extend configuration schemas under `src/config/` for mission rule files, sampler settings, verifier plugins, guidance hashes, and fail-fast toggles (primarily booleans that default to "on").
- Provide an optional wrapper script (`scripts/stage_b_run.sh`) that instantiates the pipeline and calls `run_all()`; no multi-step CLI is required.
- Update documentation (`docs/training/REFERENCE.md`, `docs/DATA_AND_DATASETS.md`, new how-to under `docs/`) to explain the training-free + reflection lifecycle, exception queue workflow, and how `StageBPipeline.run_all()` is invoked in automation.
- Add tests covering rule file validation, guidance hash generation, deterministic scoring + trust weighting, reflection application, exception queue writes, and export integrity.

## Non-Goals
- No gradient updates or ms-swift GRPO integration; LoRA/adapter fine-tuning stays out of scope, and the previous GRPO reward hooks are retired unless reimplemented atop the new pipeline.
- No SQL/NoSQL databases, vector indexes, or embedding models.
- No retrieval-augmented prompting in the initial implementation.
- Stage-A generation remains unchanged aside from optional “second-look” hook triggered by judge flags.
- We will not replace existing historical datasets; instead, we augment them with judge-calibrated metadata.

## Validation Plan
- Unit tests for:
  - Mission prompt guidance repository load/save and step tracking.
  - Deterministic signal extraction + trust scoring, ensuring malformed trajectories log warnings and continue processing.
  - Reflection guardrails (single proposal, direct application, warning logs) with noisy-label fixtures.
  - CLI validation hooks ensuring each `--step` refuses to run when required artifacts are absent.
- Integration smoke:
  - Instantiate `StageBPipeline` and call `run_all()` on a 10–20 ticket slice using real Stage-A summaries; verify warnings are logged and processing continues.
- Baseline comparison:
  - Diff selections vs greedy inference, record pass/fail deltas and warning counts, and publish the stats in docs.
- Manual inspection:
  - Review `output_post/stage_b/<mission>/<run>/` manifests (trajectories, selections, reflection) for schema compliance and matching guidance steps.
  - Confirm each applied reflection in `reflection.jsonl` has evidence group IDs audit trail.
- API verification:
  - Document how to call `StageBPipeline.run_all()` (or the wrapper script) and how the pipeline logs warnings for operators to review.

## Interfaces & Contracts

### Data Contracts
- Stage-A JSONL (input):
  - `group_id: str`, `mission: str`, `label: "pass"|"fail"`, `per_image: {image_i: str}` plus `stage_a_complete: bool` and optional `label_source`, `label_timestamp`. Missing fields block ingestion.
- Mission experiences file (`guidance.json`):
  - `{ mission: { step: int, updated_at: iso8601, experiences: {"G0": str, "G1": str, ...} } }`. Missions absent from the file cause the run to abort; experiences must be a non-empty dict.
- Run directory manifests:
  - `stats.json`, `epoch_{k}/shuffled_data.jsonl`, and `step_{n}/rollout.jsonl` use deterministic naming for reproducibility. Re-runs produce fresh artifacts; no guaranteed resume semantics in v1.
- Trajectory log (`output_post/stage_b/<mission>/<run>/step_{n}/trajectories.jsonl`):
  - `group_id`, `mission`, `candidate_index`, `decode` ({temperature, top_p, max_new_tokens, seed}), `response_text`, `guidance_step`, `created_at`, `verdict`, `reason`, `confidence`, `signals` ({label_match: Optional[bool], self_consistency: Optional[float], candidate_agreement: Optional[bool] (deprecated, always None), confidence: Optional[float], label_trust: Optional[float] (deprecated, always None)}), `critic?: {summary: str, critique: str, root_cause?: str, issues?: list[str], candidate_ops?: list[object], uncertainty_note?: str}`, `warnings: [str]`.
- Selection export (`selections.jsonl`):
  - `group_id`, `mission`, `verdict`, `reason`, `confidence`, `label_match`, `selected_candidate`, `guidance_step`, `reflection_change` (nullable), `eligible: bool`, `ineligible_reason?: str`, `warnings: [str]`. Note: `label_trust` is deprecated and not exported.
- Reflection log (`reflection.jsonl`):
  - `reflection_id`, `mission`, `proposal` ({action, summary, critique, operations[{op, key, text, rationale, evidence, merged_from?}], evidence_group_ids, uncertainty_note}), `eligible: bool`, `applied: bool`, `guidance_step_before`, `guidance_step_after`.

### Config Additions (YAML → `src/stage_b/config.py`)
- `StageBConfig` (new):
  - `stage_a_paths: list[str]` (absolute or relative paths to Stage-A JSONL files); empty lists are invalid.
  - `model: {model_name_or_path: str, torch_dtype?: str, device_map?: str}`.
  - `guidance: {path: str, retention: int}` controls mission experiences storage.
  - `output: {root: str, run_id?: str, fail_if_exists: bool=true}` ensures run directories are unique and writable.
- `sampler: {group_batch_size: int (32 or 64), samples_per_decode: int, grid: list[{temperature: float, top_p: float, max_new_tokens: int, seed?: int, stop?: list[str]}], enforce_three_line: bool=true}` enforces prompt/response structure before decoding.
- `critic: {enabled: bool=true, prompt_path: str, temperature: float, top_p: float, max_new_tokens: int, max_candidates: int<=6, summary_max_chars: int, critique_max_chars: int, prefilter?: {top_k?: int}}` controls CriticEngine behavior (strict JSON output; template validated at init; low temperature 0.1–0.3; top_p≈0.9; bounded tokens).
- `signals: {store_confidence: bool, enable_consistency: bool, weights?: {label_match: float, self_consistency: float, confidence: float}}` configures minimal deterministic metrics for tie-breaking and context. Note: `candidate_agreement` and `label_trust` are deprecated and always `None`; the minimal signal set consists of `label_match`, `confidence`, and optionally `self_consistency`. Signals do not replace reflection.
  - `reflection: {prompt_path: str, batch_size: int, apply_if_delta: float, allow_uncertain: bool, max_operations: int, change_cap_per_epoch: int, eligibility_policy: "selected_mismatch_or_all_wrong"|"contradictions_only"|"contradictions_or_all_wrong", all_wrong_strategy: "reflect_diagnose"|"manual_review", rapid_mode?: bool=false}` (these keys MUST be set in YAML; recommended values: `max_operations=3`, `change_cap_per_epoch=10`) for structured, incremental updates. The `eligibility_policy` option `contradictions_or_all_wrong` triggers reflection when there are contradictions (mixed `label_match` values) OR all candidates are wrong (all `label_match=False`). The `all_wrong_strategy` knob controls behavior for all-wrong groups: `reflect_diagnose` (default) allows reflection, `manual_review` short-circuits to a noop proposal flagged for manual review. Uses the same checkpoint as Stage-B model, passed directly to `ReflectionEngine` for in-process model reuse.
- `selection: {policy: "label_first", tie_break: "confidence"|"temperature"}`.
- `runner: {epochs: int, shuffle_seed: int}` orchestrates deterministic shuffling (shuffle each epoch, skip last incomplete batch).

### Public Entry Points
- Python module: `python -m src.stage_b.runner --config /abs/path/to/stage_b_training_free.yaml --step all [--log-level {debug|logging|warning}]`.
- Script wrapper: `bash scripts/stage_b_run.sh /abs/path/to/stage_b_training_free.yaml` (internally uses `conda run -n ms`).

## Runner Flow (Training-Free + Reflection Cycle)
1. **Ingest**: `ingest_stage_a` loads Stage-A JSONL, validates mandatory fields (including `stage_a_complete`), hydrates the mission experiences, and materialises the run directory skeleton; any missing artifact aborts the command.
2. **Rollout**: `RolloutSampler` builds prompts by formatting the current experiences as a single text block (e.g., `"[G0]. ...\n[G1]. ..."`) and prepending it, verifies that the constructed prompt contains the expected experiences, and generates K three-line candidates per group across the decode grid.
3. **Signal Extraction**: `compute_signals` records deterministic metrics (`label_match`, `self_consistency`, `candidate_agreement`, `confidence`, `label_trust`) with centralised verdict normalisation; groups with missing metrics log warnings and continue processing.
4. **Reflect**: `reflect_on_batch` packages winners/losers + signals + CriticEngine fields (per-candidate strict JSON), prompts an LLM (same checkpoint as Stage-B) to return a strictly-typed incremental edit payload (`{action: "refine"|"noop", summary?: str, critique?: str, operations: [{op: "upsert"|"remove"|"merge", key?: str, text?: str, rationale?: str, evidence: [...], merged_from?: ["Gk", ...]}], evidence_group_ids: [...], uncertainty_note?: str}`), and evaluates it against holdout uplift before committing. The engine enforces budgets (`max_operations` per proposal; `change_cap_per_epoch` per epoch), supports configurable eligibility policies (default `selected_mismatch_or_all_wrong`), and has no text-fallback path (invalid JSON is rejected and logged). Operations merge into the existing experiences map (no wholesale replacement). A preview guidance map is built to compute post-uplift metrics; proposals that fail the `apply_if_delta` threshold or carry `uncertainty_note` when `allow_uncertain=false` are logged but not applied.
   - Template validation: ReflectionEngine validates at initialization that the prompt template mentions the allowed operations and the K budget, and includes all required placeholders; missing items MUST raise a configuration error.
5. **Select**: `select_for_group` prioritises GT-aligned candidates while logging warnings for problematic tickets and stamping guidance step + reflection id onto each record.
6. **Export**: `export_trajectories`/`export_selections` capture outcomes with warning flags so operators can identify tickets that may need review.

## Reflection Workflow
   - Mission experiences JSON stores an `experiences` dict: `{ "step": int, "updated_at": iso8601, "experiences": {"G0": "...", "G1": "...", ...} }`. Experiences are stored internally as numbered entries (matching Youtu-Agent's pattern) and formatted as a single text block using `"\n".join([f"[{i}]. {e}" for i, e in experiences.items()])` when prepending to prompts (e.g., `"[G0]. ...\n[G1]. ..."`).
   - Each reflection cycle generates an **experience bundle** containing winners, conflicting candidates, deterministic signals, trust summaries, and Stage-A deficiency notes. The reflection LLM consumes the current experiences (rendered as numbered snippets) plus the bundle, then emits `{action, summary?, critique?, operations[], evidence_group_ids, uncertainty_note?}`. When `action="refine"`, the operations list encodes minimal edits: `upsert` (add/update) or `remove`. Evidence references the contributing groups (`第N组` / actual ids) and optional rationales explain why the change is needed.
   - **Incremental mid-epoch updates**: Proposals are previewed against holdout uplift using a cloned guidance map. Only when the uplift delta meets `apply_if_delta` (or no holdout is available) does the engine persist the operations to `guidance.json`, incrementing the step counter and recording per-entry metadata (`reflection_id`, evidence sources, rationale, timestamp). Ineligible proposals (`noop`, insufficient uplift, or blocked uncertainty) leave the guidance untouched but are logged for operator review.

## Outcome Targets (Acceptance Criteria)
- Accuracy: ≥ +3–5% absolute vs greedy baseline per mission on a held-out set (n ≥ 300 tickets), measured as exact pass/fail match. Baseline follows Youtu-Agent approach: greedy inference (temperature=0, single sample, same model checkpoint).
- Format validity: ≥ 99% compliance with the three-line contract.
- pass@K (K=8): ≥ +5–10% over greedy baseline on the same split.
- Reflection precision: ≥ 80% of **applied** reflections show improved label_match in the next batch (measured by comparing label_match rates before and after guidance updates).
- Reflection recall: ≥ 70% of GT-aligned failure modes surface in reflection proposals within two epochs.
- Latency: median ≤ 800 ms per candidate on local GPU; reflection prompt adds ≤ 300 ms per batch.
- Storage: 100% reproducible re-runs (idempotent ingest; deterministic signals given fixed seeds). Artifacts under `output_post/` only.

## Risks & Mitigations
- Noisy group labels can bias updates → treat `label_match` as the only hard reward, layer LLM-derived `label_trust`, and apply proposals directly (operator can review reflection logs and rollback if needed).
- Experiences drift → enforce one-change-per-reflection, store provenance, and include guidance step in every artifact.
- Overly broad prompts → prompt builder asserts the presence of experiences and aborts when token budgets are exceeded.
- LLM availability/cost → reflection/triage steps fail fast when the backing LLM call fails, ensuring no silent guidance drift; operators can rerun once the dependency is restored.
- Backwards compatibility → new runner lives alongside existing GRPO; we keep adapter shims for legacy imports until downstream consumers migrate.

## Phased Rollout
- Phase 0: Scaffolding (runner, config, storage), unit tests for rollout + signal extraction schema.
- Phase 1: Single-mission pilot with manual reflection approval; verify experience bundles and logging.
- Phase 2: Implement direct application of reflection proposals, provenance logs, and wire them through `StageBPipeline.run_all();` document operator workflow.
- Phase 3: Add telemetry (reflection precision/recall, acceptance rate) and optional multi-mission batching.
- Phase 4: Stabilize, document, and set default configs; evaluate retrieval/few-shot hints only if the reflection loop plateaus.

## Implementation Status

### Delivered

- Single-process architecture with shared model instance between rollout and reflection.
- Mission-specific output structure under `{output.root}/{run_name}/{mission}/`.
- Initial reflection loop with direct application and debug logging for parse failures.
- **Signal Reduction (COMPLETE)**: `label_trust` and `candidate_agreement` are deprecated and always set to `None`. Minimal signal set consists of `label_match`, `confidence`, and optionally `self_consistency`. All tests updated to reflect this change.
- **Eligibility Policy Enhancements (COMPLETE)**: Added `contradictions_or_all_wrong` eligibility policy that triggers reflection when there are contradictions (mixed `label_match` values) OR all candidates are wrong. Added `all_wrong_strategy` config knob (`reflect_diagnose` or `manual_review`) to control behavior for all-wrong groups. Comprehensive tests added for both features.
- Normalize deterministic signals so English/Chinese verdict aliases both map to Stage-B labels.
- Rebuild eligibility gating around the selected candidate and guarantee "all wrong" bundles trigger critique.
- Harden `GuidanceRepository` with non-empty enforcement, atomic writes, and durable snapshots.
- Require syntactically valid JSON for both reflection and critic outputs; remove heuristic text fallback and adjust stop strings to prevent truncation.
- Introduce CriticEngine (strict-JSON per-candidate) and persist its fields within trajectories; pool/dedupe candidate_ops as suggestions for reflection.
- Expand exports/docs/configs to cover CriticEngine fields and new guardrail toggles (`apply_if_delta`, `rapid_mode`).

## Open Questions

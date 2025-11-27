# stage-b-training-free Specification

## Purpose
Capture the training-free Stage-B reflection pipeline: numbered experiences schema, structured reflection/critique flows, eligibility and budget guardrails, and shared model/critic integration used for group-level reasoning without additional supervised data.
## Requirements
### Requirement: Stage-B SHALL evaluate group tickets without gradient updates by orchestrating a training-free sampling and judging loop.
Stage-B SHALL rely on frozen checkpoints and orchestrated sampling rather than parameter updates while still supporting multi-attempt verdict generation for each mission batch, and must fail fast when prerequisite artifacts (Stage-A summaries, guidance snapshot, verifier config) are missing.
#### Scenario: A mission batch of Stage-A summaries is processed for verdict inference
- Given normalized Stage-A summaries and historical human verdicts, when Stage-B ingest + rollout commands run, then the system must use the frozen Stage-B checkpoint (no parameter updates) to generate one or more candidate verdicts per group and store them for judging.
- Generated verdicts must include decoding metadata (temperature, top_p, prompt variant) so later analysis can compare sampling strategies.

### Requirement: Stage-B SHALL emit reproducible verdict artifacts that satisfy existing downstream contracts.
Stage-B SHALL emit deterministic verdict artifacts while preserving the established three-line format (verdict, rationale, confidence) and JSONL schema documented for downstream integrations, even when processing multiple epochs with shuffled inputs, and SHALL refuse to export when required metadata (signals, guidance step, reflection id) is missing.
#### Scenario: Selection completes for a mission batch
- When Stage-B selects final verdicts, then the system SHALL persist a deterministic selection record keyed by `group_id` (and epoch index when `epochs>1`), including the three-line response and supporting metadata (deterministic signals, provenance) so `docs/REFERENCE.md` workflows remain valid.
- The workflow SHALL provide an export step that writes JSONL artifacts under `{output.root}/{run_name}/{mission}/` with stable schema and configurable destination, without mutating prior runs when re-executed with the same seed and shuffle configuration.

### Requirement: Stage-B SHALL persist mission guidance and trajectory logs as fail-fast file artifacts for reuse.
Stage-B SHALL capture reusable priors using simple JSON/TXT files rather than databases while retaining enough metadata to audit runs, and SHALL abort immediately if any artifact cannot be read or written.
#### Scenario: Trajectories are written after rollout or judge execution
- The system SHALL append trajectory metadata (group_id, decode params, response text, deterministic signals, confidence, guidance step, reflection_cycle, epoch, critic.summary, critic.critique, optional critic.root_cause/issues/uncertainty_note) to JSONL logs stored under `{output.root}/{run_name}/{mission}/trajectories.jsonl`. Candidate-level `candidate_ops` from CriticEngine SHALL NOT be persisted per-candidate; they are pooled and deduplicated as suggestions for reflection. A separate `critic.jsonl` file MUST NOT be created.
- Mission-specific guidance files MUST store a step counter (`step: int`), last-updated timestamp (`updated_at: iso8601`), and an `experiences` dict (e.g., `{"G0": "...", "G1": "..."}`) that is formatted as a single text block and prepended to prompts. The storage schema SHALL be `{step: int, updated_at: iso8601, experiences: {"G0": str, "G1": str, ...}}`. Experiences are stored internally as numbered entries (matching Youtu-Agent's pattern) and formatted as a single text block when prepending to prompts (e.g., `"[G0]. ...\n[G1]. ..."`). Missing fields SHALL raise and block the run.
- Guidance writes SHALL be atomic: Stage-B MUST write to a temporary file in the same directory, perform a `rename` to overwrite the live file, and only then prune snapshots. Snapshot filenames SHALL include microsecond-resolution timestamps to avoid collisions (`guidance-YYYYMMDD-HHMMSS-ffffff.json`).
- The repository SHALL reject any operation sequence that leaves the `experiences` dict empty and shall surface a validation error instead of mutating disk.
- Reflection decisions SHALL be logged to `reflection.jsonl` with evidence group ids, summary, critique, eligibility flag, and the guidance step (`guidance_step_before`, `guidance_step_after`) so operators can audit every change. The log schema SHALL include: `epoch`, `reflection.reflection_id`, `reflection.mission`, `reflection.proposal` ({action, summary, critique, operations[{op, key, text?, rationale?, evidence}], evidence_group_ids, uncertainty_note}), `reflection.applied: bool`, `reflection.pre_uplift`, `reflection.post_uplift`, `reflection.guidance_step_before`, `reflection.guidance_step_after`, `reflection.debug_info` (optional with parser exception details).
- Operators MUST be able to edit guidance files manually; pipeline updates SHALL remain idempotent, preserve prior versions via timestamped backups, and fail when edits conflict with the current step counter.

### Requirement: Stage-B SHALL assemble prompts with mission experiences explicitly and abort when injection fails.
The rollout builder MUST format the mission experiences as a single text block (e.g., `"[G0]. ...\n[G1]. ..."`) and prepend it before each Stage-A summary so that every candidate shares the same context. Experiences are stored internally as numbered entries (dict: `{"G0": "...", "G1": "..."}`) and formatted as a single text block when prepending to prompts (matching Youtu-Agent's pattern).
#### Scenario: A rollout batch is prepared
- The system SHALL format the current experiences dict as a single text block using `"\n".join([f"[{i}]. {e}" for i, e in sorted(experiences.items())])` and prepend it to each prompt. The `_render_guidance_snippets()` function SHALL accept an `experiences: Dict[str, str]` parameter (not a `GuidanceEntry` list) and format it as numbered experiences with sorted keys for consistent ordering.
- If the experiences dict is empty, the pipeline SHALL abort (no empty fallback). The prompt building logic SHALL validate that experiences dict is non-empty before proceeding.
- If the experiences dict is unreadable or exceeds the configured token budget, the rollout command SHALL stop and surface the error; there is no empty fallback.
- The pipeline SHALL log the guidance step (not version) before sampling so operators can verify the prompt state in telemetry or logs.

### Requirement: Stage-B SHALL record minimal deterministic signals for each candidate to support LLM reflection.
The signal subsystem SHALL compute mission-agnostic metrics that downstream automation and the reflection LLM can consume without manual parsing, following an LLM-first philosophy where deterministic signals provide minimal context only. The system SHALL log warnings when any metric is missing (but continue processing).
#### Scenario: Signals are generated after rollout
- Signal extraction MUST emit a minimal schema (`label_match: Optional[bool]`, `self_consistency: Optional[float]`, `confidence: Optional[float]`) compatible with automated parsing and stash it alongside each trajectory. The `DeterministicSignals` dataclass SHALL include these fields plus deprecated fields `candidate_agreement: Optional[bool]` and `label_trust: Optional[float]` which are always set to `None` and MUST NOT be used for decision-making.
- Verdict strings SHALL be normalised through a shared helper that maps both English (`"pass"`, `"fail"`) and Chinese (`"通过"`, `"不通过"`) tokens to the canonical `GroupLabel` before computing `label_match`.
- Label agreement (`label_match`) remains the only hard supervision signal; `confidence` and optional `self_consistency` provide minimal context for tie-breaking and LLM reflection prompts.
- Signals are used for selection tie-breaking and LLM context only; they MUST NOT replace the reflection step. The LLM reflection engine is the primary decision-making mechanism.
- Signal outputs must be stored without overwriting the original labels or responses. If signal extraction encounters malformed output (e.g., missing confidence), the system MUST log a warning and continue processing (exception queue is simplified to warnings only).

### Requirement: Stage-B SHALL operate without retrieval or embedding dependencies.
The baseline implementation SHALL rely solely on mission guidance files and Stage-A summaries to guide sampling and selection.
#### Scenario: Rollout is executed using default configuration
- The sampler MUST function when retrieval is disabled, using only configured decode grids and guidance-derived prompt tweaks.
- Configuration SHALL NOT require embedding providers or vector indexes; any retrieval features MUST be disabled by default and safely ignored when unset.
- Selection MUST rely on ground-truth alignment (`label_match`) with trust-weighted tie-breakers using minimal deterministic signals (confidence, label_trust, etc.); no complex rule-based judge scoring is required and retrieval remains optional/disabled by default.

### Requirement: Stage-B tooling SHALL expose a fail-fast `run_all()` orchestration entry point.
The pipeline SHALL provide a single programmatic entry (`src.stage_b.runner.run_all`) that executes ingest → rollout → signals → reflect → apply → select/export, validating inputs before each phase and emitting telemetry for monitoring.
#### Scenario: Operators invoke Stage-B end-to-end
- `run_all()` MUST raise when required artifacts are missing or malformed and leave partial outputs for inspection.
- The pipeline MUST log telemetry summaries (counts of processed tickets, reflection proposals, applied/rejected stats, quarantined tickets) and write them into the run directory for downstream monitoring.
- **Single-process architecture**: The pipeline MUST run in a single Python process with one entry point (`python -m src.stage_b.runner`). The model is loaded once and shared between rollout sampling, CriticEngine, and reflection. Multi-GPU scaling is handled via `device_map="auto"` or explicit DDP wrapping if needed.

### Requirement: Stage-B SHALL treat LLM reflection-guided experiences updates as the optimizer step with direct application.
Mission experiences updates SHALL only occur through the reflection workflow, which proposes at most one change per batch, attaches provenance, and applies it directly for the next batch. This aligns with Youtu-Agent's approach: experiences evolve after each batch based on critiques, then apply globally to the next batch.
#### Scenario: Reflection runs after each complete batch
- Given trajectories, deterministic signals, trust scores, and GT labels from a complete batch, when reflection executes, it MUST produce at most one proposal `{action: "refine" | "noop", summary: str, critique: str, operations: [...], evidence_group_ids: [...], uncertainty_note?: str}`. The `operations` list SHALL encode incremental edits where each element is `{op: "upsert" | "remove" | "merge", key: str, text?: str, rationale?: str, evidence: [...]}`. The `ReflectionAction` type SHALL remain restricted to `"refine"` or `"noop"`.
- **CriticEngine (LLM-based per-candidate evaluation)**: The pipeline SHALL include a CriticEngine that emits strict-JSON per-candidate records with schema `{summary: str, critique: str, root_cause?: str, issues?: [str], candidate_ops?: [...], uncertainty_note?: str}`. The CriticEngine MUST validate its prompt template at initialization, use stable generation params (low temperature 0.1–0.3, top_p ~0.9, bounded `max_new_tokens`), enforce per-field length caps, and evaluate at most `critic.max_candidates` (<= 6) per group after a cheap pre-filter. Critic outputs MUST be persisted with trajectories and bundled into reflection inputs; `candidate_ops` are pooled/deduplicated as suggestions for reflection and are not applied directly.
- **Eligibility gating**: The CritiqueScreener MUST deem a bundle eligible based on the configured `eligibility_policy`:
  - `selected_mismatch_or_all_wrong` (default): eligible when (a) the selected candidate for any group violates `label_match` OR (b) every candidate in a group disagrees with the ground-truth label (all `label_match=False`).
  - `contradictions_only`: eligible when there are contradictions (mixed `label_match` values across candidates within a group) OR a selected mismatch; uniform all-wrong batches become ineligible.
  - `contradictions_or_all_wrong`: eligible when there are contradictions (mixed `label_match` values) OR all candidates are wrong (all `label_match=False`).
- **All-wrong strategy**: When `all_wrong_strategy="manual_review"` and a bundle contains all-wrong groups, the engine SHALL short-circuit to a noop proposal with `ineligible_reason="all_wrong_manual_review"` and `critique="Flagged for 人工复核"` without calling the reflection LLM. When `all_wrong_strategy="reflect_diagnose"` (default), all-wrong groups proceed to reflection for diagnosis.
- Ineligible bundles SHALL include `ineligible_reason` metadata in logs and exported artifacts.
- **Structured JSON only**: Reflection responses SHALL be emitted as strict JSON. The engine MUST disable `_parse_experiences_from_text()` style fallbacks, treat truncated JSON (e.g., due to stop-string cropping) as fatal, and surface parser errors through telemetry and `reflection.debug_info`.
- **Incremental merge**: When `action="refine"`, the system SHALL merge the provided operations into the existing experiences dictionary (supporting add/update/remove semantics) rather than replacing it wholesale. Each applied operation MUST capture metadata (`reflection_id`, evidence sources, optional rationale, updated_at) for provenance. `remove` operations SHALL be rejected when they would leave the experiences dict empty.
- **Batch boundary enforcement**: Reflection MUST run **after each complete batch within the epoch**, ensuring all samples in the batch see the same experiences step. Updates apply to subsequent batches in the same epoch to maintain rapid feedback while avoiding intra-batch version drift.
- **Holdout gating (configurable)**: The engine SHALL preview the proposal against holdout tickets (when available) and only persist operations if the measured uplift in `label_match_rate` meets or exceeds `apply_if_delta`. Proposals carrying `uncertainty_note` SHALL be rejected automatically when `allow_uncertain=false`. A `rapid_mode: true` debug flag MAY disable uplift gating and raise change caps for fast iteration; production defaults keep guardrails enabled.
- Every applied proposal MUST increment the mission guidance step (not version) and persist provenance to `reflection.jsonl` with `guidance_step_before`/`guidance_step_after`, along with the serialized summary, critique, operations list, eligibility flag, and sample evidence.
- **In-process model reuse**: Reflection MUST use the same model instance as rollout sampling, passed directly to `ReflectionEngine` (no client-server architecture). The model is loaded once in `runner.py` and shared between `RolloutSampler`, `CriticEngine`, and `ReflectionEngine`. This enables single-process execution with shared GPU resources and simplifies multi-GPU scaling via `device_map="auto"`.

### Requirement: Stage-B SHALL log warnings for problematic tickets without complex exception queue management.
When Stage-B encounters issues (malformed responses, low trust scores, etc.), it SHALL log warnings but continue processing. The system does not maintain a complex exception queue; operators can review warnings in logs.
#### Scenario: Warning is triggered during selection or reflection
- The system SHALL log warnings with `group_id`, reason, guidance step, and (when reflection is skipped) `ineligible_reason` for problematic tickets.
- The pipeline SHALL continue processing all tickets; warnings are informational only.
- Export artifacts SHALL include warning flags, `summary`, `critique`, and `ineligible_reason` (when present) so downstream quality control can identify tickets that may need review.

### Requirement: Stage-B SHALL include an LLM-based CriticEngine for per-candidate evaluations.
Stage-B SHALL produce human-auditable per-candidate evaluations using the CriticEngine, which outputs strict-JSON fields consumed by reflection and exports.
#### Scenario: A batch completes rollout
- For every processed group, the CriticEngine SHALL emit `{summary, critique, root_cause?, issues?, candidate_ops?, uncertainty_note?}` for up to `critic.max_candidates` candidates (<= 6), enforcing per-field length caps (`critic.summary_max_chars`, `critic.critique_max_chars`).
- The `summary`/`critique` fields SHALL be persisted in trajectories and exported alongside selections with the corresponding `group_id`; optional `root_cause`/`issues`/`uncertainty_note` MAY be persisted for auditability. `candidate_ops` are pooled/deduplicated and passed to reflection as suggestions.
- Reflection prompts SHALL reference the per-candidate critic fields when constructing operations so reviewers can trace guidance changes back to concrete observations.

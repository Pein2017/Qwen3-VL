## ADDED Requirements

### Requirement: Stage-B SHALL evaluate group tickets without gradient updates by orchestrating a training-free sampling and judging loop.
Stage-B SHALL rely on frozen checkpoints and orchestrated sampling rather than parameter updates while still supporting multi-attempt verdict generation for each mission batch, and must fail fast when prerequisite artifacts (Stage-A summaries, guidance snapshot, verifier config) are missing.
#### Scenario: A mission batch of Stage-A summaries is processed for verdict inference
- Given normalized Stage-A summaries and historical human verdicts, when Stage-B ingest + rollout commands run, then the system must use the frozen Stage-B checkpoint (no parameter updates) to generate one or more candidate verdicts per group and store them for judging.
- Generated verdicts must include decoding metadata (temperature, top_p, prompt variant) so later analysis can compare sampling strategies.

### Requirement: Stage-B SHALL emit reproducible verdict artifacts that satisfy existing downstream contracts.
Stage-B SHALL emit deterministic verdict artifacts while preserving the established three-line format (verdict, rationale, confidence) and JSONL schema documented for downstream integrations, even when processing multiple epochs with shuffled inputs, and SHALL refuse to export when required metadata (signals, guidance step, reflection id) is missing.
#### Scenario: Selection completes for a mission batch
- When Stage-B selects final verdicts, then the system SHALL persist a deterministic selection record keyed by `group_id` (and epoch index when `epochs>1`), including the three-line response and supporting metadata (deterministic signals, provenance) so `docs/REFERENCE.md` workflows remain valid.
- The workflow SHALL provide an export step that writes JSONL/Parquet artifacts under `{output.root}/{run_name}/{mission}/` with stable schema and configurable destination, without mutating prior runs when re-executed with the same seed and shuffle configuration.

### Requirement: Stage-B SHALL persist mission guidance and trajectory logs as fail-fast file artifacts for reuse.
Stage-B SHALL capture reusable priors using simple JSON/TXT files rather than databases while retaining enough metadata to audit runs, and SHALL abort immediately if any artifact cannot be read or written.
#### Scenario: Trajectories are written after rollout or judge execution
- The system SHALL append trajectory metadata (group_id, decode params, response text, deterministic signals, confidence, guidance step, reflection_cycle, epoch, summary, critique) to JSONL logs stored under `{output.root}/{run_name}/{mission}/trajectories.jsonl`.
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

### Requirement: Stage-B SHALL record deterministic signals and trust scores for each candidate to support LLM reflection.
The signal subsystem SHALL compute mission-agnostic metrics that downstream automation and the reflection LLM can consume without manual parsing, and SHALL log warnings when any metric is missing (but continue processing).
#### Scenario: Signals are generated after rollout
- Signal extraction MUST emit a schema (`label_match: Optional[bool]`, `self_consistency: Optional[float]`, `candidate_agreement: Optional[bool]`, `confidence: Optional[float]`, `label_trust: Optional[float]`) compatible with automated parsing and stash it alongside each trajectory. The `DeterministicSignals` dataclass SHALL include all five fields.
- Verdict strings SHALL be normalised through a shared helper that maps both English (`"pass"`, `"fail"`) and Chinese (`"通过"`, `"不通过"`) tokens to the canonical `GroupLabel` before computing `label_match`.
- Label agreement remains the only hard supervision signal; trust scores weight reflection decisions to guide proposal quality.
- Signal outputs must be stored without overwriting the original labels or responses. If signal extraction encounters malformed output (e.g., missing confidence or `label_trust`), the system MUST log a warning and continue processing (exception queue is simplified to warnings only).

### Requirement: Stage-B SHALL operate without retrieval or embedding dependencies.
The baseline implementation SHALL rely solely on mission guidance files and Stage-A summaries to guide sampling and selection.
#### Scenario: Rollout is executed using default configuration
- The sampler MUST function when retrieval is disabled, using only configured decode grids and guidance-derived prompt tweaks.
- Configuration SHALL NOT require embedding providers or vector indexes; any retrieval features MUST be disabled by default and safely ignored when unset.
- Selection MUST rely on deterministic judge scores and guidance hints without requiring retrieval results.

### Requirement: Stage-B tooling SHALL expose a fail-fast `run_all()` orchestration entry point.
The pipeline SHALL provide a single programmatic entry (`StageBPipeline.run_all`) that executes ingest → rollout → signals → reflect → apply → select/export, validating inputs before each phase and emitting telemetry for monitoring.
#### Scenario: Operators invoke Stage-B end-to-end
- `StageBPipeline.run_all()` MUST raise when required artifacts are missing or malformed and leave partial outputs for inspection.
- The pipeline MUST log telemetry summaries (counts of processed tickets, reflection proposals, applied/rejected stats, quarantined tickets) and write them into the run directory for downstream monitoring.
- **Single-process architecture**: The pipeline MUST run in a single Python process with one entry point (`python -m src.stage_b.runner`). The model is loaded once and shared between rollout sampling and reflection. Multi-GPU scaling is handled via `device_map="auto"` or explicit DDP wrapping if needed.

### Requirement: Stage-B SHALL treat LLM reflection-guided experiences updates as the optimizer step with direct application.
Mission experiences updates SHALL only occur through the reflection workflow, which proposes at most one change per batch, attaches provenance, and applies it directly for the next batch. This aligns with Youtu-Agent's approach: experiences evolve after each batch based on critiques, then apply globally to the next batch.
#### Scenario: Reflection runs after each complete batch
- Given trajectories, deterministic signals, trust scores, and GT labels from a complete batch, when reflection executes, it MUST produce at most one proposal `{action: "refine" | "noop", summary: str, critique: str, operations: [...], evidence_group_ids: [...], uncertainty_note?: str}`. The `operations` list SHALL encode incremental edits where each element is `{op: "upsert" | "remove" | "merge", key: str, text?: str, rationale?: str, evidence: [...]}`. The `ReflectionAction` type SHALL remain restricted to `"refine"` or `"noop"`.
- **Youtu-Agent style summaries/criticues**: The reflection pipeline SHALL include a SampleSummarizer component that emits per-record `summary` and `critique` strings following the Youtu-Agent format (summary first restates success, critique outlines failure modes). These strings MUST be persisted with trajectories and reflection inputs.
- **Eligibility gating**: The CritiqueScreener MUST deem a bundle eligible when either (a) the selected candidate for any group violates `label_match` or (b) every candidate in a group disagrees with the ground-truth label. Ineligible bundles SHALL include `ineligible_reason` metadata in logs and exported artifacts.
- **Structured JSON only**: Reflection responses SHALL be emitted as strict JSON. The engine MUST disable `_parse_experiences_from_text()` style fallbacks, treat truncated JSON (e.g., due to stop-string cropping) as fatal, and surface parser errors through telemetry and `reflection.debug_info`.
- **Incremental merge**: When `action="refine"`, the system SHALL merge the provided operations into the existing experiences dictionary (supporting add/update/remove semantics) rather than replacing it wholesale. Each applied operation MUST capture metadata (`reflection_id`, evidence sources, optional rationale, updated_at) for provenance. `remove` operations SHALL be rejected when they would leave the experiences dict empty.
- **Batch boundary enforcement**: Reflection MUST run **after each complete batch within the epoch**, ensuring all samples in the batch see the same experiences step. Updates apply to subsequent batches in the same epoch to maintain rapid feedback while avoiding intra-batch version drift.
- **Holdout gating**: The engine SHALL preview the proposal against holdout tickets (when available) and only persist operations if the measured uplift in `label_match_rate` meets or exceeds `apply_if_delta`. Proposals carrying `uncertainty_note` SHALL be rejected automatically when `allow_uncertain=false`.
- Every applied proposal MUST increment the mission guidance step (not version) and persist provenance to `reflection.jsonl` with `guidance_step_before`/`guidance_step_after`, along with the serialized summary, critique, operations list, eligibility flag, and sample evidence.
- **In-process model reuse**: Reflection MUST use the same model instance as rollout sampling, passed directly to `ReflectionEngine` (no client-server architecture). The model is loaded once in `runner.py` and shared between `RolloutSampler` and `ReflectionEngine`. This enables single-process execution with shared GPU resources and simplifies multi-GPU scaling via `device_map="auto"`.

### Requirement: Stage-B SHALL log warnings for problematic tickets without complex exception queue management.
When Stage-B encounters issues (malformed responses, low trust scores, etc.), it SHALL log warnings but continue processing. The system does not maintain a complex exception queue; operators can review warnings in logs.
#### Scenario: Warning is triggered during selection or reflection
- The system SHALL log warnings with `group_id`, reason, guidance step, and (when reflection is skipped) `ineligible_reason` for problematic tickets.
- The pipeline SHALL continue processing all tickets; warnings are informational only.
- Export artifacts SHALL include warning flags, `summary`, `critique`, and `ineligible_reason` (when present) so downstream quality control can identify tickets that may need review.

### Requirement: Stage-B SHALL emit Youtu-Agent style per-sample summaries and critiques.
Stage-B SHALL produce human-auditable textual summaries of each winning candidate and critiques of residual failure modes, aligned with Youtu-Agent conventions, and expose them to downstream analysis.
#### Scenario: A batch completes rollout
- For every processed group, the SampleSummarizer SHALL emit a `summary` (2–3 sentences restating what the candidate concluded) and a `critique` (1–2 sentences highlighting failure cues or follow-up guidance needs).
- These strings SHALL be persisted in trajectories, bundled into reflection input payloads, and exported alongside selections with the corresponding `group_id`.
- Reflection prompts SHALL reference the per-sample `summary` and `critique` fields when constructing operations so reviewers can trace guidance changes back to concrete observations.

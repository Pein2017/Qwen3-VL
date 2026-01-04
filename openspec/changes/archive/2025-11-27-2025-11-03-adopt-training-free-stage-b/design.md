# Training-Free Stage-B Design

## Architecture Overview
The updated Stage-B flow mirrors Youtu-Agent's training-free GRPO recipe: a frozen policy repeatedly samples candidates, judges them against ground-truth labels, then runs a **three-stage reflection pipeline** (CriticEngine → CritiqueScreener → GuidanceConsolidator) to synthesise the next guidance cue. The LLM-based CriticEngine produces per-candidate strict‑JSON evaluations; deterministic signals are minimal and used for selection tie‑breaking and context only — they never replace the reflection step and must be reliable across locale variants.

## Verification & Guardrails
- **Template validation**: ReflectionEngine and CriticEngine both validate their prompt templates at initialization (required placeholders, allowed ops, JSON-only instructions, `K` budget hints). Missing hints raise `ValueError` before any sampling starts.
- **Strict JSON + token budgets**: Reflection parsing no longer falls back to text heuristics; truncated or non-JSON responses surface parser errors through `reflection.debug_info`. Prompt construction is capped by `reflection.token_budget` (default 1536 tokens) with prioritised record packing (contradictions > selected mismatches > others).
- **Guidance schema enforcement**: `GuidanceRepository` enforces the `{step, updated_at, experiences}` schema, rejects empty experience maps (even after `remove` operations), and performs temp-file → rename → snapshot writes to keep artifacts atomic.
- **Conservative selection overrides**: The selection policy is label-first with deterministic tie-breakers, but CriticEngine uncertainty signals (`needs_recheck`, `evidence_sufficiency=false`, `recommended_action="人工复核"`) force a `"不通过"` verdict and log warnings for auditability.
- **Legacy deprecation**: Historical migration helpers and the SampleSummarizer have been removed; CriticEngine summaries/criticisms flow through trajectories, reflection bundles, and exports.

## Terminology Simplification
**Numbered Experiences** (Youtu-Agent pattern): Experiences are stored internally as numbered entries (e.g., `{"G0": "...", "G1": "..."}`) and formatted as a single text block when prepended to prompts. This aligns with Youtu-Agent's approach where experiences are stored as a dict internally but formatted as a single text block for prompts.

The final result after reflection is a formatted text block (e.g., `"[G0]. ...\n[G1]. ..."`) that is prepended to all prompts and helps the model perform better in the downstream task. This maintains the simplicity of a single text block in prompts while preserving the ability to track and evolve individual guidance points.

1. **Ingestion & Run Directory**
   - Consumes Stage-A JSONL outputs produced via `scripts/stage_a_infer.sh`.
   - Materialises a run directory (`{output.root}/{run_name}/{mission}/`) with mission-specific files: `trajectories.jsonl`, `selections.jsonl`, `guidance.json`, `reflection.jsonl`.
   - Fails immediately if required artifacts (Stage-A summaries, guidance shell, writable run dir) are missing; there is no silent defaulting.

2. **Mission Experiences (Global per Mission)**
   - `{output.root}/{run_name}/{mission}/guidance.json` stores a step counter (`step: int`), last-updated timestamp (`updated_at: iso8601`), and an `experiences` dict (e.g., `{"G0": "...", "G1": "..."}`) that is formatted as a single text block and prepended to all prompts. Snapshot rotation uses microsecond-resolution filenames (e.g., `guidance-20251106-153045-123456.json`), and writes flow through a temp-file (`guidance.json.tmp`) followed by `rename` for atomicity.
   - **Never-empty enforcement**: Experiences are shared globally per mission, but the repository now rejects any write that would leave the dict empty (even after `remove` operations). This ensures prompts always inject at least one numbered cue and prevents wedge states after reflection errors.
   - Every rollout builds prompts by formatting the current experiences dict as a single text block using `"\n".join([f"[G{i}]. {e}" for i, e in sorted(experiences.items())])` (e.g., `"[G0]. ...\n[G1]. ..."`) and prepending it ahead of the Stage-A summary. Missing guidance files or empty experiences dict abort the run rather than falling back to empty text.
   - Provenance logs remain for auditability, tracking when and how experiences were updated. Reflection logs use `guidance_step_before` and `guidance_step_after` (not `guidance_version`). Snapshot retention applies after successful atomic writes so history is never clobbered.

3. **Sampler**
   - Uses the frozen Stage-B checkpoint with `transformers.generate` and refuses to start until prompts confirm the prompt guidance was injected.
   - Supports multi-attempt sweeps (temperature ladder) and requires each candidate to emit three lines (verdict, rationale referencing Stage-A text, model confidence); malformed structures are rejected on the spot and logged as warnings.
   - Logs each attempt with decode parameters, raw response, timestamp, and the guidance step applied so reflection investigations are unambiguous.

4. **Signal Extraction & Trust Scoring**
   - `compute_signals` records `label_match: Optional[bool]`, `self_consistency: Optional[float]`, `candidate_agreement: Optional[bool]`, `confidence: Optional[float]`, and `label_trust: Optional[float]` scalar derived from verifier plugins (geometry checks, secondary LLM judge) plus Stage-A completeness flags. The `DeterministicSignals` dataclass includes all five fields.
   - Verdict strings (`"pass"`, `"fail"`, `"通过"`, `"不通过"`) are normalised through a single helper so localisation never flips `label_match` to `False` erroneously.
   - Signals are written alongside trajectories. If signal extraction encounters malformed output (e.g., missing confidence or `label_trust`), the system logs a warning and continues processing (exception queue is simplified to warnings only).

5. **Reflection Engine (Incremental, Three-Stage)**
   - `reflect_on_batch` runs **after each complete batch within the epoch** (mid-epoch approach) and now orchestrates three explicit subcomponents:
     1. **CriticEngine**: generates per-candidate strict-JSON evaluations with schema `{summary: str, critique: str, root_cause?: str, issues?: [str], uncertainty_note?: str}`. Shares the same in-process Qwen3-VL instance as rollout/reflection; validates the critic prompt template at initialization; uses low temperature (0.1-0.3), top_p ~0.9, and bounded `max_new_tokens` for stability. Enforces per-field length caps (summary/critique each <= `config.critic.summary_max_chars` / `config.critic.critique_max_chars`) and evaluates at most `config.critic.max_candidates` (<= 6) per group, selected by a cheap pre-filter (e.g., label mismatch, low self_consistency, contradictions). Candidate-level operations are not collected; prompts rely on summaries/critiques and deterministic signals only.
     2. **CritiqueScreener**: inspects deterministic signals for the *winning* candidate and checks whether any group had consensus failure (`label_match=False` for all candidates). Only when at least one of those conditions is met does the bundle become eligible for reflection; otherwise it is skipped with an explicit `ineligible_reason`.
     3. **GuidanceConsolidator**: consumes CriticEngine outputs, the candidate pool, and deterministic signals and expects strict JSON: `{"action": "refine"|"noop", "summary": str, "critique": str, "operations": [...], "evidence_group_ids": [...], "uncertainty_note"?: str}`. No heuristic text fallback exists; truncation or malformed JSON is treated as a fatal error and surfaced up the stack.
   - **Experience parsing**: only the structured `operations` list (`{op: "upsert"|"remove"|"merge", key: str, text?: str, rationale?: str, evidence: [...]}`) is accepted. `_parse_experiences_from_text()` is removed, forcing prompt/template compliance. Evidence identifiers resolve to mission group ids (supporting `第N组` and raw ids).
   - **Incremental merge + gating**: Operations are staged and merged into the existing experiences dict only after holdout uplift validates the proposal (`apply_if_delta` threshold). Guardrails (uplift gating, `max_operations`, `change_cap_per_epoch`) are configurable; a `rapid_mode: true` debug flag may disable uplift gating and raise caps for fast iteration, while the default production mode keeps guardrails enabled. `allow_uncertain=false` rejects proposals carrying `uncertainty_note`. Per-entry metadata (`reflection_id`, evidence sources, rationale, updated_at) is recorded for provenance. `remove` operations can only execute when at least one other entry remains; otherwise they fail validation.
   - **In-process model reuse**: CriticEngine and Reflection use the same model instance as rollout sampling, passed directly within the process (no client-server architecture). The model is loaded once in `runner.py` and shared between `RolloutSampler`, `CriticEngine`, and `ReflectionEngine`. This enables single-process execution with shared GPU resources and simplifies multi-GPU scaling via `device_map="auto"`.
   - **Batch boundary enforcement**: All samples in a batch see the same prompt guidance step; updates committed after holdout gating apply to subsequent batches in the same epoch, mirroring Youtu-Agent's mid-epoch evolution without intra-batch drift.
   - Applied changes bump the mission guidance step and append a `reflection.jsonl` entry with eligibility, uplift metrics, serialized operations, summary/critique text, and evidence group IDs.

6. **Selection, Warnings & Export**
   - `select_for_group` prioritises GT-aligned candidates with trust-weighted tie-breakers; if every candidate disagrees with guidance and verifier trust is low, the group is logged with a warning (no complex exception queue).
   - Warnings are logged with `group_id`, reason, and guidance step. All tickets continue to be processed; warnings are informational only.
   - `export_trajectories`, `export_selections`, and the new `export_reflection_inputs` include deterministic signals, confidence, guidance step, reflection id, warning flags, and the CriticEngine’s critic fields (`summary`, `critique`, optional `root_cause`/`issues`/`uncertainty_note`) so downstream QC can identify tickets that may need manual follow-up.

7. **Interfaces & Package Layout**
   - `src/stage_b/ingest.py`: load Stage-A outputs, hydrate prompt guidance, enforce run-dir setup, and fail fast when prerequisites are missing.
   - `src/stage_b/rollout.py`: run the sampler with explicit prompt guidance injection and persist trajectory logs keyed by step.
   - `src/stage_b/signals.py`: compute deterministic metrics, trust scores, and log warnings if any metric cannot be produced (no exception queue).
   - `src/stage_b/critic/engine.py`: LLM-based per-candidate critic with strict-JSON outputs, template validation, stable generation params, per-field length caps, and candidate pre-filtering.
   - `src/stage_b/reflection/engine.py`: reflection engine (Stage-B training-free), orchestrates experience bundle creation, strict JSON parsing, applies structured updates to `GuidanceRepository`, and emits `reflection.jsonl`.
   - `src/stage_b/scoring/selection.py`: choose final verdicts using label-first policy with trust-weighted tie-breakers and warning flags.
   - `src/stage_b/io/export.py`: emit JSONL artifacts and reflection logs.
   - Legacy GRPO modules will be deleted or shimmed; public imports (e.g., `src.stage_b.build_stage_b_messages`) re-point to new implementations.
   - `StageBPipeline` aggregates these modules and exposes a single `run_all()` entry point that processes batches sequentially: for each batch, it orchestrates rollout → signals → reflection → apply prompt guidance updates → selection/export, then processes the next batch. This ensures all samples in a batch see the same prompt guidance step and updated prompt guidance applies to subsequent batches.

## Data Flow
```
For each batch:
  Stage-A summaries → ingestion
  → rollout (generate K candidates with current prompt guidance) → trajectory logs (JSONL)
  → deterministic signal extraction (label_match, consistency, confidence)
  → reflection (after batch completes: LLM proposes prompt guidance update)
  → apply prompt guidance update (for next batch)
  → selection (label-first with tie-breakers)
  → export (JSONL + reflection logs)

Note: Prompt guidance updates happen at batch boundaries; updated prompt guidance applies to the next batch (one-step lag). This mirrors Youtu-Agent's approach: prompt guidance evolves after each batch and applies globally to the next batch.
```

## Reflection Step: How It Works

**Purpose**: After each complete batch, the reflection engine proposes structured, incremental edits to the mission experiences. Responses MUST be strict JSON (no text fallback). Edits are budgeted, eligibility-gated, previewed for uplift, and then applied atomically.

**Process**:

1. **Build Experience Bundle** (after batch completes):
   - Collect all candidates from the batch with their deterministic signals:
     - **GT-aligned winners** and **conflicting candidates** (order preserved)
     - Signals: `label_match`, `self_consistency`, `candidate_agreement`, `confidence`, `label_trust`
   - For each candidate, CriticEngine emits strict-JSON `{summary, critique, root_cause?, issues?, uncertainty_note?}` with per-field length caps (`config.critic.summary_max_chars`, `config.critic.critique_max_chars`). Evaluate at most `config.critic.max_candidates` (<= 6) per group after a cheap pre-filter; bundle all critic fields (not only the winner) into reflection inputs.

2. **LLM Reflection Call** (same checkpoint as Stage‑B):
   - Inputs: current experiences formatted as a single text block (e.g., `"[G0]. ...\n[G1]. ..."`) + the experience bundle (winners/losers, signals, summaries/critiques, deficiency notes). The system prompt MUST state allowed ops and the K budget.
   - Output (strict JSON only; invalid/truncated JSON is fatal and logged via `reflection.debug_info`):
     ```json
     {
       "action": "refine" | "noop",
       "summary": "...",
       "critique": "...",
       "operations": [
         {"op": "upsert"|"remove"|"merge", "key": "G8"|null, "text": "...", "rationale": "...", "evidence": ["第1组","第3组"], "merged_from": ["G1","G3"]?}
       ],
       "evidence_group_ids": ["QC-001","QC-003"],
       "uncertainty_note": null
     }
     ```
   - Evidence identifiers MAY use short forms (如“第N组”); the engine resolves them to real `group_id`s.

3. **Apply Update** (preview → commit):
   - Enforce budgets:
     - Per‑proposal cap `reflection.max_operations` (truncate with a warning).
     - Epoch change cap `reflection.change_cap_per_epoch` (skip further updates once reached; record `ineligible_reason="change_cap_reached"`). Counters are mission-scoped and reset at the start of each epoch.
     - Prompt token budget `reflection.token_budget` (default 1536) with prioritized packing (contradictions > selected‑mismatch > others); tail trimming decisions are logged.
   - Eligibility policy (configurable):
     - `selected_mismatch_or_all_wrong` (default): eligible if any selected candidate mismatches GT or any group is all‑wrong.
     - `contradictions_only`: eligible if contradictions within the bundle (mixed label_match/verdict) or a selected mismatch; uniform all‑wrong bundles are ineligible.
   - Holdout gating: compute uplift on a preview guidance map; persist only if `delta ≥ apply_if_delta`. When `allow_uncertain=false`, proposals carrying `uncertainty_note` are rejected.
   - Merge semantics: create/target `key` with `text`; deprecate/remove `merged_from` keys; record provenance (`reflection_id`, `merged_from`, `evidence`, `rationale`, `updated_at`); reject removals that would empty the set; prefer stable `G<number>` IDs (no global reindex).
   - Persistence: write temp → atomic rename → copy snapshot (microsecond filename) → prune; bump `guidance_step`; append `reflection.jsonl` with proposal, eligibility, operations, and evidence. `guidance.json` remains experiences-only (no per-entry metadata); provenance lives in `reflection.jsonl`.

**Prompt expectations (concise)**:
- 允许的操作: `upsert|remove|merge`；若为 `merge` 必须包含 `merged_from`。
- 每次反思最多 K=`{reflection.max_operations}` 个操作；超出将被忽略并记录警告。
- 所有候选均提供 `summary`/`critique` 作为证据，保持候选顺序稳定以便交叉引用。

- **Template validation**: Both CriticEngine and ReflectionEngine validate their prompt templates at initialization. ReflectionEngine checks for required placeholders (`{mission}`, `{focus}`, `{experiences}`, `{bundle}`), JSON output requirement, allowed operations mention (`upsert|remove|merge`), and budget symbol `K` when `max_operations` is set. CriticEngine checks for required placeholders (`{mission}`, `{stage_a_summary}`, `{candidate_response}`, `{signals}`) and JSON output requirement. Missing items MUST raise a `ValueError` at initialization (fail-fast behavior).
- **Experience parsing**: No text fallback exists; `_parse_experiences_from_text()` is removed. Invalid or truncated JSON is fatal and surfaced via `reflection.debug_info`.
## Selection Step: How It Works

**Purpose**: After rollout generates K candidates per group and signals are computed, select the final verdict for each group based on ground-truth alignment and trust scores.

**Process**:

1. **Priority Ordering** (`label_first` policy):
   - **Primary**: Select candidates where `label_match=True` (candidate verdict matches ground-truth label)
   - **Secondary**: Among GT-aligned candidates, use tie-breakers:
     - `label_trust` (higher is better)
     - `confidence` (higher is better)
     - `self_consistency` (higher is better)
     - `candidate_agreement` (higher is better)

2. **Tie-Breaking**:
   - If multiple GT-aligned candidates have similar scores, break ties using:
     - `tie_break: "confidence"` → prefer candidate with highest model confidence
     - `tie_break: "temperature"` → prefer candidate from lower temperature decode (more deterministic)

3. **Conservative override (default on)**:
   - If the CriticEngine signals uncertainty for the chosen candidate (`needs_recheck=true`, `evidence_sufficiency=false`, or `recommended_action="人工复核"`), the final verdict is forced to `"不通过"` and a warning is appended.

4. **Warning Logging**:
   - If no GT-aligned candidates exist or `label_trust` is very low:
     - Log warning with `group_id`, reason, and `guidance_step`
     - Continue processing (no exception queue)
     - Warning flag is set in export artifacts

5. **Stamping**:
   - Each selected verdict is stamped with:
     - `guidance_step` (which prompt guidance step was used)
     - `reflection_id` (if prompt guidance was updated in this batch)
     - `selected_candidate` (index of the chosen candidate)
     - `verdict`, `reason`, `confidence`, `label_match`, `label_trust`
     - `eligible: bool`, `ineligible_reason?: str`
     - `warnings: [str]` (e.g., anomalous signals, low confidence, contradictory candidates)

**Example**:
- Group `QC-TEMP-20241206-0015502` has 4 candidates:
  - Candidate 0: `verdict="pass"`, `label_match=True`, `label_trust=0.85`, `confidence=0.92`
  - Candidate 1: `verdict="pass"`, `label_match=True`, `label_trust=0.82`, `confidence=0.88`
  - Candidate 2: `verdict="fail"`, `label_match=False`, `label_trust=0.45`, `confidence=0.65`
  - Candidate 3: `verdict="pass"`, `label_match=True`, `label_trust=0.80`, `confidence=0.90`
- Selection: Choose Candidate 0 (highest `label_trust` among GT-aligned candidates)
- Export: `selected_candidate=0`, `verdict="pass"`, `label_match=True`, `label_trust=0.85`, `warning=False`

**Key Properties**:
- **GT-aligned first**: Always prioritizes candidates that match ground-truth labels
- **Trust-weighted**: Uses trust scores to break ties when multiple GT-aligned candidates exist
- **Warning-based**: Logs warnings for problematic tickets but continues processing
- **Auditable**: Every selection includes `guidance_step` and `reflection_id` for provenance

## Alignment with Existing Work
- Complements open change `2025-10-24-add-grpo-group-reasoning`; this proposal supersedes its gradient-based scope for Stage-B and coordinates on mission definitions.
- Reuses Stage-A summary normalization logic; Stage-B now relies on LLM-driven evaluation (CriticEngine) with minimal deterministic signals for selection/tie-breaking.

## Key Decisions & Trade-offs
- **Fail-Fast Surfaces**: every phase inside `run_all()` validates its prerequisites (Stage-A summaries, prompt guidance file, verifier availability, writable run dir). Missing prerequisites abort the process instead of substituting defaults.
- **Storage**: rely solely on file artifacts (mission prompt guidance JSON, trajectory JSONL, selection JSONL, reflection logs). No complex exception queue; warnings are logged only.
- **LLM-Centred Optimisation**: prompt guidance updates always originate from the reflection LLM (using same checkpoint as Stage-B); deterministic signals and trust scores steer proposals. Proposals are applied directly (no replay validation) to keep the first version simple and practical.
- **Direct Application with Guardrails**: proposals may include up to `reflection.max_operations` operations and are applied atomically after validation (must leave at least one cue in the map and uphold atomic write/snapshot guarantees).
- **Global Experiences**: experiences are shared globally per mission. All samples in a mission see the same experiences, which are stored as numbered entries internally and formatted as a single text block when prepending to prompts.
- **Youtu-Agent Pattern**: experiences stored as numbered entries internally (e.g., `{"G0": "...", "G1": "..."}`), formatted as single text block for prompts (e.g., `"[G0]. ...\n[G1]. ..."`). CriticEngine outputs mirror Youtu-Agent’s `Summary` / `Critique` headings to ease prompt sharing across projects.
- **Simplicity vs. Analytics**: no vector stores or cue mining; analytics happen via logged trajectories + reflection history.
- **Batch Processing**: batch size 32 or 64 (configurable in YAML), shuffle data each epoch, skip last incomplete batch.
- **No Resumability**: resumability is not implemented; each run is independent.

## Implementation Details

### Output Directory Structure
- **Mission-specific outputs**: `{output.root}/{run_name}/{mission}/`
- **Files per mission**:
  - `trajectories.jsonl`: All trajectory records with signals and metadata
  - `selections.jsonl`: Final verdict selections for each group
  - `guidance.json`: Mission-specific guidance with experiences dict
  - `reflection.jsonl`: Reflection outcomes with debug info when parsing fails

### Debug Information Logging
When reflection parsing fails, debug information is automatically included in `reflection.jsonl` under the `reflection.debug_info` field. This eliminates the need for a separate debug file. The debug info includes:
- Full reflection response (not truncated by `stop_strings`)
- Parser exception message and JSON pointer that failed
- Response length and candidate count
- Mission and timestamp information

This debug information is stored temporarily in `_last_debug_info` during parsing and cleared after logging or on successful parse.

### Single-Process Model Sharing
The implementation uses a single model instance loaded in `runner.py`:
- Model loaded once with `device_map="auto"` for multi-GPU support
- Shared between `RolloutSampler`, `CriticEngine`, and `ReflectionEngine`
- No client-server architecture; direct model passing
- Enables efficient GPU memory usage

### Batch Processing Details
- Reflection triggers when `len(pending_records) >= config.reflection.batch_size`
- Incomplete batches at end of epoch are processed after epoch completion
- Each batch sees the same guidance step; updated guidance applies to next batch
- Reflection cycle counter increments after each reflection (batch or end-of-epoch)
- CriticEngine summaries/critiques are generated even for ineligible bundles so operators can inspect why reflection was skipped.

### Mission Guidance Setup
- Global guidance file at `config.guidance.path` contains mission-specific sections
- During setup, mission-specific guidance is extracted and copied to `{mission_dir}/guidance.json`
- Each mission maintains its own guidance file with step counter and experiences dict
- Guidance updates are applied directly to mission-specific files

## Open Questions / Follow-Ups
- Verifier plugins: rely solely on LLM-based verification (no separate verifier plugins needed).

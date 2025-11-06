# Training-Free Stage-B Design

## Architecture Overview
The updated Stage-B flow mirrors Youtu-Agent's training-free GRPO recipe: a frozen policy repeatedly samples candidates, judges them against ground-truth labels, then runs a **three-stage reflection pipeline** (SampleSummarizer → CritiqueScreener → GuidanceConsolidator) to synthesise the next guidance cue. Deterministic metrics exist only as context for the LLM—they never replace the reflection step, but they must be reliable across locale variants.

## Terminology Simplification
**Numbered Experiences** (Youtu-Agent pattern): Experiences are stored internally as numbered entries (e.g., `{"G0": "...", "G1": "..."}`) and formatted as a single text block when prepended to prompts. This aligns with Youtu-Agent's approach where experiences are stored as a dict internally but formatted as a single text block for prompts.

The final result after reflection is a formatted text block (e.g., `"[G0]. ...\n[G1]. ..."`) that is prepended to all prompts and helps the model perform better in the downstream task. This maintains the simplicity of a single text block in prompts while preserving the ability to track and evolve individual guidance points.

1. **Ingestion & Run Directory**
   - Consumes Stage-A JSONL outputs produced via `scripts/stage_a_infer.sh`.
   - Materialises a run directory (`{output.root}/{run_name}/{mission}/`) with mission-specific files: `trajectories.jsonl`, `selections.jsonl`, `selections.parquet`, `guidance.json`, `reflection.jsonl`.
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
     1. **SampleSummarizer**: transforms each selected candidate into a `{summary, critique}` pair following Youtu-Agent’s “analysis + fix” format (`summary` restates the winning rationale in 2–3 sentences; `critique` flags residual failure cues). The summariser stores its output alongside trajectories for downstream auditability.
     2. **CritiqueScreener**: inspects deterministic signals for the *winning* candidate and checks whether any group had consensus failure (`label_match=False` for all candidates). Only when at least one of those conditions is met does the bundle become eligible for reflection; otherwise it is skipped with an explicit `ineligible_reason`.
     3. **GuidanceConsolidator**: feeds the summariser output, candidate pool, and deterministic signals into the LLM and expects strict JSON: `{"action": "refine"|"noop", "summary": str, "critique": str, "operations": [...], "evidence_group_ids": [...], "uncertainty_note"?: str}`. No heuristic text fallback exists; truncation or malformed JSON is treated as a fatal error and surfaced up the stack.
   - **Experience parsing**: only the structured `operations` list (`{op: "upsert"|"remove"|"merge", key: str, text?: str, rationale?: str, evidence: [...]}`) is accepted. `_parse_experiences_from_text()` is removed, forcing prompt/template compliance. Evidence identifiers resolve to mission group ids (supporting `第N组` and raw ids).
   - **Incremental merge + gating**: Operations are staged and merged into the existing experiences dict only after holdout uplift validates the proposal (`apply_if_delta` threshold). `allow_uncertain=false` rejects proposals carrying `uncertainty_note`. Per-entry metadata (`reflection_id`, evidence sources, rationale, updated_at) is recorded for provenance. `remove` operations can only execute when at least one other entry remains; otherwise they fail validation.
   - **In-process model reuse**: Reflection uses the same model instance as rollout sampling, passed directly to `ReflectionEngine` (no client-server architecture). The model is loaded once in `runner.py` and shared between `RolloutSampler` and `ReflectionEngine`. This enables single-process execution with shared GPU resources and simplifies multi-GPU scaling via `device_map="auto"`.
   - **Batch boundary enforcement**: All samples in a batch see the same prompt guidance step; updates committed after holdout gating apply to subsequent batches in the same epoch, mirroring Youtu-Agent's mid-epoch evolution without intra-batch drift.
   - Applied changes bump the mission guidance step and append a `reflection.jsonl` entry with eligibility, uplift metrics, serialized operations, summary/critique text, and evidence group IDs.

6. **Selection, Warnings & Export**
   - `select_for_group` prioritises GT-aligned candidates with trust-weighted tie-breakers; if every candidate disagrees with guidance and verifier trust is low, the group is logged with a warning (no complex exception queue).
   - Warnings are logged with `group_id`, reason, and guidance step. All tickets continue to be processed; warnings are informational only.
   - `export_trajectories`, `export_selections`, and the new `export_reflection_inputs` include deterministic signals, confidence, guidance step, reflection id, warning flags, and the summariser’s `{summary, critique}` fields so downstream QC can identify tickets that may need review.

7. **Interfaces & Package Layout**
   - `src/stage_b/ingest.py`: load Stage-A outputs, hydrate prompt guidance, enforce run-dir setup, and fail fast when prerequisites are missing.
   - `src/stage_b/rollout.py`: run the sampler with explicit prompt guidance injection and persist trajectory logs keyed by step.
   - `src/stage_b/signals.py`: compute deterministic metrics, trust scores, and log warnings if any metric cannot be produced (no exception queue).
   - `src/stage_b/reflect.py`: orchestrate experience bundle creation, staged reflection prompts (summariser → screener → consolidator), strict JSON parsing, and warning logs. `_last_debug_info` is retained but now captures the full JSON parse error to help diagnose truncation.
   - `src/stage_b/select.py`: choose final verdicts using label-first policy with trust-weighted tie-breakers and warning flags.
   - `src/stage_b/export.py`: emit JSONL/Parquet artifacts and reflection logs.
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
  → export (JSONL/Parquet + reflection logs)

Note: Prompt guidance updates happen at batch boundaries; updated prompt guidance applies to the next batch (one-step lag). This mirrors Youtu-Agent's approach: prompt guidance evolves after each batch and applies globally to the next batch.
```

## Reflection Step: How It Works

**Purpose**: After each complete batch, the reflection engine analyzes the batch results and proposes an improved version of the prompt guidance text to help the model perform better on future batches.

**Process**:

1. **Build Experience Bundle** (after batch completes):
   - Collect all candidates from the batch with their signals:
     - **GT-aligned winners**: Candidates that match the ground-truth label (`label_match=True`)
     - **Conflicting candidates**: Candidates that disagree with GT or show inconsistent patterns
     - **Deterministic signals**: `label_match`, `self_consistency`, `candidate_agreement`, `confidence`, `label_trust`
     - **Trust summaries**: Aggregated statistics about candidate quality
     - **Stage-A deficiency notes**: Any issues with Stage-A summaries that may have affected performance

2. **LLM Reflection Call**:
   - Input to reflection LLM (same checkpoint as Stage-B model):
     - Current experiences formatted as text block (e.g., `"[G0]. ...\n[G1]. ..."`)
     - Experience bundle (winners, losers, signals, trust scores, deficiency notes)
   - LLM returns: `{action: "refine" | "noop", text: "formatted experiences text block...", evidence_group_ids: [...]}`. The reflection action is limited to `"refine"` (replace entire experiences dict) or `"noop"` (no change).
   - The returned text is parsed into numbered experiences using patterns like `[G0]. ...`, `[G1]. ...` to extract a dict `{"G0": "...", "G1": "..."}`. The implementation includes a `_parse_experiences_from_text()` function to extract numbered experiences from reflection response text.

3. **Apply Update** (direct application, no replay validation):
   - If proposal is accepted (`action="refine"`):
     - Parse the returned text into numbered experiences using `_parse_experiences_from_text()` (extract `[G0]. ...`, `[G1]. ...` patterns into `{"G0": "...", "G1": "..."}`)
     - Replace entire experiences dict with the parsed experiences (no add/refine/remove targeting logic)
     - Bump `guidance_step` counter (e.g., step 1 → step 2)
     - Update `updated_at` timestamp
     - Write updated `guidance.json` atomically
     - Append entry to `reflection.jsonl` with:
       - `reflection_id`, `mission`, `proposal`, `applied: true`, `guidance_step_before`, `guidance_step_after`, `evidence_group_ids`
   - Updated experiences apply **immediately to the next batch** (one-step lag)

**Example**:
- Batch 1 uses experiences at step 1: `{"G0": "Check for missing components and verify installation quality."}`
- Formatted as: `"[G0]. Check for missing components and verify installation quality."`
- After Batch 1 completes, reflection analyzes: "Many candidates missed subtle installation gaps mentioned in 备注."
- LLM proposes step 2: `"[G0]. Check for missing components and verify installation quality.\n[G1]. Pay special attention to 备注 field which often contains critical installation details that candidates overlook."`
- Parsed into: `{"G0": "Check for missing components and verify installation quality.", "G1": "Pay special attention to 备注 field which often contains critical installation details that candidates overlook."}`
- Batch 2 uses experiences at step 2 (updated experiences apply).

**Key Properties**:
- **Mid-epoch updates**: Updates happen within the epoch (not deferred to epoch boundaries), allowing faster feedback loops
- **Global per mission**: All samples in the mission see the same experiences step
- **Single change per reflection**: At most one experiences update per batch (simplifies first version)
- **Direct application**: No replay validation; operator can review reflection logs and rollback if needed
- **Youtu-Agent pattern**: Experiences stored as numbered entries internally, formatted as single text block for prompts

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

3. **Warning Logging**:
   - If no GT-aligned candidates exist or `label_trust` is very low:
     - Log warning with `group_id`, reason, and `guidance_step`
     - Continue processing (no exception queue)
     - Warning flag is set in export artifacts

4. **Stamping**:
   - Each selected verdict is stamped with:
     - `guidance_step` (which prompt guidance step was used)
     - `reflection_id` (if prompt guidance was updated in this batch)
     - `selected_candidate` (index of the chosen candidate)
     - `verdict`, `reason`, `confidence`, `label_match`, `label_trust`
     - `warning: bool` (if problematic ticket)

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
- Reuses Stage-A summary normalization logic, but Stage-B now owns rule management and deterministic judging.

## Key Decisions & Trade-offs
- **Fail-Fast Surfaces**: every phase inside `StageBPipeline.run_all()` validates its prerequisites (Stage-A summaries, prompt guidance file, verifier availability, writable run dir). Missing prerequisites abort the process instead of substituting defaults.
- **Storage**: rely solely on file artifacts (mission prompt guidance JSON, trajectory JSONL, selection JSONL/Parquet, reflection logs). No complex exception queue; warnings are logged only.
- **LLM-Centred Optimisation**: prompt guidance updates always originate from the reflection LLM (using same checkpoint as Stage-B); deterministic signals and trust scores steer proposals. Proposals are applied directly (no replay validation) to keep the first version simple and practical.
- **Direct Application with Guardrails**: single change per reflection survives only after validation that the operations leave at least one cue in the map and that atomic write/snapshot guarantees hold.
- **Global Experiences**: experiences are shared globally per mission. All samples in a mission see the same experiences, which are stored as numbered entries internally and formatted as a single text block when prepending to prompts.
- **Youtu-Agent Pattern**: experiences stored as numbered entries internally (e.g., `{"G0": "...", "G1": "..."}`), formatted as single text block for prompts (e.g., `"[G0]. ...\n[G1]. ..."`). Sample summariser output mirrors Youtu-Agent’s `Summary` / `Critique` headings to ease prompt sharing across projects.
- **Simplicity vs. Analytics**: no vector stores or cue mining; analytics happen via logged trajectories + reflection history.
- **Batch Processing**: batch size 32 or 64 (configurable in YAML), shuffle data each epoch, skip last incomplete batch.
- **No Resumability**: resumability is not implemented; each run is independent.

## Implementation Details

### Output Directory Structure
- **Mission-specific outputs**: `{output.root}/{run_name}/{mission}/`
- **Files per mission**:
  - `trajectories.jsonl`: All trajectory records with signals and metadata
  - `selections.jsonl`: Final verdict selections for each group
  - `selections.parquet`: Parquet version of selections (for analysis)
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
- Shared between `RolloutSampler` and `ReflectionEngine`
- No client-server architecture; direct model passing
- Enables efficient GPU memory usage

### Batch Processing Details
- Reflection triggers when `len(pending_records) >= config.reflection.batch_size`
- Incomplete batches at end of epoch are processed after epoch completion
- Each batch sees the same guidance step; updated guidance applies to next batch
- Reflection cycle counter increments after each reflection (batch or end-of-epoch)
- Sample summaries/critiques are generated even for ineligible bundles so operators can inspect why reflection was skipped.

### Mission Guidance Setup
- Global guidance file at `config.guidance.path` contains mission-specific sections
- During setup, mission-specific guidance is extracted and copied to `{mission_dir}/guidance.json`
- Each mission maintains its own guidance file with step counter and experiences dict
- Guidance updates are applied directly to mission-specific files

## Open Questions / Follow-Ups
- Verifier plugins: rely solely on LLM-based verification (no separate verifier plugins needed).

## High Priority: Schema & Type Updates (Blocking Spec Alignment)

- [x] **Update `MissionGuidance` schema in `src/stage_b/types.py`**: Change `version: int` to `step: int` and replace `guidance: Tuple[GuidanceEntry, ...]` with `experiences: Dict[str, str]` (numbered experiences dict: `{"G0": "...", "G1": "..."}`). Remove or deprecate `GuidanceEntry` and `GuidanceProvenance` types (keep only for internal provenance tracking if needed).
- [x] **Update `ReflectionAction` and `ReflectionProposal` in `src/stage_b/types.py`**: Change `ReflectionAction` to only allow `"refine" | "noop"` (remove `"add"` and `"remove"`). Remove `target_index` and `target_text` fields from `ReflectionProposal` (spec only supports `refine` operation that replaces entire experiences dict).
- [x] **Add `label_trust` signal in `src/stage_b/types.py`**: Add `label_trust: Optional[float]` field to `DeterministicSignals` dataclass. Update all places that create `DeterministicSignals` to include this field.
- [x] **Update `SelectionResult` and `ExperienceRecord` in `src/stage_b/types.py`**: Change `guidance_version` to `guidance_step` in `SelectionResult`, `ExperienceRecord`, and `ExperienceBundle` to match the new schema.

## Medium Priority: Implementation Logic Updates

- [x] **Update `GuidanceRepository` in `src/stage_b/guidance.py`**: Modify `load()` and `_write()` to parse/store `{step, updated_at, experiences: {...}}` schema instead of `{version, guidance: [...]}`. Update `apply_reflection()` to merge incremental operations (`upsert`/`remove`), track per-entry metadata (`reflection_id`, sources, rationale, updated_at), and expose a preview helper for holdout uplift checks.
- [x] **Update reflection parsing in `src/stage_b/reflect.py`**: Modify `_parse_reflection_response()` to parse the structured payload `{action, summary?, critique?, operations[], evidence_group_ids, uncertainty_note?}` and map short-form evidence identifiers to mission group ids. Preserve `_parse_experiences_from_text()` as a fallback when `operations` is absent. Update `reflect()` to skip trivial bundles, enforce `allow_uncertain`, and stage operations until holdout uplift is evaluated.
- [x] **Update prompt formatting in `src/stage_b/prompts.py`**: Modify `_render_guidance_snippets()` to accept `experiences: Dict[str, str]` instead of `GuidanceEntry` list. Change format to `"\n".join([f"[{i}]. {e}" for i, e in experiences.items()])` (numbered experiences format). Update `build_user_prompt()` to use experiences dict and ensure empty experiences dict causes abort (no empty fallback).
- [x] **Update reflection logging in `src/stage_b/reflect.py`**: Change `_append_log()` to use `guidance_step_before` and `guidance_step_after` instead of `guidance_version_before` and `guidance_version_after` in reflection log schema.

## Lower Priority: Runner & Signal Updates

- [x] **Update `runner.py` to use new schema**: Change all references from `guidance.version` to `guidance.step` throughout `src/stage_b/runner.py`. Update reflection outcome tracking to use `guidance_step` terminology.
- [x] **Update signal extraction in `src/stage_b/signals.py`**: Implement `label_trust` computation in signal extraction logic. Ensure all `DeterministicSignals` creation includes `label_trust` field. Log warnings when `label_trust` cannot be computed but continue processing.
- [x] **Verify reflection prompt template**: Review reflection prompt template file (from config) to ensure it matches spec requirements (includes formatted experiences text block, experience bundle information). Update `_build_reflection_prompt()` in `src/stage_b/reflect.py` if needed.

## Configuration & Documentation

- [x] **Extend Stage-B config/schema in `src/stage_b/config.py`**: Batch size is configurable via `ReflectionConfig.batch_size` (default: 16 in config YAML), shuffle seed is configurable via `RunnerConfig.shuffle_seed` (default: 17 in config YAML). Fail-fast enforcement is already implemented (raises RuntimeError on missing artifacts/invalid states). Config already mirrors these in `configs/stage_b_training_free.yaml`.
- [x] **Update `StageBPipeline.run_all()` in `src/stage_b/runner.py`**: Orchestration handles new experiences schema correctly (ingest → rollout → signals → reflect → apply → select/export). Batch processing logic verified: batch size configurable via `config.reflection.batch_size`, reflection triggers when `len(pending_records) >= config.reflection.batch_size` (line 191), incomplete batches at end of epoch are handled (lines 208-216).
- [x] **Enrich trajectory, selection, and reflection logs in `src/stage_b/export.py`**: Ensure logs include `guidance_step` (not `guidance_version`), reflection ids, evidence group ids, and warning flags. Verify all export artifacts use consistent schema terminology.

## Testing & Documentation

- [x] **Add unit tests in `tests/stage_b/`**: Added unit tests for experience parsing logic (`test_experience_parsing.py`), simplified reflection action parsing (`test_reflection_parsing.py`), label_trust signal computation (`test_label_trust.py`), updated guidance repository schema (`test_guidance_repository.py`), and prompt formatting with numbered experiences (`test_prompt_formatting.py`). Added integration smoke test covering proposal accept/reject paths (`test_reflection_integration.py`).
- [x] **Document the operator workflow in `docs/REFERENCE.md`**: Updated documentation to reflect numbered experiences format, simplified reflection workflow, and new schema terminology (step instead of version).

## Additional Implementation Details

- [x] **Debug information logging**: Implemented `_last_debug_info` mechanism to store debug information when reflection parsing fails. Debug info is included in `reflection.jsonl` under the `reflection.debug_info` field, eliminating need for separate debug file.
- [x] **Single-process model sharing**: Model is loaded once in `runner.py` and shared between `RolloutSampler` and `ReflectionEngine` for efficient GPU usage.
- [x] **Mission-specific output structure**: Outputs organized by mission under `{output.root}/{run_name}/{mission}/` with mission-specific guidance files.
- [x] **Batch processing**: Reflection triggers when batch size reached; incomplete batches processed after epoch completion.
- [x] **Experience sorting**: Experiences are sorted by key when formatting to ensure consistent ordering in prompts.

## Newly Identified Gaps (To Do)

- [x] **Normalize verdict signals**: Update `src/stage_b/signals.py` (and helpers) so `label_match` compares normalised verdicts across English and Chinese aliases, with unit coverage for mixed-language trajectories.
- [x] **Fix reflection eligibility gating**: Refactor `ReflectionEngine` to gate on the selected candidate plus an explicit all-wrong shortcut, and surface ineligible reasons in logs/exports.
- [x] **Enforce non-empty guidance + atomic snapshots**: Harden `GuidanceRepository` to reject empty experience maps, write via temp file rename, and generate microsecond-resolution snapshot names.
- [x] **Require structured JSON reflection payloads**: Remove `_parse_experiences_from_text()` fallback, adjust prompt stop strings, and plumb parser errors through `_last_debug_info`.
- [x] **Introduce SampleSummarizer / Critique artifacts**: Add per-ticket `summary` and `critique` generation aligned with Youtu-Agent style, persist them with trajectories and reflection bundles, and expose in exports.
- [x] **Update exports/config/docs**: Extend selections/reflection logs, config schema, and operator docs to cover summary/critique fields, eligibility reasons, and new guidance guardrails.
- [ ] **Add regression tests**: Cover verdict normalisation, guidance non-empty enforcement, atomic snapshot rotation, JSON parse failures, and summary/critique export wiring.

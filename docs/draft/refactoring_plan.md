# Refactoring Plan (post-RL)

## Objective
Reduce redundancy and drift risk in summary-mode prompts, summary sanitization, and mode/template validation while preserving current behavior (including the single-line "无关图片" contract).

## Constraints
- Behavior-preserving only (no prompt text changes, no new rules).
- Keep changes isolated from new RL modules.
- Any architectural or API change should be proposed via OpenSpec before implementation.

## Priority (most beneficial, lowest risk)

### P1 — Centralize summary sanitization
**Why**: Two similar implementations already diverge; highest risk of inconsistent "无关图片" handling and JSON cleanup.

**Targets**:
- `src/stage_a/inference.py` (summary sanitization helpers)
- `src/stage_a/postprocess.py` (summary sanitization helpers)

**Approach**:
- Extract a shared helper module (e.g., `src/stage_a/summary_sanitize.py` or `src/utils/summary_sanitize.py`).
- Preserve current behavior with explicit flags (e.g., `strip_remark`, `drop_format_version`, `allow_json_rewrite`).
- Replace both call sites with the shared helper.

**Non-goals**:
- No behavior change to JSON handling or category filtering.

---

### P2 — Unify summary prompt composition
**Why**: Summary prompt logic is split across core, profile, Stage-A, and config modules; easy to drift when updating irrelevant summary rules.

**Targets**:
- `src/prompts/summary_core.py`
- `src/prompts/summary_profiles.py`
- `src/prompts/stage_a_summary.py`
- `src/config/prompts.py`

**Approach**:
- Promote a single canonical builder in `summary_profiles.py` (or a new `summary_factory.py`).
- Stage-A prompt builders should call the canonical builder and only add Stage-A-specific blocks.
- Keep the text identical (string equality) to current outputs.

---

### P3 — Centralize mode/template validation and prompt resolution
**Why**: Mode resolution and template compatibility checks are duplicated in SFT + fusion dataset code; inconsistent errors and drift risk.

**Targets**:
- `src/sft.py`
- `src/datasets/unified_fusion_dataset.py`
- `src/datasets/wrappers/__init__.py`

**Approach**:
- Add a small helper (e.g., `resolve_mode_and_prompts(...)`) that returns:
  - resolved mode
  - resolved system/user prompts
  - prompt source (default/domain/dataset)
- Use helper consistently in SFT and fusion datasets.

---

### P4 — Share Stage-B summary parsing + ordering utilities
**Why**: Stage-B contains multiple ad-hoc JSON parsing and summary ordering helpers that can drift and subtly change behavior.

**Targets**:
- `src/stage_b/sampling/prompts.py`
- `src/stage_b/reflection/engine.py`

**Approach**:
- Extract shared helpers (e.g., `parse_summary_json`, `format_summary_json`, `sorted_stage_a_summaries`, `estimate_obj_count`) into `src/stage_b/utils/summary.py`.
- Replace ad-hoc JSON detection in reflection/engine with the shared parser.
- Keep exact behavior (ordering, required keys, formatting).

---

### P5 — Unify dataset JSONL loading + validation + annotation
**Why**: BaseCaptionDataset and FusionCaptionDataset each load/validate/annotate records with overlapping logic and slightly different error paths.

**Targets**:
- `src/datasets/dense_caption.py` (from_jsonl path)
- `src/datasets/unified_fusion_dataset.py` (`_load_records`, `_annotate_record`)
- `src/datasets/fusion.py` (`_annotate_record`)
- `src/datasets/utils.py` (new helper)

**Approach**:
- Add a shared helper (e.g., `load_jsonl_validated(...)`) to handle:
  - `resolve_relative` image paths
  - optional `sample_limit`
  - `validate_conversation_record` (and error prefixing)
  - optional `annotate_metadata` callback (for `_fusion_source`/template/mode).
- Replace per-site loaders with the shared helper; preserve error messages.

---

## Secondary cleanups (safe, low effort)

### S1 — Remove unreachable cleanup code
**Target**:
- `src/datasets/dense_caption.py`

**Approach**:
- Remove unreachable `kwargs.pop(...)` after raising `summary_ratio` error.

### S2 — Centralize `_fusion_source` label resolution
**Why**: Repeated inline resolution logic; trivial dedup.

**Targets**:
- `src/data_collators/dataset_metrics.py`
- `src/datasets/dense_caption.py`

**Approach**:
- Add a tiny helper like `resolve_fusion_label(record, default)` in `src/datasets/utils.py` or `src/utils/metadata.py`.
- Replace inline copies.

### S3 — Template system mutation guard
**Why**: Defensive, improves clarity and reduces side effects.

**Targets**:
- `src/datasets/dense_caption.py`
- `src/datasets/unified_fusion_dataset.py`

**Approach**:
- Add a context manager (e.g., `template_system_context(template, system_prompt)`) to ensure restore.
- Behavior remains identical.

### S4 — Consolidate boolean parsing helpers
**Why**: Multiple near-identical bool coercers increase drift and inconsistent error messages.

**Targets**:
- `src/config/loader.py`
- `src/config/schema.py`
- `src/stage_b/config.py`

**Approach**:
- Create a shared `coerce_bool(value, field_name)` helper in a small utility module and reuse it.
- Preserve current accepted literals and error text as much as possible.

### S5 — Consolidate deprecated-field guards
**Why**: Repeated checks for removed fields risk inconsistent messaging and missed paths.

**Targets**:
- `src/config/loader.py` (`summary_ratio`)
- `src/config/schema.py` (`summary_ratio`, `summary_label_grouping`)
- `src/datasets/fusion.py` (`summary_label_grouping`)
- `src/datasets/dense_caption.py` (`summary_ratio`)

**Approach**:
- Centralize deprecated-field validation in a small helper and call it early in config parsing.

### S6 — Unify assistant-prefix resolution
**Why**: Base and fusion datasets have similar logic to construct assistant prefixes; hard to keep consistent.

**Targets**:
- `src/datasets/dense_caption.py`
- `src/datasets/unified_fusion_dataset.py`

**Approach**:
- Add a helper in `src/datasets/assistant_prefix.py` to build prefixes from dataset key + mode.
- Reuse in both datasets; preserve error messages.

### S7 — Centralize summary-category constants and regexes
**Why**: Summary sanitization duplicates category sets and regex helpers across Stage-A inference and postprocess.

**Targets**:
- `src/stage_a/inference.py`
- `src/stage_a/postprocess.py`

**Approach**:
- Move `_BBU_CATEGORIES`, `_RRU_CATEGORIES`, group prefix regexes, and remark stripping into the shared sanitization helper.

---

## Alignment with `update-irrelevant-summary-format`
- Preserve the single-line "无关图片" contract in prompts and any sanitization logic.
- Ensure summary sanitization does not assume two-line output.

---

## Suggested order after RL module merge
1. P1 (summary sanitization)
2. P2 (summary prompt unification)
3. P3 (mode/template resolution)
4. P4 (Stage-B summary utils)
5. P5 (JSONL load/validate/annotate)
6. S1–S7 (quick cleanups)

---

## Minimal validation checklist
- Existing summary prompt strings unchanged (byte-for-byte).
- Stage-A inference + postprocess produce identical outputs for known samples.
- Fusion dataset training + eval still resolve prompts/modes identically.
- No changes to data contract or dataset sampling.

---

## Notes
- If any of P1–P5 is expanded beyond pure refactor, create an OpenSpec change before editing.

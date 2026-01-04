# Design: GRPO Rewards Refactor + Config Standardization

## Goals
- Consolidate duplicated GRPO reward parsing/gating logic into shared utilities.
- Make reward identifiers explicit and namespaced (`summary.*`) with no legacy aliases.
- Move GRPO-only config validation (notably CHORD) into a single validator with typed config support.
- Keep reward semantics unchanged while improving maintainability.

## Non-Goals
- Any changes to reward definitions or GRPO optimization behavior.
- Changes to ms-swift trainer interfaces.
- Compatibility shims for old identifiers or old config paths.

## Proposed Module Layout
```
src/rlhf/grpo/
  __init__.py                # explicit register_grpo_rewards()
  rewards/
    registry.py              # registry + name constants
    summary/
      parsing.py             # text extraction, strict header checks, JSON parsing helpers
      context.py             # SummarySample dataclass (pre-parsed sample)
      facts.py               # fact extraction + F1/Tversky helpers
      rewards.py             # reward classes (score(SummarySample))
```

### SummarySample (context)
Pre-parse each completion into a `SummarySample` that carries:
- `raw_text`, `lines`, `is_irrelevant`, `domain_token`
- `strict_json_line` (only when the header is valid)
- `pred_json` and `ref_json` (optional; parse only if requested)
- `summary_ref` and normalized summary metadata

This eliminates repeated parsing blocks and keeps reward classes focused on scoring logic.

### Reward Registry
- Replace implicit import-time registration with a single explicit `register_grpo_rewards()` in `src/rlhf/grpo/__init__.py`.
- Register namespaced identifiers (see mapping below) to reward classes.
- `src/sft.py` calls `register_grpo_rewards()` exactly once before trainer initialization.

## Reward Identifier Scheme
New identifiers use a dot namespace:
- `summary.format`
- `summary.header`
- `summary.strict`
- `summary.parse`
- `summary.no_dup_keys`
- `summary.dataset`
- `summary.objects_total`
- `summary.objects_total_lb`
- `summary.category_recall`
- `summary.category_f1`
- `summary.content_eq` (exact JSON equivalence)
- `summary.content_f1`
- `summary.content_structured_tversky`
- `summary.text_bbu`
- `summary.notes_bbu`
- `summary.notes_presence`
- `summary.group_stats_presence`

Legacy names (e.g., `summary_format`) are removed.

## Config Surface Changes
- Move CHORD config from `custom.grpo_chord` to `custom.grpo.chord`.
- Introduce typed config for GRPO-specific fields (e.g., `GrpoConfig`, `GrpoChordConfig`) and validate centrally.
- Validation includes:
  - reward_funcs/weights length alignment
  - num_generations divisibility
  - CHORD required fields when enabled

## Config Migration (No Back-Compat)
- Update `configs/grpo/*` to the new reward identifiers and `custom.grpo.chord`.
- Remove any legacy config paths and ensure validation fails on old forms.

## Testing
- Update reward tests to import from new modules and validate the same behavior.
- Add focused tests for shared parsing/context helpers if not already covered.
- Run config validation + `pytest tests/test_summary_grpo_rewards.py`.

## Duplication Reduction Approach
- Consolidate per-reward parsing and gating logic into shared helpers (`SummarySample`, parsing utilities, and fact helpers).
- Keep each reward implementation focused on scoring logic only (no repeated parsing blocks).

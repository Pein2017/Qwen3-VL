# Proposal: Remove `objects_total` From Summary JSON Contract

## Why
- `objects_total` is a global count that the model must commit to early in generation, while the structured `统计` details often require longer decoding and can invalidate the early count.
- The value is mechanically derivable post-hoc (from a summary object model or heuristic) and does not need to be authored by the LLM, reducing token waste and format failure risk.
- For training, forcing `objects_total` encourages a spurious objective (global count accuracy) that is known to be imperfect for small objects, while the primary business signal is structured per-category/per-attribute counts under `统计`.

## Scope
- **Data contract change (breaking)**: summary-mode JSON for BBU/RRU removes the top-level `objects_total` field.
  - Required keys become: `dataset`, `统计`.
  - Optional keys remain: `备注` (BBU), `分组统计` (RRU).
- **data_conversion**: summary builder no longer emits `objects_total`.
- **src/**:
  - Stage-A summary prompt contract stops requiring `objects_total`.
  - Stage-A parsing/sanitization does not depend on `objects_total`; any legacy `objects_total` key is ignored/stripped when encountered.
  - Stage-B prompt construction no longer requires `objects_total` in the incoming summary JSON; any “image complexity” hint is derived without relying on an LLM-authored count.
  - Summary GRPO config/rewards stop depending on `objects_total` (remove `summary.objects_total*` from the default reward stack).
- Update tests and documentation for the new schema.

## Non-Goals
- Changing the semantics of `统计` counting or the key=value desc contract.
- Adding new reward functions for counting (this change only removes the `objects_total` objective).
- Providing backward-compatible training or runtime support for legacy summaries that include `objects_total`.

## Compatibility
- **Breaking change** for any consumer that requires `objects_total` to exist in summary JSON.
- Existing JSONL corpora MUST be regenerated so the `summary` field no longer includes `objects_total`.

## Risks
- Some Stage-B prompt variants currently use `objects_total` as an “image complexity” hint. Removing it may reduce robustness unless an alternative estimate is provided.
- Existing datasets on disk still include `objects_total` in `summary`; training on those JSONLs would teach the legacy schema.

## Success Criteria
- Summary-mode training targets no longer include `objects_total`.
- Stage-A summary prompt and validators no longer require `objects_total`.
- Stage-B prompt building succeeds on summaries without `objects_total` (and remains robust for migrated data).
- Targeted unit tests pass (`pytest tests/test_summary_grpo_rewards.py`, Stage-B prompt tests, and JSONL conversion tests).

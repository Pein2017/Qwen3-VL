# Proposal: Remove Legacy Total-Count Field From Summary JSON Contract

## Why
- The legacy total-count field is a global count that the model must commit to early in generation, while the structured `统计` details often require longer decoding and can invalidate the early count.
- The value is mechanically derivable post-hoc (from a summary object model or heuristic) and does not need to be authored by the LLM, reducing token waste and format failure risk.
- For training, forcing a global count encourages a spurious objective (count accuracy) that is known to be imperfect for small objects, while the primary business signal is structured per-category/per-attribute counts under `统计`.

## Scope
- **Data contract change (breaking)**: summary-mode JSON for BBU/RRU removes the legacy top-level total-count field.
  - Required keys become: `dataset`, `统计`.
  - Optional keys remain: `备注` (BBU), `分组统计` (RRU).
- **data_conversion**: summary builder no longer emits the legacy total-count field.
- **src/**:
  - Stage-A summary prompt contract stops requiring the legacy total-count field.
  - Stage-A parsing/sanitization does not depend on legacy total-count keys.
  - Stage-B prompt construction no longer requires a total-count field in the incoming summary JSON; any “image complexity” hint is derived without relying on an LLM-authored count.
  - Summary GRPO config/rewards stop depending on the legacy total-count rewards.
- Update tests and documentation for the new schema.

## Non-Goals
- Changing the semantics of `统计` counting or the key=value desc contract.
- Adding new reward functions for counting (this change only removes the legacy total-count objective).
- Providing backward-compatible training or runtime support for legacy summaries that include a total-count field.

## Compatibility
- **Breaking change** for any consumer that requires a total-count field to exist in summary JSON.
- Existing JSONL corpora MUST be regenerated so the `summary` field no longer includes the total-count field.

## Risks
- Some Stage-B prompt variants currently use a total-count hint for “image complexity”. Removing it may reduce robustness unless an alternative estimate is provided.
- Existing datasets on disk still include a total-count field in `summary`; training on those JSONLs would teach the legacy schema.

## Success Criteria
- Summary-mode training targets no longer include a total-count field.
- Stage-A summary prompt and validators no longer require a total-count field.
- Stage-B prompt building succeeds on summaries without the total-count field (and remains robust for migrated data).
- Targeted unit tests pass (`pytest tests/test_summary_grpo_rewards.py`, Stage-B prompt tests, and JSONL conversion tests).

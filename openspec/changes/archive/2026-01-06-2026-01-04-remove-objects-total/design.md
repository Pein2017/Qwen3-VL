# Design: Removing Legacy Total-Count Field From Summary JSON

## Goals
- Remove the legacy total-count field from the summary JSON contract to reduce token waste and early-commit errors.
- Keep summary-mode training focused on structured per-value counts under `统计`.
- Preserve Stage-B prompt robustness without requiring an LLM-authored global count.

## Non-Goals
- Introducing a new counting reward or changing `统计` semantics.
- Providing backward-compatible training inputs that still include a total-count field.

## New Summary JSON Schema

### Required keys
- `dataset` (string; `BBU` or `RRU`)
- `统计` (list)

### Optional keys (unchanged)
- `备注` (BBU-only; list of strings; may be absent or empty)
- `分组统计` (RRU-only; dict of group_id → count; may be absent)

### Removed key
- The legacy total-count field is removed from the schema and from training targets.

## Stage-B “image complexity” hint (replacement)
Some Stage-B prompts include `ImageN(obj=...)` as a soft signal.

Options:
1) **Remove the hint** entirely.
2) **Derive an estimate** from the structured summary without requiring a total-count field.

Default approach for this change: **derive an estimate** so Stage-B retains the signal without token cost.

Proposed estimator (deterministic, best-effort):
- Parse the summary JSON (if parseable).
- For each `统计` entry:
  - If the entry has no counted attributes, treat presence as `1`.
  - Otherwise, for each attribute dict `attr: {value: count, ...}` compute the attribute total as the sum of its counts.
  - Take the per-category estimate as the max attribute total across attributes.
  - Special-case `类别=标签`: estimate as `文本_total + 可读性_total` (readable vs unreadable are disjoint in the prompt contract), and take the max with the generic estimate.
- Sum category estimates across entries to produce `obj_estimate`.

This intentionally trades exactness for robustness; it is only used as a soft signal in Stage-B prompts.

## Training / Reward Implications
- Remove legacy total-count rewards from the default GRPO reward stack used by summary GRPO configs.
- Keep `summary.content_structured_tversky` as the primary count-weighted structured objective under `统计`.

## Migration Plan
- No backward-compatibility is provided for training corpora: JSONL files MUST be regenerated so `summary` omits the total-count field.
- data_conversion emits the new schema by default.

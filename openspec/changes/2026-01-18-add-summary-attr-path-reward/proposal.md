# Proposal: Add Summary Attr-Path Reward For Schema Completeness

## Why
Stage-B decisions are sensitive to whether Stage-A summary outputs include *complete* attribute structure for detected objects.

We see frequent cases where a model "sees" an object and emits the category, but omits critical nested attributes (e.g., `电线.捆扎.整齐`, `光纤.弯曲半径.半径合理`, `可见性.完整/部分`, `符合性.符合/不符合`). This creates noise and ambiguity downstream: Stage-B cannot reliably distinguish "not present" vs "present but unspecified".

The user preference is:
- It is acceptable for the model to omit an object (treat as "not seen") when uncertain.
- It is **not** acceptable to emit an object category while failing to provide complete nested attribute structure for stable semantic fields.

## What Changes
Add a new optional summary-mode GRPO reward that measures nested attribute *path* coverage (depth=2) for a curated set of stable semantic parent keys (based on Stage-A rollout observations).

Key properties:
- Works for both BBU and RRU summary modes (uses `metadata.summary_ref` as ground truth).
- Focuses on nested semantic keys like `捆扎/整齐`, `可见性/完整`, `符合性/符合`, etc.
- Excludes free-text fields (`文本`) and dynamic keys (pure digits, `key=value`-style, long OCR-like tokens).
- Designed as a completeness signal: it rewards nested-path coverage for categories the model chooses to emit **when those categories also exist in the ground-truth summary** (to align with the preference "better omit than half-describe").

## Impact
- No breaking changes: existing reward IDs and configs continue to work.
- New reward ID can be enabled via YAML to improve schema completeness and reduce downstream ambiguity.

## Out of Scope
- Stage-B prompt/guidance changes.
- Changing the definition/semantics of existing rewards.

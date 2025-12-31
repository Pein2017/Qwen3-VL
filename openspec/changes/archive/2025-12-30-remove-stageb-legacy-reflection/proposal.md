# Proposal: remove-stageb-legacy-reflection

## Why
Legacy reflection mode is deprecated in this repository. Keeping both paths increases maintenance cost and can cause accidental usage. The project now needs Stage‑B to support rule-search only.

## What
- Remove legacy_reflection mode support from Stage‑B config/schema and runner paths.
- Simplify Stage‑B runtime and artifacts to rule-search only.
- Update docs/runbooks to reflect rule-search as the only supported Stage‑B mode.

## Impact
- Breaking change: configs that rely on legacy_reflection or `sampler` will no longer load/run.
- Stage‑B entrypoints will reject legacy mode and legacy-only fields.
- Documentation and operational guidance will be updated accordingly.

## Rollout / Validation
- Update Stage‑B configs to remove legacy-only fields.
- Run a smoke rule-search config to verify no regression in Stage‑B startup and outputs.

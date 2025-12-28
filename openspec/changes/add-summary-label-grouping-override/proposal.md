# Proposal: add-summary-label-grouping-override

## Summary
Introduce an optional per-dataset override in fusion configs to control summary label grouping (标签/* → 标签/可以识别) on a dataset-by-dataset basis, while keeping the existing global `custom.summary_label_grouping` default.

## Motivation
Summary-mode training currently applies label grouping globally. For fused summary configs, only BBU/RRU summaries should preserve exact OCR labels, while other summary datasets can keep the default behavior. This requires a dataset-scoped toggle in YAML without adding CLI flags.

## Scope
- **In**: Fusion config schema, dataset spec propagation, summary preprocessor selection per dataset, and config updates for BBU/RRU summary entries.
- **Out**: CLI changes, non-fusion datasets, or changes to summary prompt content.

## Impact
- Backward-compatible: if the per-dataset field is absent, behavior remains controlled by `custom.summary_label_grouping`.
- Only summary-mode datasets are affected; dense mode ignores the toggle.

## Rollout
- Update fusion YAML to disable grouping for BBU/RRU summary datasets.
- Keep irrelevant-summary unchanged (no 标签/* entries).

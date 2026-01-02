# Proposal: Multi-target fusion with ratio-balanced targets

## Problem
- Fusion currently accepts a single `target` and treats all other datasets as `sources` with ratios tied to that one target.
- Configs like `configs/fusion/bbu_rru_lvis_coig.yaml` must mislabel true targets (e.g., RRU) as sources, so they lose target-only behaviors (augmentation, curriculum, eval inclusion).
- No way to balance multiple targets of different sizes; majority targets dominate steps, hurting minority coverage and eval comparability.

## Goals
- Accept multiple targets in fusion configs (backward compatible with existing single-target configs).
- Enable optional per-target `ratio` to downsample larger targets each epoch toward a shared baseline derived from the smallest target.
- Keep target-only behaviors (augmentation, curriculum, eval) for all targets; sources remain aux-only for training loss.
- Preserve deterministic scheduling and source quota semantics (source quota scales with total target quota).

## Non-Goals
- No change to templates or prompt resolution priority.
- No change to object format/contract or augmentation algorithms.
- No new source eval flows beyond existing target-only default.

## Success Criteria
- A config with `targets: [bbu, rru]` and sources `lvis`, `lang_chat` loads without code changes to caller APIs.
- Per-epoch plan shows balanced target quotas when ratios are provided; sources sampled off combined target quota.
- Eval iterates all target val splits and excludes sources.
- Backward compatibility: existing single-target fusion configs run unchanged.

## Risks / Mitigations
- Ratio math errors could under/over-sample: add unit tests for quota computation and seed determinism.
- Name collisions across targets/sources: add validation and explicit errors.
- Regression in telemetry: extend epoch_plan / last_sample_debug to include dataset and quota info for debugging.

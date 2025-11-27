# Proposal: Remove Hard-Sample Mining and restore packed SFT defaults

## Why
- Experiments (2025-11-26) show token-acc-based hard-sample mining brings no quality gain, likely due to annotation noise.
- The feature blocks `training.packing` and adds per-sample metadata plumbing and callbacks that complicate training stability.
- Returning to standard packed SFT simplifies configs and avoids accidental coupling.

## What
- Deprecate `custom.hard_sample_mining` entirely; validation fails fast with guidance.
- Drop mining callbacks, dataset metadata hooks, and external schedule plumbing.
- Remove HSM examples/tests and refresh docs to highlight removal and packed training path.

## Scope
- Config schema, `src/sft.py`, datasets, callbacks, docs (`REFERENCE`, `TRAINING_PLAYBOOK`, `UNIFIED_FUSION_DATASET`), sample configs/tests.

## Non-Goals
- No replacement mining strategy; no offline mining; no change to augmentation or fusion ratios beyond removing HSM hooks.

## Risks / Mitigations
- **Legacy configs break**: provide explicit validation error and doc note.
- **Unexpected imports**: ensure missing callback module errors clearly indicate removal.


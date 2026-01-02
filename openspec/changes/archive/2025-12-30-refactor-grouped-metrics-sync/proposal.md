# Proposal: Refactor grouped metric sync to avoid double reduction

## Problem
Per-group fusion metrics are now visible on rank0, but the current sync path performs per-step all-reduce with possibly mismatched metric orderings, adding communication cost and risking cross-metric bleed/inflated values.

## Goal
Make grouped metric synchronization deterministic and single-pass by reusing Swift's existing logging-time reduction while keeping rank0 visibility for small groups.

## Scope
- Restrict `_sync_group_metrics` to key-union only (deterministic order), letting `MeanMetric.compute()` handle aggregation.
- Add regression coverage for distributed key sync without per-step all-reduce.
- Update specs/docs to codify the single-reduction behavior and ordering requirement.

## Out of Scope
- Changing metric definitions or logging cadence.
- Altering upstream ms-swift behavior.
- Adding new datasets or revisiting packing semantics (packing is not supported).

## Success Criteria
- Rank0 logs still include groups that appear only on other ranks.
- No duplicate/double reduction; per-group values match single-process runs.
- New spec/tests describe and protect deterministic key sync behavior.

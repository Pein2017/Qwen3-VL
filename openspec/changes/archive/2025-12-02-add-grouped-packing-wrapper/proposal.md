# Proposal: Group-aware packing wrapper without upstream edits

## Problem
- Enabling `packing` in ms-swift merges samples from different datasets into one packed sequence, losing dataset identity and preventing per-dataset loss monitoring.
- We cannot modify the installed ms-swift source; changes must live in this repo.

## Goal
- Provide an opt-in, group-aware packing path in our repo that keeps packing efficiency while preserving dataset labels for loss logging.

## Scope
- Add a local Packing wrapper that buckets sequences by a group key (e.g., `_fusion_source`) and packs only within that bucket.
- Propagate the group label into the packed batch for metrics.
- No upstream/ms-swift source edits; default behavior unchanged when the feature is off.
- Ignore drop_last for now (keep all partial packs).

## Success Criteria
- A config flag (e.g., `packing_group_key`) turns on grouped packing; when unset, behavior matches current ms-swift packing.
- Packed batches never mix group labels; per-dataset loss metrics can be logged.
- Packing remains deterministic and performant.

## Risks / Mitigations
- Performance regression from per-group binpacking: mitigate by minimal fork mirroring ms-swift logic and by keeping grouping optional.
- Drift vs upstream: keep the wrapper small and isolated; reuse ms-swift utilities where possible.

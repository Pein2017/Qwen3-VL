# Proposal: Remove packing path and enforce padding-only training

## Problem
- Packing is expensive and unreliable in our current pipeline; it slows CPU preprocessing and complicates augmentation.
- We want padding-only training as the enforced default, with clear rejection of any `training.packing=true` usage.
- Packing code, configs, docs, and tests linger in the tree (including cached-length tooling and grouping knobs), creating confusion and maintenance cost.

## Goals (in scope)
- Ban packing in the runtime: if `training.packing` is true or packing knobs are present, fail fast with actionable guidance to use standard padding.
- Remove/isolated packing code paths (datasets, collators, trainers, length caches) so the padded path is the only active path.
- Clean configs to be padding-only (no packing keys, no cached-length references).
- Prune packing docs/design details and mark packing as removed/experimental future work.
- Update OpenSpec to reflect packing removal and supersede prior packing changes.
- Remove packing-focused tests while keeping padded-path coverage.

## Non-goals (out of scope)
- Designing a new or replacement packing algorithm.
- Performance tuning beyond removing packing-related overhead.
- UI/visualization changes.

## Impact / risks
- Users relying on packing must switch to padding; mitigation is a clear error with migration note.
- Removing code reduces maintenance surface and removes accidental activation risk.

# Proposal: Packing optimizations (rank0 broadcast + cached lengths)

## Problem
- Packing still runs a full augmentation+encode pre-pass each epoch; even with rank0 broadcast, that pre-pass is expensive and duplicated work relative to training-time augmentation.
- Related specs are split across multiple changes, making it hard to implement and ship together.

## Goal
- Consolidate packing optimizations into one change: rank0 broadcast to eliminate per-rank pre-pass, plus cached-length packing to skip augmentation during pack build while keeping training-time augmentation.

## Scope
- Group requirements for:
  - Rank0 pack construction + broadcast to all ranks in DDP.
  - Exact cached-length packing (opt-in, validated cache) so pack build avoids augmentation/image loading.
- Keep default behavior unchanged when features are disabled.

## Success Criteria
- Single change/spec covering both broadcast and cached-length capabilities.
- DDP runs build packs once on rank0 and broadcast results.
- When cached-length mode is enabled and cache is valid, pack build skips augmentation/encode; training still augments per sample.
- Cache hash validation prevents stale lengths.

## Risks / Mitigations
- Stale cache → hash/version guard + clear fail/fallback guidance.
- Large broadcast payload → pack metadata only (indices/lengths/groups/domains).
- Backward compatibility → opt-in flags; defaults match current behavior.

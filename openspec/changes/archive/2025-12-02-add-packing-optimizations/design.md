# Design: Packing optimizations (broadcast + cached lengths)

## Components
1. **Rank0 broadcast**: rank0 builds `_packs` and `group_domains`, broadcasts metadata via `torch.distributed` to other ranks after pack build and per-epoch rebuild; barrier to sync before iteration.
2. **Cached-length packing**: opt-in mode where `_build_packs` reads precomputed exact lengths (no augmentation/image load) when a valid cache is present.

## Cache design (exact lengths)
- Cache generation tool runs full augmentation + template encode (`return_length=True`) to compute per-sample lengths and writes:
  - `length` values
  - cache hash/version = hash(augmentation config, template id/version, dataset fingerprint)
- Validation: packing checks hash; mismatch → error or fallback per policy (configurable).
- No approximations allowed.

## Control flow
- Config flags:
  - `custom.cached_lengths.enabled` (opt-in)
  - `custom.cached_lengths.fail_on_miss` (or similar) to choose error vs fallback on missing/invalid cache
- Pack build:
  - If cache enabled & valid → use cached lengths, skip augmentation/encode
  - Else → current behavior (full pre-pass) with guidance
- DDP:
  - Rank0-only pack build (cache or not), then broadcast packs/domains; others never run pre-pass.

## Validation & Tests
- Unit: cache hit (no dataset augmentation calls during pack build), cache miss/mismatch handling, DDP broadcast with cached lengths.
- Integration: startup time improvement with cache enabled, training still sees augmentation variability.

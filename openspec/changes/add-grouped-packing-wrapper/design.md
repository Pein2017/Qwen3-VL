# Design: Group-aware packing (local wrapper)

## Approach
- Implement a local `GroupedPackingDataset` / `GroupedIterablePackingDataset` in this repo (e.g., `src/packing/grouped_packing.py`).
- Mirror ms-swift PackingDataset behavior but add:
  - `group_key` (string): field in each encoded row used to bucket sequences; packs are built within a single group only.
  - `packed_group` attached to the packed row (single value per pack).
- Keep `drop_last` out-of-scope: retain partial packs to avoid data loss.
- Determinism: preserve the existing ordering/shuffle behavior; binpacking remains deterministic given fixed input order and seeds.

## Integration
- In `src/sft.py` (or dataset factory), when `packing_group_key` is set:
  - Ensure the encoded row exposes that key (e.g., map `_fusion_source` → `packing_group`).
  - Replace ms-swift PackingDataset creation with `GroupedPackingDataset`; otherwise, keep the default path unchanged.
- Collate/trainer: carry `packed_group` through to metrics; accumulate per-dataset loss by `packed_group` while leaving the global loss untouched.

## Config
- New optional config field: `packing_group_key` (default: unset). When unset, behavior is identical to current packing.
- No new drop_last flag for now.

## Validation
- Unit tests:
  - Packs never mix groups when `packing_group_key` is set.
  - Determinism with fixed seeds.
  - Partial packs are retained (drop_last ignored by design).
- Integration smoke: multi-target fusion + packing_group_key produces per-dataset loss metrics and stable pack counts.

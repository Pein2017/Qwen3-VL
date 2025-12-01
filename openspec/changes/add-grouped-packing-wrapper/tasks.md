1. Review existing packing usage in `src/sft.py` and the fusion dataset output to confirm available metadata (e.g., `_fusion_source`).
2. Add `packing_group_key` to training config schema/CLI (default unset).
3. Implement local `GroupedPackingDataset` (and iterable variant if needed) that buckets by group_key, packs per group, retains partial packs, and records `packed_group`.
4. Wire `sft.py` to use the grouped wrapper when `packing_group_key` is provided; otherwise, keep current ms-swift packing path.
5. Ensure encoded samples expose the group key (e.g., copy `_fusion_source` to `packing_group`).
6. Add per-dataset loss logging keyed by `packed_group` (without changing global loss).
7. Tests: unit (no cross-group mixing, determinism), integration smoke (multi-target fusion + grouped packing logs per-dataset loss).
8. Docs: update `UNIFIED_FUSION_DATASET.md` / `DATA_AND_DATASETS.md` with the new knob and usage note (partial packs kept).
9. Run `openspec validate add-grouped-packing-wrapper --strict` before sharing.

- [x] Review current packing expectations in `src/sft.py`, BaseCaptionDataset, and FusionCaptionDataset (metadata fields, epoch callback).
- [x] Add config knobs: `packing_group_key` (enum/string), optional `packing_length_override`; defaults preserve existing behavior.
- [x] Implement `GroupedPackingDataset` (+ iterable variant) with group bucketing, epoch-aware bin rebuild, partial-pack retention, and `packed_group` side channel.
- [x] Expose a collator shim and trainer hook to strip metric fields before forward and log per-group loss.
- [x] Wire `sft.py` to wrap datasets (Base or Fusion) when `packing` is true; choose grouped wrapper when group key set, else ms-swift packer. Forward `set_epoch` to the packer.
- [x] Ensure encoded samples expose configurable group keys (e.g., `_fusion_domain` / `_fusion_source` / constant for single dataset) without mutating model inputs.
- [x] Tests:
  - Unit: no cross-group mixing, epoch rebuild, determinism, multimodal packed tensors intact, long-sequence single-pack.
  - Integration (using `configs/fusion/bbu_rru_lvis_coig.yaml`): 
    * tiny-target + large-source fusion run with grouping,
    * multi-target ratio change across epochs,
    * grouping disabled fallback matches ms-swift pack counts.
- [x] Docs: update `UNIFIED_FUSION_DATASET.md`, `DATA_AND_DATASETS.md`, and any README snippet to explain knobs, grouping modes, partial-pack behavior, and the above edge cases.
- [x] Run `openspec validate add-grouped-packing-wrapper --strict`.

# Design: Group-aware & epoch-aware packing (local wrapper)

## Approach
- Implement local `GroupedPackingDataset` / `GroupedIterablePackingDataset` (e.g., `src/packing/grouped_packing.py`) that mirrors ms-swift packing but adds:
  - `group_key` bucketing (none|dataset|domain|custom) to prevent cross-group mixing.
  - `packed_group` side channel attached to each packed item for metrics.
  - Epoch-aware bin rebuild via `set_epoch(epoch)` to track Fusion quotas/shuffle changes.
  - Partial packs retained (no drop_last).
- Keep rows unmodified: no mutation of `input_ids/labels/pixel_values/position_ids`; only regroup indices.
- Provide a collator shim that:
  - unwraps packed rows → calls `template.data_collator`
  - reattaches `packed_group`
  - strips metric fields before model forward.

## Integration
- In `src/sft.py`, after dataset construction (Base or Fusion):
  - If `training.packing` is true:
    - Choose packing class: grouped wrapper when `packing_group_key` set; else fall back to ms-swift packer.
    - Pass `group_key_fn` (default: constant for single dataset; fusion: `_fusion_domain` or `_fusion_source` per config).
    - Wire a Trainer callback (or small trainer mixin) to pop `packed_group`, log per-group loss, and forward clean kwargs.
  - Ensure `FusionEpochCallback` calls `set_epoch` on both the dataset and the packing wrapper.
- Config surface:
  - `packing_group_key`: string or enum (`none|dataset|domain|custom_field`), optional.
  - Optional `packing_length_override` (falls back to template.max_length).
- Compatibility:
  - Works with BaseCaptionDataset (single dataset) and FusionCaptionDataset (multi).
  - No upstream ms-swift edits; padding-free/flash attention stays intact.

## Validation
- Unit tests:
  - No cross-group mixing when grouping is enabled.
  - Epoch increment causes different bin composition matching new Fusion quotas.
  - Deterministic packing given fixed seeds.
  - Partial packs emitted (no drop_last).
  - Multimodal packed batch preserves `position_ids/text_position_ids` and vision tensors.
- Integration smoke:
  - Single-dataset run with packing=true succeeds.
  - Fusion run with grouping emits per-group loss logs and stable pack counts.

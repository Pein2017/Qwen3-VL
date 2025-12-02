# sft-training (Delta)

## ADDED Requirements

### Requirement: Opt-in grouped packing without upstream changes
The SFT pipeline SHALL provide an opt-in grouped packing mode implemented within this repository (no edits to the installed ms-swift package) that packs sequences only within the same group key.

#### Scenario: Enable grouped packing
- **WHEN** training is configured with `packing_group_key: "_fusion_source"`
- **THEN** the dataloader uses the local grouped packing wrapper instead of ms-swift’s default packer
- **AND** each packed sample contains only records sharing that group key.

### Requirement: Support single- and multi-dataset training
Grouped packing SHALL work for both single-dataset (BaseCaptionDataset) and fusion (FusionCaptionDataset) runs without changing user YAML beyond the existing `packing: true` plus optional group config.

#### Scenario: Single dataset with packing
- **WHEN** `packing: true` is set and no `packing_group_key` is provided
- **THEN** the dataset is packable with unchanged behavior vs today (no grouping), and training succeeds.

#### Scenario: Fusion dataset with grouping
- **WHEN** `packing: true` and `packing_group_key` is set to a fusion field (e.g., `_fusion_domain` or `_fusion_source`)
- **THEN** packing is applied and sequences never mix records from different groups.

### Requirement: Epoch-aware packing
The grouped packing wrapper SHALL rebuild its binning each epoch so it stays aligned with Fusion per-epoch quotas/shuffle.

#### Scenario: Epoch advances
- **WHEN** the training epoch increments (callback `set_epoch`)
- **THEN** the packed bins are recomputed using the dataset’s current schedule, reflecting any new target downsampling or source resampling.

### Requirement: Configurable grouping mode
The system SHALL expose a config knob to choose the grouping key (`none|dataset|domain|custom_field`), defaulting to current ms-swift behavior when unset.

#### Scenario: Default mode
- **WHEN** the grouping knob is absent
- **THEN** packing behaves exactly like ms-swift (no grouping, same outputs as today).

#### Scenario: Domain grouping
- **WHEN** `packing_group_key: "_fusion_domain"` is set
- **THEN** packs are constrained to a single domain (e.g., `target` vs `source`).

### Requirement: Preserve default behavior when unset
The grouped packing mode SHALL be disabled by default; when `packing_group_key` is not set, packing behavior remains identical to current ms-swift packing.

#### Scenario: Group key absent
- **WHEN** `packing_group_key` is omitted
- **THEN** the pipeline uses the existing packing path with no grouping and identical outputs to today.

### Requirement: Retain partial packs (no drop_last)
The grouped packing implementation SHALL keep partial packs (no drop_last) to avoid data loss, matching current throughput expectations.

#### Scenario: Underfilled pack
- **WHEN** the final pack for a group is underfilled
- **THEN** it is still emitted rather than dropped.

### Requirement: Propagate packed group for metrics
The grouped packing implementation SHALL attach the group identifier (e.g., `_fusion_source`) to each packed batch so that per-dataset loss metrics can be computed without changing the global loss.

#### Scenario: Per-dataset loss logging
- **WHEN** a packed batch is produced under grouped packing
- **THEN** the batch includes `packed_group`, and the trainer can log mean loss by `packed_group` while keeping overall loss intact.

### Requirement: Preserve multimodal fields and masks
Packing MUST leave per-row multimodal tensors (`pixel_values`, `image_grid_thw`, etc.) and `position_ids` intact so Qwen3-VL flash-attention and padding-free paths remain correct.

#### Scenario: Packed multimodal batch
- **WHEN** a packed batch contains vision tokens
- **THEN** the collator still emits correct `position_ids/text_position_ids` (and `cu_seq_lens` where applicable) and no additional attention mask is required.

### Requirement: Strip aux metadata before model forward
Any grouping/metric metadata added for packing SHALL be removed or ignored before calling the model forward to avoid unexpected-kwargs errors.

#### Scenario: Forward pass
- **WHEN** a packed batch reaches the model
- **THEN** only model-accepted kwargs remain; `packed_group`/aux fields are consumed by the trainer or callback layer.

### Requirement: Edge-case coverage for packing
The grouped packing path SHALL be validated against extreme and boundary conditions using the local fusion config `configs/fusion/bbu_rru_lvis_coig.yaml`.

#### Scenario: Single tiny target, large sources
- **GIVEN** only one small target split (e.g., `val_tiny` for rru) and large sources
- **WHEN** packing with grouping is enabled
- **THEN** packs are built from the tiny target without crash, sources are excluded from eval, and per-group loss logging still functions.

#### Scenario: Uneven target ratios with epoch rebuild
- **GIVEN** multi-target quotas derived from ratios that change the sampled subset each epoch
- **WHEN** advancing epoch triggers `set_epoch`
- **THEN** packed bins are recomputed and reflect the new sampled indices (no reuse of stale bins).

#### Scenario: Multimodal packed batch (images present)
- **WHEN** packing a batch that includes image tokens (Qwen3-VL vision fields)
- **THEN** `text_position_ids`/`position_ids` are preserved, flash-attention runs without attention_mask errors, and image grids align with token placeholders.

#### Scenario: Long-sequence near packing_length
- **WHEN** sequences approach `packing_length` and cannot co-pack
- **THEN** they are emitted as single-item packs without truncation or drop.

#### Scenario: Fallback with grouping disabled
- **WHEN** `packing: true` but `packing_group_key` is unset
- **THEN** behavior matches ms-swift default packing (no group leakage, identical pack counts as baseline).

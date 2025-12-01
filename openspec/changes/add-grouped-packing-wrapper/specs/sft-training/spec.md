# sft-training (Delta)

## ADDED Requirements

### Requirement: Opt-in grouped packing without upstream changes
The SFT pipeline SHALL provide an opt-in grouped packing mode implemented within this repository (no edits to the installed ms-swift package) that packs sequences only within the same group key.

#### Scenario: Enable grouped packing
- **WHEN** training is configured with `packing_group_key: "_fusion_source"`
- **THEN** the dataloader uses the local grouped packing wrapper instead of ms-swift’s default packer
- **AND** each packed sample contains only records sharing that group key.

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

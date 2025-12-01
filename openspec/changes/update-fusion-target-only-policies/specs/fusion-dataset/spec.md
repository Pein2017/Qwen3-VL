# fusion-dataset Delta (update-fusion-target-only-policies)

## ADDED Requirements
### Requirement: Target-only evaluation
The fusion loader SHALL build evaluation datasets from the target domain only; source splits are ignored even when provided, and no augmentation or object caps run during evaluation.

#### Scenario: Source eval ignored
- **WHEN** a fusion config includes a target `val_jsonl` and one or more source `val_jsonl`
- **THEN** the eval dataset is constructed solely from the target split (or errors if missing)
- **AND** source-domain samples are excluded from evaluation metrics and epoch counts
- **AND** no augmentation or object caps are applied in evaluation mode.

### Requirement: Domain-isolated packing and provenance
The fusion loader SHALL tag every sample with `_fusion_domain`, `_fusion_source`, and `_fusion_template`, and packing strategies SHALL group records so that packed sequences do not mix domains (target vs source).

#### Scenario: Domain grouping for packing
- **WHEN** packed sequences are constructed from fusion samples
- **THEN** the packer uses a grouping key derived from `_fusion_domain` (or stricter `_fusion_source`) so a single packed sequence never mixes target and source records
- **AND** mixed-domain packing attempts are rejected or re-grouped.

#### Scenario: Provenance metadata available
- **WHEN** a sample is emitted by the fusion loader (online or offline fused JSONL)
- **THEN** its metadata contains `_fusion_domain`, `_fusion_source`, and `_fusion_template` fields
- **AND** these fields are preserved through preprocessing and encoding for downstream debugging and packing.

## MODIFIED Requirements
### Requirement: Per-source augmentation and curriculum policy
The fusion loader SHALL honor per-source policies for augmentation and curriculum: targets inherit the configured augmentation/curriculum; sources remain clean with augmentation/curriculum disabled regardless of configuration.

#### Scenario: Target-only augmentation
- **WHEN** a target-domain sample is fetched during training and global augmentation is enabled
- **THEN** augmentation and optional curriculum preprocessors run on the target sample
- **AND** source-domain samples in the same epoch bypass augmentation and curriculum even if their entries request it.

### Requirement: Per-source object cap
The fusion loader SHALL support per-source object caps to control sequence length, applying caps only to source-domain samples during training; targets remain uncapped in both training and evaluation.

#### Scenario: Source train cap applied
- **WHEN** the split is `train` and a source-domain policy sets `max_objects_per_image=K`
- **THEN** the loader enforces the cap deterministically before encoding that source sample
- **AND** target-domain samples ignore any cap and keep all objects
- **AND** evaluation mode ignores caps for all domains.

## RENAMED Requirements
- Optional per-source evaluation splits → Target-only evaluation

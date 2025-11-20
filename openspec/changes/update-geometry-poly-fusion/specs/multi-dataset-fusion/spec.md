# multi-dataset-fusion Specification (delta)

## ADDED Requirements

### Requirement: Canonical detection JSONL contract
The system SHALL use a single canonical JSONL schema for all detection-style datasets (BBU and auxiliary sources).

#### Scenario: Canonical fields per record
- **WHEN** any training JSONL record is consumed by `DenseCaptionDataset`
- **THEN** it provides `images` (list of image refs), `objects` (list), `width`, and `height` fields
- **AND** each object has exactly one geometry field (`bbox_2d`, `poly`, or `line`) plus a non-empty `desc` string
- **AND** all geometry coordinates are integer pixels in the image frame before encoding.

#### Scenario: Cross-dataset compatibility
- **WHEN** a record originates from an external dataset (e.g., LVIS/COCO) via a converter
- **THEN** its geometry and fields conform to the same contract as BBU records, allowing interchangeability in `DenseCaptionDataset`.

#### Scenario: Configurable polygon fallback
- **GIVEN** a dataset whose JSONL contains only canonical geometry fields (`bbox_2d`, `poly`, `line`)
- **AND** a training/fusion config that sets `poly_fallback: bbox_2d` for that dataset
- **WHEN** the dataset is loaded
- **THEN** the loader converts each `poly` to an equivalent `bbox_2d` envelope on-the-fly (before augmentation) while leaving the source JSONL unchanged
- **AND** when `poly_fallback` is disabled (default), all `poly` geometries are preserved as-is.
### Requirement: Offline fusion builder
The system SHALL provide an offline fusion builder that mixes multiple datasets into a single training JSONL according to per-source ratios.

#### Scenario: Per-source auxiliary ratios
- **GIVEN** a fusion config that declares a target dataset `bbu` and auxiliary datasets `coco` with `ratio: 0.1` and `objects365` with `ratio: 0.05`
- **WHEN** the fusion builder runs
- **THEN** it reads the declared `train_jsonl` files, computes each auxiliary quota as `round(ratio * N_bbu)` (e.g., `10` COCO + `5` Objects365 records when `N_bbu = 100`), samples that many records **with replacement** for each source every epoch, tags each record with `metadata.dataset`, shuffles, and writes a fused `train_fused.jsonl`.

#### Scenario: Extensibility to additional sources
- **WHEN** a new auxiliary dataset is added to the fusion config with a valid `train_jsonl` path and `ratio`
- **THEN** the fusion builder can include it in the fused output without changes to `DenseCaptionDataset` or the augmentation pipeline.

### Requirement: Source-aware augmentation
Augmentation SHALL be configurable per dataset source without modifying the dataset builder.

#### Scenario: Enable augmentation only for target domain
- **GIVEN** a fused training JSONL where each record has `metadata.dataset` set
- **AND** augmentation config that lists `augment_sources: ["bbu"]`
- **WHEN** the source-aware augmentation preprocessor runs
- **THEN** it applies the configured image/geometry augmentations only to records with `metadata.dataset == "bbu"`
- **AND** passes through auxiliary records unchanged (no augmentation applied).

#### Scenario: Opt-in augmentation for auxiliary datasets
- **WHEN** `augment_sources` includes both `"bbu"` and an auxiliary dataset like `"lvis"`
- **THEN** the preprocessor applies augmentations to records from both sources, using the same pipeline, without duplicating code paths.

### Requirement: Evaluation scoped to target dataset
The default evaluation configuration SHALL use the target dataset’s validation split only.

#### Scenario: Target-only evaluation
- **GIVEN** a fusion config that declares a target dataset `bbu` with `val_jsonl`
- **WHEN** the training config is resolved for evaluation
- **THEN** `custom.val_jsonl` points to the BBU `val_jsonl`
- **AND** auxiliary datasets are not included in evaluation metrics unless explicitly configured.

### Requirement: Uniform loss across datasets
Mixed batches SHALL optimize the same teacher-forcing cross-entropy objective for every sample regardless of dataset provenance.

#### Scenario: Standard CE over fused batches
- **WHEN** the training loop builds a batch that contains both BBU and auxiliary records
- **THEN** it runs the regular dense-caption CE loss over all targets without applying dataset-specific scaling factors
- **AND** any per-source loss numbers (e.g., `loss_bbu`, `loss_aux`) are logged for observability only and do not alter gradient weights.

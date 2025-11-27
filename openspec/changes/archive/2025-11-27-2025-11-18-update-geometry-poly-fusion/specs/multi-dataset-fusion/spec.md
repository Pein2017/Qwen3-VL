# multi-dataset-fusion Specification (delta)

## ADDED Requirements

### Requirement: Canonical detection JSONL contract
The system SHALL use a single canonical JSONL schema for all detection-style datasets (target and source domains).

#### Scenario: Canonical fields per record
- **WHEN** any training JSONL record is consumed by `DenseCaptionDataset`
- **THEN** it provides `images` (list of image refs), `objects` (list), `width`, and `height` fields
- **AND** each object has exactly one geometry field (`bbox_2d`, `poly`, or `line`) plus a non-empty `desc` string
- **AND** all geometry coordinates are integer pixels in the image frame before encoding.

#### Scenario: Cross-dataset compatibility
- **WHEN** a record originates from an external dataset (e.g., LVIS/COCO) via a converter
- **THEN** its geometry and fields conform to the same contract as BBU records, allowing interchangeability in `DenseCaptionDataset`.

#### Scenario: Offline polygon simplification by vertex count
- **GIVEN** a source dataset conversion step configured with `poly_max_points: N`
- **WHEN** an object has a `poly` with more than `N` vertices (or `poly_points > N`)
- **THEN** the converter downgrades only that object to `bbox_2d` while preserving polygons at or below the threshold in the emitted JSONL
- **AND** loaders consume the JSONL as-is without additional polygon fallback.
- **AND** `public_data/scripts/convert_rescale_source.sh` (or any wrapper around `convert_lvis.py`) can be invoked with `--poly-max-points 12` to produce `public_data/lvis/rescale_32_768_poly_max_12/train.jsonl`, which fusion configs can reference directly.

#### Scenario: Image paths resolve relative to JSONL location
- **GIVEN** a dataset JSONL whose `images` entries are relative paths (e.g., `../raw/images/...`)
- **WHEN** the loader instantiates records for training
- **THEN** it resolves each image path against the JSONL’s parent directory and stores the absolute filesystem path in-memory
- **AND** training never depends on process working directories or manually created symlinks to locate images.

### Requirement: Domain-aware dataset wrappers
The fusion stack SHALL load every dataset (BBU, RRU, COCO, Objects365, etc.) through a domain-specific wrapper that emits normalized records and exposes whether augmentation is allowed.

#### Scenario: Target/source wrappers feed a unified factory
- **WHEN** a config references `target: {dataset: bbu}` or `sources: [{dataset: coco, ratio: 0.1}, ...]`
- **THEN** the fusion factory instantiates the corresponding wrapper classes, each of which loads its JSONL/images, enforces the canonical schema, and declares `domain ∈ {"target","source"}` plus `supports_augmentation` and `template_id`
- **AND** the factory hands their normalized records to `DenseCaptionDataset` without requiring per-record metadata shims.

#### Scenario: Wrapper extensibility
- **WHEN** a new dataset (e.g., `rru` or `flickr3k`) gains a wrapper that subclasses the shared base contract
- **THEN** it can be plugged into the fusion config with no changes to the fusion builder or augmentation logic.

### Requirement: Offline fusion builder (static ratios)
The system SHALL provide an offline (and equivalent online) fusion builder that mixes datasets according to fixed per-source ratios computed once per epoch.

#### Scenario: Per-source auxiliary ratios
- **GIVEN** a fusion config that declares a target dataset `bbu` and auxiliary datasets `coco` with `ratio: 0.1` and `objects365` with `ratio: 0.05`
- **WHEN** the fusion builder runs for an epoch
- **THEN** it computes each auxiliary quota as `round(ratio * N_target)` (e.g., `10` COCO + `5` Objects365 records when `N_target = 100`), samples that many records **with replacement** once for the epoch, and interleaves them with the `N_target` records without requiring any runtime schedule updates mid-epoch.

#### Scenario: Extensibility to additional sources
- **WHEN** a new auxiliary dataset is added to the fusion config with a valid `train_jsonl` path and `ratio`
- **THEN** the fusion builder can include it in the fused output without changes to `DenseCaptionDataset` or the augmentation pipeline.

### Requirement: Domain-scoped augmentation policy
Augmentation SHALL be enabled or disabled per domain via dataset-wrapper configuration, not per-record metadata.

#### Scenario: Source domain skips augmentation by default
- **GIVEN** `source` wrappers whose `supports_augmentation` flag is `false`
- **WHEN** the unified dataset constructs `DenseCaptionDataset` instances
- **THEN** it omits the augmentation preprocessor for those wrappers so COCO/Objects365-style data flows through unchanged regardless of global augmentation settings.

#### Scenario: Opt-in augmentation for select sources
- **WHEN** a source wrapper explicitly sets `supports_augmentation=true`
- **THEN** the same augmentation pipeline (and optional curriculum) can be attached to that dataset while still avoiding code duplication across wrappers.

### Requirement: Evaluation scoped to target dataset with optional source splits
The default evaluation configuration SHALL use the target dataset’s validation split, with optional source `val_jsonl` opt-in when explicitly provided.

#### Scenario: Target-only evaluation
- **GIVEN** a fusion config that declares a target dataset `bbu` with `val_jsonl`
- **WHEN** the training config is resolved for evaluation without source `val_jsonl`
- **THEN** `custom.val_jsonl` points to the BBU `val_jsonl`
- **AND** auxiliary datasets are not included in evaluation metrics.

#### Scenario: Source eval provided
- **GIVEN** a fusion config whose source dataset provides its own `val_jsonl`
- **WHEN** evaluation is built with source eval enabled
- **THEN** the loader includes that source split (no shuffling) alongside the target eval set, assuming the source split was prepared offline (no train/val splitting inside the loader).

### Requirement: Uniform loss across datasets
Mixed batches SHALL optimize the same teacher-forcing cross-entropy objective for every sample regardless of dataset provenance.

#### Scenario: Standard CE over fused batches
- **WHEN** the training loop builds a batch that contains both target and source records
- **THEN** it runs the regular dense-caption CE loss over all targets without applying dataset-specific scaling factors or additional benchmarking passes

### Requirement: Source-domain prompt constraints
Source-domain datasets SHALL use an auxiliary prompt that emphasizes concise English class names instead of QC-specific descriptions.

#### Scenario: Aux prompt emits simple class labels
- **WHEN** a wrapper declares `template: aux_dense`
- **THEN** the bound prompts instruct the model to output norm1000 geometry with short English class names (one or two words) and no completeness or quality commentary
- **AND** LVIS/COCO style categories remain readable without leaking BBU-specific attributes into auxiliary supervision.

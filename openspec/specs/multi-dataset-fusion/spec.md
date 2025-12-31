# multi-dataset-fusion Specification

## Purpose
Capture the wrapper registry and geometry expectations for combining multiple datasets (target + sources) with fixed ratios, polygon support, and augmentation eligibility.
## Requirements
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
The offline fusion builder SHALL mix datasets according to fixed per-source ratios computed once per epoch/configuration, honoring per-source policies including optional `sample_without_replacement`; when enabled and the quota is within the source pool, the builder SHALL draw unique samples (no duplicates) using a deterministic shuffle, and when the quota exceeds the pool it SHALL fall back to deterministic with-replacement sampling and surface the fallback in its logs/telemetry. Targets continue to follow the existing ratio-based sampling rules (with replacement when ratios upsample).

#### Scenario: Offline source without-replacement within pool size
- **WHEN** a fusion config enables `sample_without_replacement: true` for a source and its offline quota is less than or equal to the pool size
- **THEN** the offline builder writes each sampled record at most once in the fused JSONL for that epoch/config, using a deterministic shuffle before taking the quota.

#### Scenario: Offline source without-replacement quota exceeds pool
- **WHEN** the same flag is set but the computed quota exceeds the source pool
- **THEN** the builder reverts to deterministic with-replacement sampling for that source while still emitting logs/telemetry that the fallback occurred.

#### Scenario: Default replacement retained (offline)
- **WHEN** a source does not enable `sample_without_replacement`
- **THEN** the offline builder samples that source with replacement (as before) while respecting per-source ratios and deterministic seeding.

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

### Requirement: Irrelevant summary target stream
The fusion stack SHALL support mixing a small "irrelevant image" dataset as an additional **target** stream in summary mode, using the existing summary template.

#### Scenario: Irrelevant target is referenced in fusion config
- **WHEN** a fusion config includes a target entry:
  - `{name: irrelevant_summary, dataset: bbu, template: bbu_summary, mode: summary, train_jsonl: data/irrelevant_summary/train.jsonl, val_jsonl: data/irrelevant_summary/train.jsonl, ratio: 1}`
  - and explicitly sets `{augmentation_enabled: false, curriculum_enabled: false}`
- **THEN** the fusion loader includes it as a target dataset in `mode: summary` using the `bbu_summary` prompts/template
- **AND** its per-epoch target quota is computed from its own pool size: `quota = round(len(pool) * ratio)` (so `ratio: 1` yields each record once per epoch)
- **AND** augmentation and curriculum remain disabled for this entry regardless of global settings.

### Requirement: Irrelevant JSONL records remain canonical
The irrelevant summary JSONL records SHALL conform to the canonical detection JSONL contract even though summary-mode encoding ignores `objects`.

#### Scenario: Dummy full-frame bbox keeps contract compatibility
- **WHEN** a record contains exactly one image, a dummy full-frame bbox object, and summary text, e.g.:
  - `images: ["images/0001.jpeg"]`
  - `width: W`, `height: H`
  - `objects: [{"bbox_2d": [0, 0, W, H], "desc": "irrelevant"}]`
  - `summary: "无关图片"`
- **THEN** it passes the canonical JSONL validator
- **AND** it is eligible for fusion sampling and summary-mode template encoding, where the assistant target is the summary string.

### Requirement: Helper for irrelevant JSONL generation
The system SHALL provide a helper that builds the irrelevant summary JSONL from a folder of JPEGs with a 1:1 image-to-record mapping.

#### Scenario: Operator generates irrelevant JSONL
- **WHEN** an operator runs the helper against `data/irrelevant_summary/images/*.jpeg`
- **THEN** it emits `data/irrelevant_summary/train.jsonl` where each line references exactly one image (relative path), sets `summary` to `无关图片`, fills `width/height` from EXIF-aware image dimensions, and emits a single dummy full-frame bbox object with a non-empty `desc`
- **AND** output ordering is deterministic (sorted by path)
- **AND** unreadable or missing images are reported and skipped without terminating the run.

### Requirement: Text-only auxiliary sources
The fusion pipeline SHALL accept text-only JSONL datasets (no images/objects) as source domains without breaking detection targets.

#### Scenario: Chat source with per-epoch ratio
- **GIVEN** a fusion config whose sources include `dataset: chat` with a valid `train_jsonl` containing `messages` turns only
- **WHEN** FusionCaptionDataset builds the epoch schedule
- **THEN** the chat source participates using its declared `ratio` (sampled with replacement) even though it has no images or geometry fields.

#### Scenario: Prompt selection for chat sources
- **GIVEN** a chat source wrapper that declares `template: chatml` (or equivalent) and provides chat-style prompts
- **WHEN** a chat record is encoded
- **THEN** the loader applies the chat template’s system/user prompts (not the detection prompts) and does **not** inject image placeholders or geometry instructions.

#### Scenario: Pass-through pre-authored conversations
- **WHEN** a record in any source contains a `messages` array and no `images`/`objects`
- **THEN** the builder reuses those messages verbatim for encoding, keeping metadata intact, instead of synthesizing detection-style user/assistant payloads.

### Requirement: Per-dataset summary label grouping override
The system SHALL allow fusion dataset entries to override summary label grouping behavior on a per-dataset basis.

#### Scenario: Fusion config overrides grouping for summary datasets
- **GIVEN** a fusion config entry in summary mode that sets `summary_label_grouping` to `true` or `false`
- **WHEN** the fused dataset is built for training
- **THEN** the summary label normalizer is applied (or skipped) for that dataset according to the override
- **AND** datasets without an override continue to follow `custom.summary_label_grouping`.

### Requirement: Provenance metadata in fused outputs
The offline fusion builder SHALL emit fused JSONL records annotated with `_fusion_domain`, `_fusion_source`, and `_fusion_template` fields consistent with the online fusion loader so downstream packers and debuggers can distinguish target vs source samples without extra configuration.

#### Scenario: Fused JSONL carries provenance
- **WHEN** the offline fusion builder materializes a fused train JSONL from a fusion config with target and auxiliary sources
- **THEN** every written record contains `_fusion_domain`, `_fusion_source`, and `_fusion_template` metadata matching its originating dataset
- **AND** consumers can group or filter fused records by these fields without inferring provenance from file paths or dataset order.


# detection-preprocessor Specification

## Purpose
Define the shared offline smart-resize preprocessor and converter contract for detection datasets (BBU + public sources), including pixel budgeting, grid alignment, geometry rescaling, and path normalization.
## Requirements
### Requirement: Canonical converter contract (all datasets)
All target and source detection converters SHALL emit the canonical BBU-style JSONL and remain resize-free. BBU/RRU converters SHALL follow the key=value description contract and JSON-string summary contract.

#### Scenario: Schema and single-geometry invariant
- **WHEN** any dataset converter (BBU, RRU, LVIS/COCO/Objects365, etc.) writes JSONL
- **THEN** each record provides `images`, `objects`, `width`, `height` where every object has exactly one geometry (`bbox_2d` or `poly`+`poly_points` or `line`) plus a non-empty `desc`
- **AND** image paths are relative to the output JSONL by default (no symlink/CWD assumptions)
- **AND** converters do not apply resizing or augmentation; they only “cook” annotations to canonical JSONL
- **AND** for BBU/RRU domains, desc uses comma-separated key=value pairs with no spaces and no slash-delimited levels

#### Scenario: Converter extensibility
- **WHEN** a new source dataset is added
- **THEN** only the converter implementation is customized; it reuses the shared smart-resize offline path (online guard is legacy-only)

#### Scenario: BBU vs. RRU desc contract
- **WHEN** a BBU converter emits object descriptions
- **THEN** desc includes `备注` when present and MUST NOT include `组`
- **AND** any review-state placeholder is forbidden
- **WHEN** an RRU converter emits object descriptions
- **THEN** desc MAY include `组=<id>` and MUST NOT include `备注`

#### Scenario: Value normalization and OCR handling
- **WHEN** a converter writes desc values
- **THEN** values MUST remove all whitespace (including fullwidth spaces)
- **AND** OCR/备注 free text preserves punctuation (including `,|=`); no symbol replacement is applied
- **AND** OCR content preserves `-` and `/` characters (no replacement)
- **AND** stray comma tokens without `key=` are folded into `备注`
- **AND** `这里已经帮助修改,请注意参考学习` is stripped from `备注` when present
- **AND** station distance values MUST normalize to an integer token (strip non-digits) and be emitted as `站点距离=<int>`

#### Scenario: Deterministic key ordering
- **WHEN** a converter emits desc for any object
- **THEN** key=value pairs follow a deterministic per-category order
- **AND** `类别` is always first
- **AND** `备注` (BBU) or `组` (RRU) is always last when present

#### Scenario: Group encoding (RRU)
- **WHEN** an RRU object has group membership
- **THEN** the desc encodes groups as a single `组=` key
- **AND** multiple groups are joined with `|` in ascending numeric order

#### Scenario: Drop occlusion judgments
- **WHEN** a converter ingests occlusion/遮挡 answers (e.g., `有遮挡`, `无遮挡`, `挡风板有遮挡`)
- **THEN** those values are not emitted in desc key=value pairs
- **AND** occlusion values are not counted in summary JSON

#### Scenario: Conflict resolution (negative precedence)
- **WHEN** a single object includes both a positive and a negative compliance choice (e.g., `符合` and `不符合/露铜`)
- **THEN** the converter selects the negative branch and records the specific issue
- **AND** multiple negative issues are joined with `|`

#### Scenario: JSON-string summaries
- **WHEN** a BBU or RRU converter writes summaries
- **THEN** the summary field is a JSON string describing per-category counts
- **AND** the summary does not use ×N aggregation
- **AND** the summary only reports observed values (no missing counts)
- **AND** BBU summaries include a global `备注` list while RRU summaries omit `备注` and MAY include group statistics
- **AND** irrelevant-image summary streams MAY use the literal string `无关图片` instead of JSON

#### Scenario: Summary JSON schema keys
- **WHEN** a BBU or RRU converter writes summaries
- **THEN** the JSON string includes `统计`
- **AND** the JSON string MUST NOT include `dataset`
- **AND** `统计` is a list of per-category objects each containing `类别` plus any observed attribute counts
- **AND** BBU summaries include a `备注` list of strings (may be empty)
- **AND** RRU summaries MAY include `分组统计` (group id → count) and per-category `组` counts
- **AND** summaries MUST NOT include any legacy total-count field (e.g., `objects_total`)
- **AND** converters MUST fail-fast on summary anomalies rather than embedding an `异常` object
- **AND** the JSON string is single-line and uses standard separators (`, ` and `: `)

### Requirement: Unified smart-resize preprocessor
The system SHALL provide a shared detection preprocessor that enforces a pixel budget and grid alignment before training or inference.

#### Scenario: Pixel budget and grid snapping
- **GIVEN** `max_pixels` and `image_factor` (e.g., 786432 and 32)
- **WHEN** the preprocessor ingests an image+record pair
- **THEN** it ensures the resized dimensions satisfy `width * height <= max_pixels` by proportional scaling if needed
- **AND** rounds dimensions to the nearest multiple of `image_factor` while preserving aspect ratio.

#### Scenario: Geometry rescaling and metadata update
- **WHEN** smart resize is applied
- **THEN** all geometries (`bbox_2d`, `poly`, `line`) are rescaled to the resized image frame
- **AND** record `width`/`height` fields are updated accordingly
- **AND** coordinates are clamped to valid pixel bounds.

#### Scenario: Idempotent execution
- **GIVEN** an image already within `max_pixels` and aligned to `image_factor`
- **WHEN** passed through the preprocessor
- **THEN** it returns the original image/geometry unchanged (no-op), making it safe to run in offline converters or online guards.

### Requirement: Path normalization
The preprocessor SHALL normalize image paths relative to the JSONL location and resolve them to absolute paths at load time.

#### Scenario: Relative paths resolved deterministically
- **GIVEN** records whose `images` entries are relative (e.g., `../raw/images/...`)
- **WHEN** the preprocessor loads or writes records
- **THEN** it resolves those paths against the JSONL directory and stores absolute paths in-memory, avoiding dependence on process CWD or symlinks.

### Requirement: Configurable invocation (offline-only for fusion)
The smart-resize preprocessor SHALL be invocable during conversion (offline). Unified fusion does not run an online guard; runtime resize hooks must reject enablement attempts.

#### Scenario: Offline conversion with smart resize
- **WHEN** a converter (e.g., LVIS/COCO) is run with `--smart-resize` and `--max_pixels/--image_factor`
- **THEN** it writes resized images to the configured output root, emits JSONL with updated dimensions/geometry, and keeps image paths relative to the resized folder.

#### Scenario: Online guard disallowed in unified fusion
- **WHEN** an online smart-resize flag is passed to the unified fusion loader
- **THEN** the loader rejects the configuration and instructs the user to run offline resize instead.

### Requirement: Single implementation across domains (offline-first)
All resize flows (BBU data_conversion, public_data converters) SHALL delegate to the shared smart-resize implementation. Online fusion guards are not supported; offline resize is the single path.

#### Scenario: data_conversion delegation
- **WHEN** the BBU resize CLI runs (e.g., `data_conversion/resize_dataset.py`)
- **THEN** it calls the shared smart-resize helpers for image size computation and geometry scaling, guaranteeing identical behaviour with source-domain converters.

#### Scenario: Shared offline wrapper
- **WHEN** any dataset invokes “convert + smart-resize” via CLI
- **THEN** it uses the same flags (`--smart-resize`, `--max_pixels`, `--image_factor`, optional `--min_pixels`) and produces co-located resized images + JSONL without dataset-specific resize code paths.

### Requirement: Generic converter runner
The system SHALL provide a generic “convert then optional smart-resize” runner so new datasets extend only the converter.

#### Scenario: Pluggable converter + shared flags
- **WHEN** a converter module/name is passed to the runner along with standard resize flags
- **THEN** the runner executes the converter to canonical JSONL, optionally applies shared smart-resize, and writes output JSONL + images under the specified root with paths relative to that JSONL.

### Requirement: Telemetry and safety logging
The preprocessor SHALL log scale factors and warn on aggressive downscales.

#### Scenario: Warning on large shrink
- **GIVEN** an image that requires >2× downscale to satisfy `max_pixels`
- **WHEN** processed
- **THEN** it logs original vs. final dimensions and the scale factor, with a remediation hint to adjust `max_pixels` or upstream image sizes.

### Requirement: Converters SHALL reject review placeholders instead of emitting them
Converters SHALL not introduce or propagate review-state placeholders when handling detection data; uncertain remarks stay as free text, and any explicit review marker is treated as invalid input.
#### Scenario: Review markers are treated as invalid input
- **WHEN** a detection converter or summary builder encounters any review-state wording (legacy third-state markers) in source annotations or intermediate desc/summary strings
- **THEN** it MUST raise a validation error and stop the conversion
- **AND** it MUST NOT rewrite desc to include a review placeholder; remarks remain free-text but keep original positive/negative values
- **AND** downstream JSONL/output summaries MUST stay binary/observational only, with no review marker keys or values


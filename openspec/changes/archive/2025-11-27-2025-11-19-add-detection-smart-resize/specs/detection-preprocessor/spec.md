# detection-preprocessor Specification (delta)

## ADDED Requirements

### Requirement: Canonical converter contract (all datasets)
All target and source detection converters SHALL emit the canonical BBU-style JSONL and remain resize-free.

#### Scenario: Schema and single-geometry invariant
- **WHEN** any dataset converter (BBU, LVIS/COCO/Objects365, etc.) writes JSONL
- **THEN** each record provides `images`, `objects`, `width`, `height` where every object has exactly one geometry (`bbox_2d` or `poly`+`poly_points` or `line`) plus non-empty `desc`
- **AND** image paths are relative to the output JSONL by default (no symlink/CWD assumptions)
- **AND** converters do not apply resizing or augmentation; they only “cook” annotations to canonical JSONL.

#### Scenario: Converter extensibility
- **WHEN** a new source dataset is added
- **THEN** only the converter implementation is customized; it reuses the shared smart-resize offline path (online guard is legacy-only).

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

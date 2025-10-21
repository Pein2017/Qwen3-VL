## ADDED Requirements
### Requirement: Image-level Augmentation Plugin System
The system SHALL provide a plugin-based mechanism to apply image-level augmentations during preprocessing without modifying dataset builders.

#### Scenario: Enable built-in augmentations via config
- **WHEN** the YAML config lists `augmentations: [hflip, rotate, scale, color_jitter]`
- **THEN** those operations are applied in order with deterministic behavior under a fixed seed

#### Scenario: Disable augmentations
- **WHEN** the YAML config sets `augmentations: []` or feature flag off
- **THEN** no augmentation is applied and original images/geometries are preserved

#### Scenario: Third-party augmentation registration
- **WHEN** a user registers a custom op via `register('my_op')(callable)` and references it in config
- **THEN** the callable is invoked within the composition and updates images/geometries accordingly

#### Scenario: Geometry consistency
- **WHEN** affine-like transforms are applied (flip/rotate/scale)
- **THEN** geometry coordinates are updated via shared affine utilities and remain within image bounds (clipped)

#### Scenario: Fail-fast validation
- **WHEN** inputs are invalid (mismatched image counts, empty list, corrupt image)
- **THEN** preprocessing raises a `ValueError` with actionable message and does not continue silently

#### Scenario: RNG injection for determinism
- **WHEN** a fixed seed RNG is passed to the preprocessor
- **THEN** the same sequence of augmentations and parameters is produced across runs

### Requirement: Geometry correctness after augmentation
All geometries (bbox_2d, quad, line) SHALL remain correct and valid in the augmented image coordinate frame.

#### Scenario: bbox remains axis-aligned and within bounds
- **WHEN** rotate/flip/scale are applied
- **THEN** bbox is recomputed as the AABB of transformed corners, quantized to int, and clamped to `[0,W-1]×[0,H-1]`

#### Scenario: quad preserves point order and stays in-bounds
- **WHEN** affine transforms are applied
- **THEN** quad points are transformed independently, quantized, and clamped; the geometry type remains `quad`

#### Scenario: line clipping and deduplication
- **WHEN** any affine transform pushes polyline segments outside the image
- **THEN** segments are clipped against the image rectangle (or points clamped when clipping disabled), and consecutive duplicates are removed; at least 2 distinct points remain

#### Scenario: single-geometry-field invariant
- **WHEN** updating per-object geometry after augmentation
- **THEN** exactly one geometry field is present per object; no implicit conversion between types occurs

#### Scenario: multi-image record consistency
- **WHEN** a record has multiple images
- **THEN** a single affine matrix is applied to all images and geometries to preserve cross-image alignment

#### Scenario: fail-fast invalid input
- **WHEN** an object contains invalid geometry (missing/extra fields) or images have mismatched sizes
- **THEN** preprocessing raises `ValueError` with actionable message and halts (or skips augmentation when strict=false)

#### Scenario: non-degenerate outputs
- **WHEN** transformed geometries collapse below configured thresholds
- **THEN** the record uses original images/geometries (strict=false) or raises (strict=true)

#### Scenario: Keep absolute pixel coordinates
- **WHEN** any augmentation is applied
- **THEN** all geometry coordinates remain integer pixels in the image frame `[0, W-1]×[0, H-1]` (no normalization to 0–1000 here)

#### Scenario: Delegated normalization
- **WHEN** training/inference encoding occurs via ms‑swift/HF processor
- **THEN** normalization to `norm1000` is handled by the processor; augmentation outputs stay in pixel space

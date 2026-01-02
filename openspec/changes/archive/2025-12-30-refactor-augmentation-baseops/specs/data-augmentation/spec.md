# data-augmentation Specification Delta — refactor-augmentation-baseops

## ADDED Requirements

### Requirement: Base operator classes for augmentation
The augmentation system SHALL provide three base operator classes—AffineOp, ColorOp, PatchOp—that encapsulate shared pipelines for matrix sampling, color transforms, and patch/object transforms while remaining compatible with the existing registry and YAML schema.

#### Scenario: Register a new affine op via base class
- **WHEN** a developer implements `class Tilt(AffineOp)` with `sample_matrix(...)` and registers it
- **THEN** Compose accumulates its matrix with other affine ops and applies a single warp to images and geometries, preserving the current axis-aligned vs general classification rules.

#### Scenario: Deferred color ops via ColorOp
- **WHEN** a ColorOp is listed in YAML after affine ops
- **THEN** Compose flushes affines first, then applies the ColorOp to every image without touching geometries, preserving record alignment.

#### Scenario: PatchOp lifecycle
- **WHEN** a PatchOp is executed
- **THEN** it follows the lifecycle (select patches → transform patch → place patch) and sets `allows_geometry_drops` to signal whether object counts may change; Compose propagates its telemetry (kept_indices, coverages, skip_reason) unchanged.

### Requirement: Typed curriculum exposure for ops
Each registered op SHALL expose a typed curriculum parameter map so the augmentation curriculum scheduler can adjust probabilities and numeric ranges without bespoke per-op logic and with fail-fast validation.

- The map MUST contain numeric scalars or 2-element numeric ranges for tunable fields (e.g., `prob`, `scale`, `gamma`), derived from the op’s current configuration.
- Only numeric values (ints/floats) are allowed; booleans, strings, and nested structures MUST be rejected.
- The scheduler MUST:
  - Read the base map for each op at initialization.
  - Reject curriculum overrides that reference unknown ops or parameters.
  - Reject overrides whose numeric dimensionality does not match the base value (scalar vs 2-range).
  - Enforce `[0.0, 1.0]` bounds for probability parameters (`prob` or `*_prob`), raising an error when violated.

#### Scenario: Curriculum scales patch prob
- **WHEN** curriculum requests `small_object_zoom_paste.prob = 0.25` at 50% progress
- **THEN** the scheduler updates the PatchOp instance via the typed param map and Compose applies the new probability within that step.

#### Scenario: Invalid curriculum override fails fast
- **WHEN** curriculum refers to `random_crop.scale = [0.5, 1.5, 2.0]` (wrong dimension) or `rotate.unknown_param = 1.0`
- **THEN** the scheduler raises a `ValueError` during configuration and training does not start.

### Requirement: Equivalence of refactored augmentation behavior
For all existing augmentation ops and YAML configs, the refactored pipeline SHALL be semantically equivalent to the pre-refactor behavior under a fixed RNG seed.

- Reference behavior is defined by golden fixtures produced before the refactor (images, geometries, telemetry) for a representative JSONL slice and fixed seeds.
- Under the same inputs and seeds, the refactored pipeline MUST produce outputs that match the golden fixtures up to the following tolerances:
  - Image pixels: identical bytes or per-pixel normalized float difference ≤ `1e-4`.
  - Geometry coordinates: absolute difference ≤ `1e-4` per coordinate.
  - Telemetry numeric values: absolute difference ≤ `1e-6`; string fields (e.g., skip reasons) MUST match exactly.
- Geometry ordering MUST respect PatchOp ordering rules (crop retains original order, duplicates appended in selection order).
- New ops MAY be added, but MUST NOT change the behavior of existing configs.

#### Scenario: Golden fixture vs refactored pipeline equivalence
- **WHEN** a fixed JSONL sample and seed are run through the refactored augmentation pipeline
- **THEN** the resulting images, geometries, and telemetry match the pre-refactor golden fixtures within the specified numeric tolerances and ordering rules.

## MODIFIED Requirements

### Requirement: Image-level Augmentation Plugin System
The plugin system SHALL treat AffineOp, ColorOp, and PatchOp uniformly through the registry, allowing new ops to be added without duplicating flush/telemetry logic.

#### Scenario: Mixed op pipeline without boilerplate
- **WHEN** YAML lists `[rotate, color_jitter, small_object_zoom_paste]`
- **THEN** Compose infers kinds from the base classes, accumulates affines, defers color, flushes before PatchOp, and no operator-specific glue code is required for orchestration.

### Requirement: PatchOp determinism, ordering, and telemetry invariants
PatchOps SHALL provide deterministic behavior under a fixed RNG seed, preserve a well-defined object ordering, and expose consistent telemetry for crops and copy/paste operations.

- Given identical inputs and RNG seed, PatchOps MUST produce identical outputs (images, geometries, and telemetry).
- Patch selection and placement MUST use deterministic ordering (no dependence on hash/dict iteration order).
- Crop-style PatchOps MUST:
  - Report `last_kept_indices` as indices into the original geometry list for retained objects.
  - Report a coverage value per kept index in `last_object_coverages`.
  - Populate `last_crop_skip_reason` and `last_skip_counters` when the op bails out and returns the original images/geometries.
  - Preserve the relative order of retained original objects in the output geometry list.
- Copy/paste PatchOps MUST:
  - Preserve all original objects and append duplicate geometries; they MUST NOT drop objects unless explicitly documented as a crop.
  - Append duplicates after all originals in deterministic selection order.
  - Enforce configured overlap/IoU rules when placing patches, skipping placements that would violate them.
  - Leave crop telemetry fields (`last_kept_indices`, `last_object_coverages`, `last_crop_skip_reason`) unset/empty so that crop telemetry continues to reflect only crop-style operations.

#### Scenario: Deterministic crop PatchOp
- **WHEN** `random_crop` is applied twice to the same record with the same RNG seed
- **THEN** the cropped images, retained object set, `last_kept_indices`, and `last_object_coverages` are identical across runs.

#### Scenario: Deterministic copy/paste PatchOp
- **WHEN** `small_object_zoom_paste` is applied twice to the same record with the same RNG seed
- **THEN** the same source objects are selected, the same number of duplicates are added at the same locations, original geometries remain present in both runs, duplicates appear after originals, and crop telemetry fields remain unset/empty.

### Requirement: TL→BR object ordering after augmentation
For dense-caption training, the system SHALL restore **top-to-bottom, left-to-right (TL→BR)** ordering of objects after augmentation and before building conversations, consistent with the prompt and data conversion pipeline.

- Sorting keys:
  - Primary: Y coordinate ascending (objects higher in the image come first).
  - Secondary: X coordinate ascending (objects further left come first).
- Reference point per geometry type:
  - `bbox_2d`: top-left corner `(x1, y1)`.
  - `poly`: first vertex `(x1, y1)` after canonicalization.
  - `line`: leftmost endpoint (smallest X; if tie, smallest Y).
- The same TL→BR sort rules MUST be used in both data conversion and training-time builders so that object enumeration (`object_1`, `object_2`, …) always matches the natural reading order described in the prompt.

#### Scenario: TL→BR ordering after rotation and copy-paste
- **WHEN** a record is augmented with rotation and small-object copy-paste, changing object positions
- **AND** the dense-caption builder prepares `object_{n}` entries
- **THEN** the builder re-sorts objects by TL→BR using the defined reference points before assigning indices, so the final enumeration order matches the prompt’s “top-to-bottom, then left-to-right” contract.

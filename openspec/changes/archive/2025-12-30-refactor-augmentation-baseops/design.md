# Design notes: augmentation base operators

## Intent
Unify augmentation ops under three base classes to reduce duplicated logic and make “object-centric” transforms easier to add and schedule.

## Proposed base types
- **AffineOp**
  - Responsibility: sample an affine matrix `M` (or `None` to skip) given `(width, height, rng)`.
  - Compose behavior: accumulates `M_total`, flushes once to images + geometries; keeps axis-aligned vs general classification semantics from existing spec.
  - Examples: hflip, vflip, rotate, scale, resize_by_scale.

- **ColorOp**
  - Responsibility: per-image pixel transforms without touching geometry.
  - Compose behavior: deferred until after affine flush; applies to every image in the record.
  - Examples: color_jitter, gamma, hsv, clahe, auto_contrast, sharpness.

- **PatchOp**
  - Responsibility: patch/object selection, transform, placement, and geometry update (including optional drops or duplicates).
  - Lifecycle hooks:
    1) `select_patches(images, geoms, rng)` → iterable of patches (bbox + object indices).
    2) `transform_patch(patch, rng)` → patch image + geom transform (affine, scale, translate).
    3) `place_patch(patch, images, geoms, rng)` → updated images + geoms, with IoU/coverage guards.
  - `allows_geometry_drops` signals whether object counts may change.
  - Examples: random_crop (filters/drops), small_object_zoom_paste (duplicates), future multi-object / line paste.

## Compose integration
- Compose inspects `kind` (`affine|color|barrier`) and base class to:
  - Accumulate affines.
  - Defer color ops.
  - Flush before/after PatchOps and propagate telemetry (kept_indices, coverage, skip_reason).
- Telemetry remains backward compatible.

## Curriculum surface
- Curriculum updates only `prob` and numeric parameters declared by each op; base classes expose a typed param map so schedules don’t need per-op bespoke logic.
- Percent/step scaling rules remain as today; refactor only changes how params are exposed.

### Typed curriculum parameter map schema
- Each op instance exposes a `curriculum_params` map:
  - Keys: parameter names (e.g., `prob`, `scale`, `gamma`).
  - Values: numeric scalars or 2-element numeric ranges, represented as a `NumericParam`-like tuple of floats.
- Only numeric scalars and 2-element numeric sequences are included; booleans, strings, and nested structures are rejected.
- The curriculum scheduler:
  - Reads the base `curriculum_params` for each op at initialization.
  - Validates overrides so that:
    - The referenced op and parameter exist in the base map.
    - The override is numeric and matches the base dimension (scalar vs 2-range).
    - Probability parameters (`prob` or `*_prob`) stay within `[0.0, 1.0]`.
  - Raises a `ValueError` on any mismatch or invalid value (fail-fast) instead of silently clamping.

## Backward compatibility
- YAML schema stays the same (op names, params).
- For existing ops and configs, the refactored pipeline must be semantically equivalent: given the same RNG seed and inputs, images and geometries after augmentation must match previous behavior up to allowable tolerances (floating-point rounding and non-observable ordering differences).
- Existing validation/error messages MUST remain materially the same (wording may be cleaned up but semantics and failure conditions are preserved).
- New ops (e.g., multi-object or line PatchOps) may be added without affecting the behavior of existing configs.

## Equivalence test oracle
- The “legacy pipeline” reference is defined by a set of golden fixtures captured before the refactor:
  - A small but representative JSONL slice (covering bbox/poly/line, crop, zoom-paste, resize, and curriculum cases).
  - For each fixture and fixed RNG seed, we store:
    - Augmented image bytes (PNG).
    - Augmented geometries list (as JSON).
    - Augmentation telemetry snapshot (kept indices, coverages, skip reasons, skip counts).
- Equivalence tests:
  - Re-run the current (refactored) pipeline on the same JSONL + seeds.
  - Compare current outputs against the golden fixtures:
    - Image bytes: identical or pixel-wise float difference ≤ 1e-4 after converting to normalized float tensors.
    - Geometry coordinates: absolute difference ≤ 1e-4 per coordinate.
    - Telemetry numeric values: absolute difference ≤ 1e-6; string fields must match exactly.
  - Geometry ordering is compared with the expectation that:
    - Crop-style ops preserve the relative order of retained original objects.
    - Copy/paste ops keep all originals first in their original order and append duplicates in deterministic selection order.

## PatchOp invariants
- Determinism:
  - PatchOps MUST use the provided RNG object exclusively for randomness.
  - Given identical inputs and RNG seed, a PatchOp MUST produce identical outputs (images, geometries, and telemetry).
- Ordering:
  - Patch selection and placement MUST be deterministic for a fixed seed (no reliance on hash/dict iteration order).
  - When multiple patches are applied, their effects are applied in a defined order (e.g., selection order).
- Geometry and placement:
  - All placed patches MUST remain fully inside the target image bounds; attempts that would exceed bounds MUST be retried or skipped.
  - Copy/paste PatchOps MUST enforce an IoU/overlap rule when configured (e.g., `overlap_threshold`), skipping placements that violate it.
- Telemetry:
  - Crop-style PatchOps (e.g., random_crop) MUST:
    - Populate `last_kept_indices` as indices into the original geometry list for objects that survived the crop.
    - Populate `last_object_coverages` with a coverage value per kept index.
    - Use `last_crop_skip_reason` and `last_skip_counters` when the crop is skipped (e.g., `min_objects`, `line_object`).
  - Duplicate-style PatchOps (e.g., zoom/copy-paste) MUST:
    - Preserve all original objects and append new ones; they MUST NOT delete geometries unless explicitly configured as a crop.
    - Leave `last_kept_indices` and `last_object_coverages` unset/empty so that crop telemetry continues to reflect only crop-style operations.
    - Respect configured overlap/IoU rules when placing duplicates but do not emit coverage-based telemetry for them.
  - Object ordering:
    - Crop-style PatchOps keep retained originals in their original relative order.
    - Duplicate-style PatchOps append duplicates after all originals, in deterministic selection order.

## Rationale for duplication
- Duplicate-style PatchOps intentionally create additional instances of hard or small objects (e.g., screws, brackets, warning labels, cable segments) within the same scene.
- This increases positive examples and encourages the model to learn object appearance independent of their original spatial arrangement or business-specific layout patterns.

## Dense captioning object ordering (TL→BR)
- Background:
  - The dense-caption system prompt (`src/config/prompts.py`) and data conversion docs (`data_conversion/README.md`) require that objects be ordered **top-to-bottom, then left-to-right**:
    - Primary key: Y (smaller Y first = visually higher).
    - Secondary key: X (smaller X first = visually more left).
    - Reference points:
      - `bbox_2d`: top-left corner `(x1, y1)`.
      - `poly`: first vertex `(x1, y1)` after canonicalization.
      - `line`: leftmost endpoint (min X; break ties by min Y).
  - Data conversion enforces this order when writing JSONL, but augmentation (crop/rotate/copy-paste) can change object positions and break the TL→BR promise if no re-sorting occurs.
- Requirement:
  - After augmentation has produced final pixel-space geometries, the training pipeline MUST re-sort objects into TL→BR order before enumerating them into `object_1`, `object_2`, … for dense captioning.
  - The TL→BR sort MUST use the same comparison rules as data conversion to keep behavior consistent across pipelines.
- Implementation guidance:
  - Introduce a shared TL→BR sort helper (reusing or mirroring `data_conversion/utils/sorting.py`).
  - Invoke this helper in the dense-caption builder (e.g., `JSONLinesBuilder._build_group_entry`) so that the sequence seen by the model always respects the TL→BR contract, regardless of internal augmentation ordering or duplication.

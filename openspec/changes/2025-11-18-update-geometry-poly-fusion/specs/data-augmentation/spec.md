# data-augmentation Delta (update-geometry-poly-fusion)

## MODIFIED Requirements

### Requirement: Geometry correctness after augmentation
All geometries (bbox_2d, poly, line) SHALL remain correct and valid in the augmented image coordinate frame.

#### Scenario: bbox remains axis-aligned and within bounds
- **WHEN** rotate/flip/scale are applied
- **THEN** bbox is recomputed as the AABB of transformed corners, quantized to int, and clamped to `[0,W-1]×[0,H-1]`

#### Scenario: poly preserves point order and stays in-bounds
- **WHEN** affine transforms are applied
- **THEN** poly points are transformed independently, quantized, and clamped; the geometry type remains `poly`

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

### Requirement: Preserve geometry under affine transforms
- The augmentation pipeline SHALL accumulate affines across sequential affine ops and apply a single transform to geometry per flush.
- For any non-axis-aligned affine (rotation, shear, non-uniform scale about non-origin, 90° rotates), bbox inputs MUST be converted to 4-point polys by transforming their four corners; do NOT re-enclose as an axis-aligned bbox.
- For axis-aligned affines (pure hflip/vflip, integer translate, uniform scale about origin/image center), bbox inputs MAY remain bboxes after exact transform.
- Polys MUST transform all vertices with the affine and preserve vertex identity (no re-enclosure to bbox).
- Lines/polylines MUST transform all vertices with the affine; clipping rules are defined separately.

#### Scenario: rotate→resize_by_scale
- WHEN input contains bbox [x1,y1,x2,y2], rotate 15° about image center, THEN resize_by_scale 0.9
- THEN output geometry type is poly (clockwise), vertices are the transformed four corners, and the shape is clipped to new image bounds.

#### Scenario: axis-aligned compose
- WHEN input contains bbox and ops are hflip THEN resize_by_scale 2.0 (aligned to multiple)
- THEN output remains a bbox with correctly mirrored and scaled coordinates within [0..W'-1, 0..H'-1].

### Requirement: Exact polygon clipping
- Polygons (including rotated 4-point polys) MUST be clipped against the image rectangle using Sutherland–Hodgman (convex) or an equivalent exact convex clipping algorithm.
- Clipping MUST occur in floating-point space; rounding to integer grid SHALL happen only after clipping.
- Coordinate bounds after rounding MUST lie in [0..W-1]×[0..H-1]; self-intersections are forbidden.

#### Scenario: edge crossing
- GIVEN a rotated poly that crosses the right boundary
- WHEN clipping is applied
- THEN resulting vertices lie on the image edge and within bounds; the polygon remains simple (non self-intersecting).

### Requirement: Clockwise ordering and degeneracy handling
- All polys MUST be clockwise ordered; duplicate or collinear vertices MUST be deduped.
- If fewer than 3 unique points remain after clipping, the geometry MUST be dropped:
  - **For crop operations**: Dropped objects are not included in filtered output (expected behavior)
  - **For non-crop operations**: Prefer original geometry; otherwise drop with logged warning

#### Scenario: Poly clipped to <3 points by crop boundary
- **WHEN** a poly is clipped against crop rectangle using Sutherland-Hodgman
- **AND** resulting polygon has <3 unique vertices
- **THEN** the object is dropped (not included in output geometries)
- **AND** counts towards filtered objects in logging

#### Scenario: collapse at corner (non-crop operations)
- **WHEN** a very thin rotated bbox is clipped near a corner and collapses to <3 unique points
- **AND** operation is NOT a crop (e.g., rotate, resize_by_scale)
- **THEN** the object is preserved with clamped fallback geometry (degenerate poly at bounds)
- **AND** a warning is logged including record index and object index

### Requirement: Minimum-area rectangle fallback
- When general polygons are not emitted (e.g., configuration chooses to approximate shapes by rectangles), the clipped polygon MUST be approximated by a minimum-area rectangle and output as a 4-point poly.

#### Scenario:
- Rotated box clipped into a pentagon (due to previous transforms). Since free-form polys are disabled, return the best-fit 4-point poly.

### Requirement: Typed geometry value objects
- The system SHALL provide immutable value objects for geometry: `BBox`, `Poly`, and `Polyline`.
- Each value object MUST expose `apply_affine(M)` returning a same-type or promoted-type (e.g., `BBox.apply_affine` MAY return `Poly` for general affines).
- Value objects MUST NOT access global state or image data; clipping and rounding are performed by explicit functions.

#### Scenario: Affine application and promotion
- **WHEN** the augmentation pipeline calls `BBox.apply_affine` with a non-axis-aligned matrix
- **THEN** it receives a `Poly` instance with transformed vertices, preserving immutability and keeping geometry-specific logic encapsulated inside the value object.

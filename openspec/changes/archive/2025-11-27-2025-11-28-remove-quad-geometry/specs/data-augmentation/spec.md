## ADDED Requirements

### Requirement: Polygon geometry key only
The system SHALL reject the `quad` geometry field; polygons MUST use the `poly` key.

#### Scenario: Record contains quad
- **WHEN** a training/eval sample includes `quad: [...]`
- **THEN** validation FAILS with a clear message to rename the key to `poly`.

#### Scenario: Multiple geometry keys including quad
- **WHEN** an object provides `bbox_2d` and `quad` (or `poly` and `quad`)
- **THEN** the record is rejected; only one geometry key is allowed and `quad` is disallowed.

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

#### Scenario: non-degenerate outputs
- **WHEN** transformed geometries collapse below configured thresholds
- **THEN** the record uses original geometries (strict=false) or raises (strict=true)

#### Scenario: keep absolute pixel coordinates
- **WHEN** any augmentation is applied
- **THEN** all geometry coordinates remain integer pixels in the image frame `[0, W-1]×[0, H-1]` (no normalization to 0–1000 here)

### Requirement: Single transform entrypoint
- A single function `transform_geometry(geom, M, width, height, policy)` MUST handle:
  - Applying affine to the value object
  - Polygon/polyline clipping against the image rect
  - Rounding and clamping according to `policy`
- All augmentation paths MUST use this entrypoint to avoid divergence.
- **Exception**: Crop operations use specialized clipping logic that combines coverage filtering with boundary truncation before calling transform_geometry for final cleanup.

#### Scenario: Crop uses specialized clipping path
- **GIVEN** a random_crop operation with min_coverage=0.3
- **WHEN** filtering and clipping geometries
- **THEN** crop operator computes coverage using AABB intersection
- **AND** clips geometries to crop boundary using Sutherland-Hodgman (polys) or Cohen-Sutherland (lines)
- **AND** translates results to crop coordinates
- **AND** optionally calls transform_geometry for final clamping/validation

### Requirement: Axis-aligned vs. general affine classification
- The system MUST classify an accumulated affine as axis-aligned IFF it composes only flips, translations, and uniform scales about the origin/center with no rotation or shear (within a small numeric tolerance, e.g., |sinθ|<1e-6).
- Axis-aligned affines MAY keep bboxes as bboxes; general affines MUST convert bboxes to 4-point polys.
#### Scenario: hflip+uniform scale
- WHEN hflip is followed by uniform scale about center
- THEN the transform is classified axis-aligned and bbox remains bbox after transform.

### Requirement: Geometry Truncation at Crop Boundaries
For objects that meet the visibility threshold but extend beyond the crop region, geometries MUST be clipped to the crop rectangle before translation to the new coordinate system.

#### Scenario: Bbox truncation
- **WHEN** a bbox [100, 100, 300, 200] is partially inside crop region [150, 0, 400, 300]
- **AND** coverage is >=30%
- **THEN** bbox is clipped to [150, 100, 300, 200] (left edge truncated)
- **AND** translated to crop coordinates [0, 100, 150, 200]

#### Scenario: Poly truncation via polygon clipping
- **WHEN** a rotated poly extends beyond crop boundary
- **AND** coverage is >=30%
- **THEN** the poly is clipped using Sutherland-Hodgman algorithm against the crop rectangle
- **AND** if clipped polygon has >4 vertices, it is approximated by minimum-area rectangle
- **AND** result is translated to crop coordinates with clockwise ordering

#### Scenario: Line truncation via segment clipping
- **WHEN** a polyline has segments crossing crop boundary
- **AND** coverage is >=30% (based on AABB of line)
- **THEN** segments are clipped using Cohen-Sutherland algorithm
- **AND** only in-bounds segments are retained
- **AND** result is translated to crop coordinates with duplicate points removed

### Requirement: Coverage Computation for Filtering
Coverage SHALL be computed as the ratio of object area inside the crop region to total object area, using axis-aligned bounding boxes for efficiency.

**Coverage is used for two purposes**:
1. **Filtering**: Drop objects with coverage < min_coverage (e.g., 0.3)
2. **Completeness**: Update "显示完整" → "只显示部分" for objects with coverage < completeness_threshold (e.g., 0.95)

#### Scenario: AABB-based coverage for all geometry types
- **WHEN** computing coverage for any geometry (bbox, poly, or line)
- **THEN** the system computes the geometry's axis-aligned bounding box (AABB)
- **AND** intersects the AABB with the crop rectangle
- **AND** returns coverage = intersection_area / geometry_aabb_area

#### Scenario: Zero coverage (fully outside)
- **WHEN** an object's AABB has no overlap with the crop region
- **THEN** coverage returns 0.0
- **AND** the object is always dropped regardless of threshold

#### Scenario: Full coverage (fully inside)
- **WHEN** an object's AABB is completely contained within the crop region
- **THEN** coverage returns 1.0
- **AND** the object is always retained

### Requirement: Geometry Translation After Crop
All retained geometries MUST be translated to the cropped image's coordinate system [0, crop_w-1] × [0, crop_h-1].

#### Scenario: Translate bbox coordinates
- **WHEN** crop region starts at (x_offset, y_offset)
- **AND** a bbox [x1, y1, x2, y2] is retained
- **THEN** bbox is translated to [x1 - x_offset, y1 - y_offset, x2 - x_offset, y2 - y_offset]
- **AND** coordinates are clamped to [0, crop_w-1] × [0, crop_h-1]

#### Scenario: Translate poly coordinates
- **WHEN** crop region starts at (x_offset, y_offset)
- **AND** a poly with 4 or more coordinates is retained
- **THEN** each coordinate pair (x, y) is translated to (x - x_offset, y - y_offset)
- **AND** coordinates are clamped to crop bounds

#### Scenario: Translate line coordinates
- **WHEN** crop region starts at (x_offset, y_offset)
- **AND** a line with N points is retained
- **THEN** all points are translated by (-x_offset, -y_offset)
- **AND** coordinates are clamped to crop bounds

### Requirement: Edge Case Handling for Crops
Crop operators MUST handle edge cases gracefully with well-defined fallback behavior.

#### Scenario: Crop larger than image
- **WHEN** sampled crop dimensions exceed image size
- **THEN** crop is clipped to image bounds [0, width] × [0, height]
- **AND** effective crop is the entire image (no-op)

#### Scenario: All objects filtered
- **WHEN** crop filters all objects (coverage < threshold for all)
- **AND** min_objects > 0
- **THEN** sample is skipped (preprocessor returns None)
- **AND** warning is logged with actionable guidance

#### Scenario: Degenerate geometry after clipping
- **WHEN** a poly is clipped to <3 vertices after crop boundary clipping
- **THEN** the object is considered dropped (counts as coverage < threshold)
- **AND** not included in output geometries

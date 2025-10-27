## ADDED Requirements

### Requirement: Smart Random Cropping with Label Filtering and Completeness Tracking
The system SHALL provide random crop operators that automatically filter objects, truncate geometries, AND update completeness fields (`显示完整`/`只显示部分`) to match visual reality in the cropped region.

#### Scenario: Random crop with coverage-based filtering and completeness update
- **WHEN** random_crop is applied with min_coverage=0.3 and completeness_threshold=0.95
- **AND** an object has 80% of its area inside the crop region
- **THEN** the object is retained and its geometry is clipped to the crop boundary
- **AND** the cropped geometry is translated to the new coordinate system [0, crop_w-1] × [0, crop_h-1]
- **AND** if the object description contains "显示完整", it is changed to "只显示部分"

#### Scenario: Drop objects below visibility threshold
- **WHEN** random_crop is applied with min_coverage=0.3
- **AND** an object has only 20% of its area inside the crop region
- **THEN** the object is removed from the output geometries list
- **AND** the builder does not include this object in the generated JSON response

#### Scenario: Fully visible objects
- **WHEN** an object is 100% inside the crop region
- **THEN** the object is retained with its full geometry translated to crop coordinates
- **AND** no clipping occurs (all vertices preserved)

#### Scenario: Skip crop operation when too few objects
- **WHEN** random_crop filters objects and fewer than 4 objects (min_objects) would remain
- **THEN** the crop operator returns original images and geometries unchanged
- **AND** a debug log is emitted: "Crop would filter to N < 4 objects. Skipping crop."
- **AND** no metadata (last_kept_indices, last_object_coverages) is set

#### Scenario: Skip crop operation when line objects present
- **WHEN** random_crop filters objects and any retained object is a line
- **THEN** the crop operator returns original images and geometries unchanged
- **AND** a debug log is emitted: "Crop region contains line object. Skipping crop to preserve cable/fiber integrity."
- **AND** sample continues training with uncropped data

### Requirement: Geometry Truncation at Crop Boundaries
For objects that meet the visibility threshold but extend beyond the crop region, geometries MUST be clipped to the crop rectangle before translation to the new coordinate system.

#### Scenario: Bbox truncation
- **WHEN** a bbox [100, 100, 300, 200] is partially inside crop region [150, 0, 400, 300]
- **AND** coverage is >=30%
- **THEN** bbox is clipped to [150, 100, 300, 200] (left edge truncated)
- **AND** translated to crop coordinates [0, 100, 150, 200]

#### Scenario: Quad truncation via polygon clipping
- **WHEN** a rotated quad extends beyond crop boundary
- **AND** coverage is >=30%
- **THEN** quad is clipped using Sutherland-Hodgman algorithm against crop rectangle
- **AND** if clipped polygon has >4 vertices, it is approximated by minimum-area rectangle
- **AND** result is translated to crop coordinates with clockwise ordering

#### Scenario: Line truncation via segment clipping
- **WHEN** a polyline has segments crossing crop boundary
- **AND** coverage is >=30% (based on AABB of line)
- **THEN** segments are clipped using Cohen-Sutherland algorithm
- **AND** only in-bounds segments are retained
- **AND** result is translated to crop coordinates with duplicate points removed

### Requirement: Completeness Field Update Based on Coverage
The system MUST update object description completeness fields (`显示完整` → `只显示部分`) when crop-induced truncation makes objects partially visible.

#### Scenario: Fully visible object keeps completeness unchanged
- **WHEN** an object has 98% coverage (≥ completeness_threshold of 0.95)
- **AND** the object description contains "显示完整"
- **THEN** the completeness field remains "显示完整" (no update)

#### Scenario: Truncated object gets completeness updated
- **WHEN** an object has 70% coverage (< completeness_threshold of 0.95 but ≥ min_coverage of 0.3)
- **AND** the object description contains "显示完整"
- **THEN** the description is updated to replace "显示完整" with "只显示部分"

#### Scenario: Already partial object remains partial
- **WHEN** an object has 60% coverage
- **AND** the object description already contains "只显示部分"
- **THEN** the completeness field remains unchanged (already marked as partial)

#### Scenario: Objects without completeness field unchanged
- **WHEN** an object description does not contain "显示完整" or "只显示部分"
- **THEN** the description remains unchanged (no completeness update applied)

### Requirement: Coverage Computation for Filtering
Coverage SHALL be computed as the ratio of object area inside the crop region to total object area, using axis-aligned bounding boxes for efficiency.

**Coverage is used for two purposes**:
1. **Filtering**: Drop objects with coverage < min_coverage (e.g., 0.3)
2. **Completeness**: Update "显示完整" → "只显示部分" for objects with coverage < completeness_threshold (e.g., 0.95)

#### Scenario: AABB-based coverage for all geometry types
- **WHEN** computing coverage for any geometry (bbox, quad, or line)
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

### Requirement: Conditional Validation for Crop Operations
The validation logic in `apply_augmentations()` MUST allow geometry count changes for crop operations while maintaining strict validation for other operations.

#### Scenario: Crop operation with geometry drops
- **WHEN** a crop operator sets `allows_geometry_drops = True`
- **AND** the output has fewer geometries than input
- **THEN** validation passes and logs the count change at debug level
- **AND** includes context (original count, filtered count, operation name)

#### Scenario: Non-crop operation with count mismatch
- **WHEN** a non-crop operator (e.g., resize_by_scale) returns different geometry count
- **AND** the operator does not set `allows_geometry_drops = True`
- **THEN** validation raises ValueError with expected vs actual counts
- **AND** fails fast to catch bugs

### Requirement: Random Crop Operator Configuration
The system SHALL provide a `RandomCrop` operator with configurable crop size, aspect ratio, coverage thresholds (for filtering and completeness), minimum object count, and line-skipping behavior.

#### Scenario: Configurable crop scale range
- **WHEN** random_crop is configured with scale=[0.6, 1.0]
- **THEN** each crop samples a uniform scale factor in [0.6, 1.0]
- **AND** crop dimensions are computed as (width × scale, height × scale)

#### Scenario: Configurable aspect ratio variation
- **WHEN** random_crop is configured with aspect_ratio=[0.8, 1.2]
- **THEN** each crop samples an aspect ratio in [0.8, 1.2]
- **AND** crop dimensions are adjusted to match the sampled ratio
- **AND** crop remains within image bounds

#### Scenario: Configurable visibility threshold for filtering
- **WHEN** random_crop is configured with min_coverage=0.3
- **THEN** objects with coverage >= 0.3 are retained and clipped
- **AND** objects with coverage < 0.3 are dropped from filtered set

#### Scenario: Configurable completeness threshold
- **WHEN** random_crop is configured with completeness_threshold=0.95
- **THEN** objects with coverage < 0.95 get "显示完整" → "只显示部分" update
- **AND** objects with coverage >= 0.95 keep original completeness field

#### Scenario: Configurable minimum object count for dense scenes
- **WHEN** random_crop is configured with min_objects=4
- **THEN** if filtered objects < 4, the entire crop operation is skipped
- **AND** original images and geometries are returned unchanged

#### Scenario: Configurable line-skipping behavior
- **WHEN** random_crop is configured with skip_if_line=true
- **AND** any filtered object contains a "line" geometry field
- **THEN** the entire crop operation is skipped to preserve cable/fiber integrity

#### Scenario: Random crop position
- **WHEN** random_crop selects a crop region
- **THEN** the top-left position is sampled uniformly from valid positions
- **AND** the crop rectangle stays within [0, width] × [0, height]

#### Scenario: Deterministic crop with fixed seed
- **WHEN** random_crop is applied with a fixed RNG seed
- **THEN** the same crop region is selected across runs
- **AND** the same objects are filtered with identical coverage values

### Requirement: Center Crop Operator
The system SHALL provide a `CenterCrop` operator as a drop-in replacement for `scale` zoom-in with proper label filtering and completeness update.

#### Scenario: Center crop with configurable zoom
- **WHEN** center_crop is configured with scale=0.75 (equivalent to 1.33x zoom)
- **THEN** crop region is computed as center (width × 0.75) × (height × 0.75)
- **AND** positioned at ((width - crop_w) / 2, (height - crop_h) / 2)

#### Scenario: Center crop with label filtering and completeness update
- **WHEN** center_crop is applied with min_coverage=0.3 and completeness_threshold=0.95
- **THEN** edge objects with <30% coverage are dropped
- **AND** objects with 30-95% coverage get "显示完整" → "只显示部分"
- **AND** objects with 95%+ coverage keep original completeness
- **AND** remaining objects are clipped and translated
- **AND** generated JSON only describes visible objects with correct completeness

### Requirement: Geometry Translation After Crop
All retained geometries MUST be translated to the cropped image's coordinate system [0, crop_w-1] × [0, crop_h-1].

#### Scenario: Translate bbox coordinates
- **WHEN** crop region starts at (x_offset, y_offset)
- **AND** a bbox [x1, y1, x2, y2] is retained
- **THEN** bbox is translated to [x1 - x_offset, y1 - y_offset, x2 - x_offset, y2 - y_offset]
- **AND** coordinates are clamped to [0, crop_w-1] × [0, crop_h-1]

#### Scenario: Translate quad coordinates
- **WHEN** crop region starts at (x_offset, y_offset)
- **AND** a quad with 8 coordinates is retained
- **THEN** each coordinate pair (x, y) is translated to (x - x_offset, y - y_offset)
- **AND** coordinates are clamped to crop bounds

#### Scenario: Translate line coordinates
- **WHEN** crop region starts at (x_offset, y_offset)
- **AND** a line with N points is retained
- **THEN** all points are translated by (-x_offset, -y_offset)
- **AND** coordinates are clamped to crop bounds

### Requirement: Crop as Barrier Operation
Crop operators MUST have `kind="barrier"` to flush accumulated affine transforms before applying the crop.

#### Scenario: Affine flush before crop
- **WHEN** rotate(15°) is followed by random_crop
- **THEN** Compose flushes the rotation affine before encountering the crop barrier
- **AND** crop is applied to the rotated image
- **AND** geometry is transformed by rotation, then filtered and clipped by crop

#### Scenario: Crop updates canvas dimensions
- **WHEN** a crop reduces image size from 800×600 to 400×300
- **THEN** subsequent operations receive width=400, height=300
- **AND** all geometry coordinates are in [0, 399] × [0, 299]

### Requirement: Logging and Metrics for Crop Operations
Crop operations MUST log geometry filtering decisions and provide actionable feedback when samples are skipped.

#### Scenario: Log filtered object count
- **WHEN** random_crop filters 11 objects to 7 objects
- **THEN** a debug-level log is emitted: "Crop filtered 11 → 7 objects (min_coverage=0.3)"
- **AND** logging is rank-aware (only rank 0 in distributed training)

#### Scenario: Log skipped samples with guidance
- **WHEN** random_crop filters to 0 objects and min_objects=1
- **THEN** a warning is logged: "Crop filtered to 0 objects (min=1). Skipping sample. Reduce min_coverage or min_objects to keep more samples."
- **AND** the preprocessor returns None to skip the sample

#### Scenario: No logging for successful crops
- **WHEN** random_crop retains all objects (no filtering)
- **THEN** no debug logs are emitted (avoid noise)

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
- **WHEN** a quad is clipped to <3 vertices after crop boundary clipping
- **THEN** the object is considered dropped (counts as coverage < threshold)
- **AND** not included in output geometries

### Requirement: Integration with Existing Augmentation Pipeline
Crop operators MUST integrate seamlessly with existing augmentation operations (rotate, scale, expand_to_fit_affine, resize_by_scale, color ops).

#### Scenario: Rotate then crop (recommended order)
- **WHEN** pipeline is [rotate(15°), expand_to_fit_affine, random_crop]
- **THEN** image is rotated and expanded, then cropped
- **AND** geometries are transformed by rotation, clipped by expansion, then filtered and clipped by crop

#### Scenario: Crop then color augmentation
- **WHEN** pipeline is [random_crop, color_jitter, gamma]
- **THEN** crop is applied first (barrier flushes affines)
- **AND** color ops are deferred and applied after crop
- **AND** geometries are unaffected by color ops

#### Scenario: Multiple crops in sequence
- **WHEN** pipeline has [resize_by_scale(0.8), random_crop]
- **THEN** resize reduces dimensions first
- **AND** random_crop samples from the resized dimensions
- **AND** geometries are scaled then filtered/clipped

## MODIFIED Requirements

### Requirement: Clockwise ordering and degeneracy handling
- All quads/polys MUST be clockwise ordered; duplicate or collinear vertices MUST be deduped.
- If fewer than 3 unique points remain after clipping, the geometry MUST be dropped:
  - **For crop operations**: Dropped objects are not included in filtered output (expected behavior)
  - **For non-crop operations**: Prefer original geometry; otherwise drop with logged warning

#### Scenario: Quad clipped to <3 points by crop boundary
- **WHEN** a quad is clipped against crop rectangle using Sutherland-Hodgman
- **AND** resulting polygon has <3 unique vertices
- **THEN** the object is dropped (not included in output geometries)
- **AND** counts towards filtered objects in logging

#### Scenario: collapse at corner (non-crop operations)
- **WHEN** a very thin rotated bbox is clipped near a corner and collapses to <3 unique points
- **AND** operation is NOT a crop (e.g., rotate, resize_by_scale)
- **THEN** the object is preserved with clamped fallback geometry (degenerate quad at bounds)
- **AND** a warning is logged including record index and object index

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
- **AND** clips geometries to crop boundary using Sutherland-Hodgman (quads) or Cohen-Sutherland (lines)
- **AND** translates results to crop coordinates
- **AND** optionally calls transform_geometry for final clamping/validation

## REMOVED Requirements

None. This change is purely additive.


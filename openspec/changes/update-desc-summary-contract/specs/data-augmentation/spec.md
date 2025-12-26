# data-augmentation Spec Delta (update-desc-summary-contract)

## MODIFIED Requirements

### Requirement: Smart Random Cropping with Label Filtering and Completeness Tracking
The system SHALL provide random crop operators that automatically filter objects, truncate geometries, AND update completeness fields (`可见性=完整`/`可见性=部分`) to match visual reality in the cropped region.

#### Scenario: Random crop with coverage-based filtering and completeness update
- **WHEN** random_crop is applied with min_coverage=0.3 and completeness_threshold=0.95
- **AND** an object has 80% of its area inside the crop region
- **THEN** the object is retained and its geometry is clipped to the crop boundary
- **AND** the cropped geometry is translated to the new coordinate system [0, crop_w-1] × [0, crop_h-1]
- **AND** if the object description contains `可见性=完整`, it is changed to `可见性=部分`

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

### Requirement: Completeness Field Update Based on Coverage
The system MUST update object description completeness fields (`可见性=完整` → `可见性=部分`) when crop-induced truncation makes objects partially visible.

#### Scenario: Fully visible object keeps completeness unchanged
- **WHEN** an object has 98% coverage (≥ completeness_threshold of 0.95)
- **AND** the object description contains `可见性=完整`
- **THEN** the completeness field remains `可见性=完整` (no update)

#### Scenario: Truncated object gets completeness updated
- **WHEN** an object has 70% coverage (< completeness_threshold of 0.95 but ≥ min_coverage of 0.3)
- **AND** the object description contains `可见性=完整`
- **THEN** the description is updated to replace `可见性=完整` with `可见性=部分`

#### Scenario: Already partial object remains partial
- **WHEN** an object has 60% coverage
- **AND** the object description already contains `可见性=部分`
- **THEN** the completeness field remains unchanged (already marked as partial)

#### Scenario: Objects without completeness field unchanged
- **WHEN** an object description does not contain `可见性=完整` or `可见性=部分`
- **THEN** the description remains unchanged (no completeness update applied)

### Requirement: Coverage Computation for Filtering
Coverage SHALL be computed as the ratio of object area inside the crop region to total object area. Poly geometries MUST use polygon clipping against the crop rectangle; bbox/line geometries MUST use axis-aligned bounding box (AABB) intersection. If polygon clipping returns zero area and bbox fallback is enabled, the system MAY return AABB-based coverage.

**Coverage is used for two purposes**:
1. **Filtering**: Drop objects with coverage < min_coverage (e.g., 0.3)
2. **Completeness**: Update `可见性=完整` → `可见性=部分` for objects with coverage < completeness_threshold (e.g., 0.95)

#### Scenario: Polygon coverage via clipping
- **WHEN** computing coverage for a polygon geometry
- **THEN** the system clips the polygon against the crop rectangle
- **AND** returns coverage = clipped_polygon_area / polygon_area
- **AND** if clipped_polygon_area is 0 and bbox fallback is enabled, returns AABB-based coverage instead

#### Scenario: AABB-based coverage for bbox/line geometries
- **WHEN** computing coverage for a bbox_2d or line geometry
- **THEN** the system computes the geometry's axis-aligned bounding box (AABB)
- **AND** intersects the AABB with the crop rectangle
- **AND** returns coverage = intersection_area / geometry_aabb_area

#### Scenario: Zero coverage (fully outside)
- **WHEN** an object's geometry has no overlap with the crop region
- **THEN** coverage returns 0.0
- **AND** the object is always dropped regardless of threshold

#### Scenario: Full coverage (fully inside)
- **WHEN** an object's geometry is completely contained within the crop region
- **THEN** coverage returns 1.0
- **AND** the object is always retained

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
- **THEN** objects with coverage < 0.95 get `可见性=完整` → `可见性=部分` update
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
- **AND** objects with 30-95% coverage get `可见性=完整` → `可见性=部分`
- **AND** objects with 95%+ coverage keep original completeness
- **AND** remaining objects are clipped and translated
- **AND** generated JSON only describes visible objects with correct completeness

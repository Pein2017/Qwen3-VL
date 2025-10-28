# data-augmentation Specification

## Purpose
TBD - created by archiving change add-image-augmentation-plugin. Update Purpose after archive.
## Requirements
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

### Requirement: Preserve geometry under affine transforms
- The augmentation pipeline SHALL accumulate affines across sequential affine ops and apply a single transform to geometry per flush.
- For any non-axis-aligned affine (rotation, shear, non-uniform scale about non-origin, 90° rotates), bbox inputs MUST be converted to quads by transforming their four corners; do NOT re-enclose as an axis-aligned bbox.
- For axis-aligned affines (pure hflip/vflip, integer translate, uniform scale about origin/image center), bbox inputs MAY remain bboxes after exact transform.
- Quads and polys MUST transform all vertices with the affine and preserve vertex identity (no re-enclosure to bbox).
- Lines/polylines MUST transform all vertices with the affine; clipping rules are defined separately.
#### Scenario: rotate→resize_by_scale
- WHEN input contains bbox [x1,y1,x2,y2], rotate 15° about image center, THEN resize_by_scale 0.9
- THEN output geometry type is quad (clockwise), vertices are the transformed four corners, and the shape is clipped to new image bounds.
#### Scenario: axis-aligned compose
- WHEN input contains bbox and ops are hflip THEN resize_by_scale 2.0 (aligned to multiple)
- THEN output remains a bbox with correctly mirrored and scaled coordinates within [0..W'-1, 0..H'-1].

### Requirement: Exact polygon clipping
- Polygons (including rotated quads) MUST be clipped against the image rectangle using Sutherland–Hodgman (convex) or an equivalent exact convex clipping algorithm.
- Clipping MUST occur in floating-point space; rounding to integer grid SHALL happen only after clipping.
- Coordinate bounds after rounding MUST lie in [0..W-1]×[0..H-1]; self-intersections are forbidden.
#### Scenario: edge crossing
- GIVEN a rotated quad that crosses the right boundary
- WHEN clipping is applied
- THEN resulting vertices lie on the image edge and within bounds; the polygon remains simple (non self-intersecting).

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

### Requirement: Minimum-area rectangle fallback
- When `poly` is not emitted, the clipped polygon MUST be approximated by a minimum-area rectangle and output as `quad`.
#### Scenario:
- Rotated box clipped into a pentagon (due to previous transforms). Since `poly` disabled, return the best-fit quad.

### Requirement: Typed geometry value objects
- The system SHALL provide immutable value objects for geometry: `BBox`, `Quad`, and `Polyline`.
- Each value object MUST expose `apply_affine(M)` returning a same-type or promoted-type (e.g., `BBox.apply_affine` MAY return `Quad` for general affines).
- Value objects MUST NOT access global state or image data; clipping and rounding are performed by explicit functions.
#### Scenario: bbox rotated 30°
- WHEN `BBox(…)
` receives a general affine M (rotation)
- THEN `apply_affine` returns a `Quad` with four transformed corners in clockwise order.

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

### Requirement: Debug alignment checks
- In debug mode, the system MUST compute an IoU between an image-warped mask of the original shape and the transformed geometry; if IoU < 0.99, it MUST log M, pivot, and save an overlay.
#### Scenario: large rotation mismatch
- WHEN a 40° rotation causes drift
- THEN an overlay PNG and a JSON log are saved with matrix, pivot, and metrics.

### Requirement: Robust polyline (line) clipping
- Polylines MUST be clipped against the image rectangle using segment-wise clipping (e.g., Cohen–Sutherland or Liang–Barsky) in floating-point space, then rounded.
- Consecutive duplicate points MUST be removed; segments shorter than 1 pixel after rounding MUST be dropped.
- If fewer than 2 points remain (i.e., <1 segment), the object MUST be dropped with a logged warning.
#### Scenario: line exiting frame
- GIVEN a 5-vertex polyline where two segments exit and re-enter the frame
- WHEN clipping is applied
- THEN the resulting polyline contains only in-frame segments with endpoints on the border; if <2 points remain the object is dropped with a warning.

### Requirement: Axis-aligned vs. general affine classification
- The system MUST classify an accumulated affine as axis-aligned IFF it composes only flips, translations, and uniform scales about the origin/center with no rotation or shear (within a small numeric tolerance, e.g., |sinθ|<1e-6).
- Axis-aligned affines MAY keep bboxes as bboxes; general affines MUST convert bboxes to quads.
#### Scenario: hflip+uniform scale
- WHEN hflip is followed by uniform scale about center
- THEN the transform is classified axis-aligned and bbox remains bbox after transform.

### Requirement: Integer rounding and bounds
- After applying affines and clipping (polygons/lines), coordinates MUST be rounded to the nearest integer and clamped to [0..W-1]×[0..H-1].
- Rounding MUST be applied consistently across all geometry types.
#### Scenario: rounding after clip
- GIVEN a polygon with fractional vertices after clipping
- WHEN rounding is applied
- THEN all coordinates are integers within bounds, and polygon simplicity is preserved.

### Requirement: Per-sample affine determinism and multi-image invariants
- All images in a sample MUST receive identical affine parameters per op application; geometry transforms MUST be identical across images for a given sample.
- Barrier ops that change size (e.g., padding, resize_by_scale) MUST update record `width`/`height` accordingly.
#### Scenario: two-image sample
- GIVEN a sample with two images
- WHEN rotate(8°) and hflip are applied
- THEN both images are rotated and flipped identically; all corresponding object geometries are transformed consistently, and output sizes match.

### Requirement: Canvas expansion to enclose affine
- Before applying a general affine (rotation, shear, non-uniform scale), the system SHALL optionally expand the canvas so that the entire original image is enclosed after the transform without truncation.
- Expansion SHALL compute the transformed image corners under M and choose a canvas that fully contains the bounding quad; empty regions are filled with middle gray RGB(128,128,128).
- After expansion, the affine SHALL be recomposed so that geometry and images align in the expanded coordinate system without loss.
#### Scenario: rotate 45° on 768×1024
- GIVEN a 768×1024 image and 45° rotation
- WHEN canvas expansion is enabled
- THEN the new canvas encloses all pixels; no geometry is clipped.

### Requirement: Multiple-of-32 sizing without truncation
- The final canvas width and height MUST be rounded up to the nearest multiple of 32 by padding (no cropping).
- Padding SHALL use middle gray RGB(128,128,128) to achieve zero in Qwen3-VL's normalized space.
- Geometry coordinates MUST remain valid and unchanged by the padding except for clamping to the new bounds.
#### Scenario: post-rotate pad
- WHEN rotation yields 1100×1100 canvas
- THEN pad to 1120×1120; geometry remains identical in pixel positions.

### Requirement: Neutral padding color to minimize distribution shift
- All padding and fill operations (canvas expansion, affine warps, alignment padding) SHALL use middle gray RGB(128,128,128).
- This color MUST map to approximately zero after Qwen3-VL's symmetric normalization: `(pixel/255 - 0.5) / 0.5`.
- Black (0,0,0) or white (255,255,255) SHALL NOT be used as they create artificial high-contrast boundaries (normalize to ±1.0).
#### Scenario: affine transform fillcolor
- WHEN `PIL.Image.transform()` applies an affine warp
- THEN the `fillcolor` parameter is set to `(128, 128, 128)`
#### Scenario: padding to multiple
- WHEN creating a new canvas via `Image.new()` for padding to 32-multiple
- THEN the background color is `(128, 128, 128)`
#### Scenario: normalized value verification
- GIVEN Qwen3-VL normalization: `image_mean=[0.5,0.5,0.5]`, `image_std=[0.5,0.5,0.5]`
- WHEN padding pixel value is 128
- THEN normalized value is `(128/255 - 0.5) / 0.5 ≈ 0.003 ≈ 0`

### Requirement: Pre-flush hook protocol for barriers
- The system SHALL support an optional `pre_flush_hook(M_total, width, height, rng)` method on barrier operators with `kind="barrier"` to modify accumulated affine matrix and canvas dimensions BEFORE warping occurs.
- The hook MUST return `(M_total_modified, new_width, new_height)` as a tuple.
- Compose SHALL call the hook before flushing affines if it exists, using returned values for the warp operation.
- Hooks MUST be deterministic (use provided rng) and MUST NOT access images or geometry directly.
#### Scenario: canvas expansion before rotation warp
- GIVEN `ExpandToFitAffine` implements `pre_flush_hook()`
- WHEN Compose encounters this barrier after accumulated rotation affine
- THEN hook computes expanded dimensions, translates affine, returns updated values
- AND Compose warps images using the expanded dimensions and modified affine

### Requirement: Pixel limit safety with proportional scaling
- Canvas expansion operations MUST enforce a configurable `max_pixels` limit (default: 921600 for Qwen3-VL).
- If expanded dimensions exceed `max_pixels`, the system SHALL scale down proportionally using `scale_factor = sqrt(max_pixels / pixel_count)`.
- The scaling SHALL be applied to both affine matrix (via `scale_matrix`) and dimensions.
- Dimensions after scaling SHALL still be rounded to the configured multiple (e.g., 32).
- A warning SHALL be logged with: original dimensions, scaled dimensions, scale factor, and suggestion to reduce augmentation strength.
#### Scenario: large rotation exceeds pixel limit
- GIVEN a 1024×1024 image, 30° rotation would expand to 1344×1344 = 1,806,336 pixels
- WHEN max_pixels is 921600 (960×960)
- THEN system scales down by factor ~0.706
- AND final dimensions are 960×960 (or nearest 32-multiple)
- AND warning is logged: "Canvas expansion (1344×1344 = 1806336 pixels) exceeds max_pixels=921600..."

### Requirement: Rank-aware centralized logging
- Augmentation operations SHALL use centralized logging via `src/utils/logger.py`.
- In distributed training, only rank 0 SHALL log by default (unless QWEN3VL_VERBOSE=1).
- Warning messages MUST include actionable information (dimensions, scale factors, suggestions).
#### Scenario: distributed training with 4 GPUs
- GIVEN training with 4 GPUs and pixel limit warnings
- WHEN warnings would be triggered
- THEN only rank 0 emits log messages
- AND messages contain specific dimensions and remediation advice

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


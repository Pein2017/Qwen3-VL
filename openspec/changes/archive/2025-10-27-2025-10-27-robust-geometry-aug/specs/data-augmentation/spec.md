## ADDED Requirements

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
- If fewer than 3 unique points remain, the geometry MUST fallback:
  - Prefer original geometry; otherwise drop the object with a logged warning.
#### Scenario: collapse at corner
- WHEN a very thin rotated bbox is clipped near a corner and collapses to <3 unique points
- THEN the object is dropped with a warning including record index and object index.

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
#### Scenario: rotate→resize_by_scale
- GIVEN a bbox and two ops composing to M_total
- WHEN `transform_geometry` is called once with M_total
- THEN output equals the result of per-op application with identical math.

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
- Expansion SHALL compute the transformed image corners under M and choose a canvas that fully contains the bounding quad; empty regions are filled with a constant color.
- After expansion, the affine SHALL be recomposed so that geometry and images align in the expanded coordinate system without loss.
#### Scenario: rotate 45° on 768×1024
- GIVEN a 768×1024 image and 45° rotation
- WHEN canvas expansion is enabled
- THEN the new canvas encloses all pixels; no geometry is clipped.

### Requirement: Multiple-of-32 sizing without truncation
- The final canvas width and height MUST be rounded up to the nearest multiple of 32 by padding (no cropping).
- Geometry coordinates MUST remain valid and unchanged by the padding except for clamping to the new bounds.
#### Scenario: post-rotate pad
- WHEN rotation yields 1100×1100 canvas
- THEN pad to 1120×1120; geometry remains identical in pixel positions.

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



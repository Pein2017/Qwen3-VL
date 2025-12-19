# Design: Multi-object and Line Copy-Paste Augmentation

## 1. Overview

We extend the existing augmentation PatchOps to better cover small objects and line geometries while preserving the current registry, Compose, and curriculum plumbing.

Key elements:
- Extend `SmallObjectZoomPaste` with background-aware placement, multi-copy mode, and an optional patch bank. Existing configs remain semantically equivalent by default.
- Introduce two new PatchOps:
  - `ObjectClusterCopyPaste` for local multi-object clusters.
  - `LineSegmentCopyPaste` for cable/line segments.
- Implement shared helpers for occupancy grids and patch placement, plus a small in-memory patch bank mixin.

## 2. Shared helpers in `ops.py`

### 2.1 Occupancy grid

We define helpers operating on the current geometry list and canvas size:

- `_compute_occupancy_grid(geoms, width, height, rows, cols, line_buffer)`
  - Partition `[0, W)×[0, H)` into `rows × cols` cells.
  - For each geom, approximate coverage per cell using:
    - `compute_polygon_coverage(geom, cell_bbox, fallback="bbox")` for bbox/poly.
    - For lines, use `get_aabb` plus `_buffer_aabb` with `line_buffer` to give them thickness.
  - Returns a 2D array of coverage values in `[0,1]`.

- `_sample_background_cell(occupancy, max_coverage, rng)`
  - From cells whose coverage ≤ `max_coverage`, sample one index uniformly (or weighted by remaining free area) and return its bbox.
  - Returns `None` if no such cells exist.

### 2.2 Patch placement helper

- `_try_place_patch(source_bbox, scale_range, width, height, working_geoms, occupancy_cfg, rng)`
  - Arguments:
    - `source_bbox = [x1, y1, x2, y2]` in the current canvas.
    - `scale_range = (lo, hi)`.
    - `working_geoms`: list of existing + newly-added geoms in the augmented image.
    - `occupancy_cfg`: struct containing `rows`, `cols`, `background_max_coverage`, `max_cell_coverage_after`, `overlap_threshold`, `line_buffer`, `max_attempts`.
  - Steps:
    1. Build or reuse occupancy grid for the current `working_geoms`.
    2. For up to `max_attempts`:
       - Choose a target cell: use `_sample_background_cell` if configured, otherwise cover the full image.
       - Sample a scale `s` from `scale_range`.
       - Compute `new_w`, `new_h` from scaled `source_bbox`.
       - Sample `(tx, ty)` so that `[tx, ty, tx+new_w, ty+new_h]`
         - lies within `[0, W)×[0, H)`, and
         - obeys IoU constraints vs `working_geoms` using `_aabb_iou` and `_buffer_aabb`.
       - Optionally recompute occupancy for affected cells and ensure per-cell coverage does not exceed `max_cell_coverage_after`.
    3. If a valid placement is found, return `(tx, ty, s)`; else return `None`.

This helper is used by all copy-paste PatchOps so IoU and background-aware behavior stay consistent.

## 3. Patch bank mixin

We introduce a small reusable mixin:

```python
class PatchBankMixin:
    def __init__(self, *, bank_capacity: int = 256, **kwargs):
        super().__init__(**kwargs)
        self._patch_bank: List[PatchEntry] = []
        self._bank_capacity = int(bank_capacity)

    def _maybe_add_patches_from_sample(self, images, geoms, width, height, rng):
        ...

    def _sample_patch_from_bank(self, kind: str, filters, rng) -> Optional[PatchEntry]:
        ...
```

- `PatchEntry` is a lightweight dataclass holding:
  - `kind: Literal["small", "cluster", "line"]`.
  - `image_patch: Image.Image` (RGB crop).
  - `geoms: List[Dict[str, Any]]` (geometries in patch-local coordinates).
  - Optional stats (e.g., `aabb`, `length` for lines) for filtering.
- The bank is per-op-instance, hence per-dataloader-worker.
- Capacity is a hard cap (simple FIFO eviction when full).
- For determinism, the bank only mutates based on the per-sample RNG already passed into `apply()`.

Ops that want cross-image reuse opt in by inheriting `PatchBankMixin` and calling `_maybe_add_patches_from_sample()` and `_sample_patch_from_bank()` with an appropriate `kind`.

## 4. Extended `SmallObjectZoomPaste`

We keep the existing class but add optional parameters and behavior:

New params (with safe defaults):
- `placement_mode: Literal["uniform", "background"] = "uniform"`
- `grid_rows: int = 0`, `grid_cols: int = 0` (0 → disable grid / keep old behavior)
- `background_max_coverage: float = 0.05`
- `max_cell_coverage_after: float = 0.4`
- `max_copies_per_target: int = 1`
- `max_total_pastes: int | None = None`
- `source_mode: Literal["local", "bank", "mixed"] = "local"`
- `bank_capacity: int = 256`
- `bank_add_prob: float = 0.0` (off by default)

Behavior:
- When `placement_mode == "uniform"` or `grid_rows == 0` or `grid_cols == 0`, selection and placement are identical to the current implementation.
- When `placement_mode == "background"` and grid is configured:
  - Use the occupancy helpers to bias target positions towards low-coverage cells.
- For each candidate small object, we attempt up to `max_copies_per_target` valid placements, subject to `max_total_pastes` for the image.
- If `source_mode != "local"`, we:
  - Occasionally push small-object patches from the current sample into the bank when `rng.random() < bank_add_prob`.
  - Sample some candidate patches from the bank (kind `"small"`) to augment the target set.

This preserves existing semantics under default parameters while enabling optional densification and object-bank behavior.

## 5. `ObjectClusterCopyPaste` PatchOp

This new PatchOp copies multi-object clusters.

Parameters (initial):
- `prob: float`
- `min_objects_in_cluster: int`
- `max_objects_in_cluster: int`
- `cluster_radius_px: float`
- `cluster_context: float`
- `max_cluster_size: float` (max of width/height in pixels)
- `max_total_clusters: int`
- `placement_mode`, `grid_rows`, `grid_cols`, `background_max_coverage`, `max_cell_coverage_after`
- `source_mode`, `bank_capacity`, `bank_add_prob`

Algorithm:
1. **Cluster construction (local source)**
   - Choose seed objects from small/medium geoms (potentially filtered by class).
   - For each seed, build a region by expanding its AABB by `cluster_radius_px`.
   - Gather all geoms whose AABB intersects this region above `cluster_iou_threshold`.
   - Compute the union AABB and discard clusters that exceed `max_cluster_size` or fall outside `[min_objects_in_cluster, max_objects_in_cluster]`.
2. **Patch extraction**
   - Crop the union AABB expanded by `cluster_context` from the first image (multi-image behavior follows existing ops: same patch area on all images).
   - Normalize cluster geometries into patch-local coordinates.
   - Optionally add this cluster to the patch bank with kind `"cluster"`.
3. **Placement**
   - For each selected cluster (local and/or bank-sourced), call `_try_place_patch` with the union AABB and scale range.
   - Build an affine (scale+translate) and apply `transform_geometry` to each cluster geom; append them to `working_geoms` on success.

Ordering and telemetry follow the existing PatchOp rules: originals retained, duplicates appended in consistent order, crop telemetry unused.

## 6. `LineSegmentCopyPaste` PatchOp

This PatchOp targets `line` geometries.

Parameters (initial):
- `prob: float`
- `min_line_length: float`
- `max_line_length: float`
- `min_segment_length: float`
- `max_segment_length: float`
- `completeness_ratio: float` (fraction of segment length that must be inside patch)
- `line_context: float`
- `max_line_segments: int`
- `scale: Tuple[float, float]` (narrow range, e.g. [0.9, 1.1])
- `grid_rows`, `grid_cols`, `background_max_coverage`, `max_cell_coverage_after`
- `overlap_threshold: float`
- `line_buffer: float`
- `source_mode`, `bank_capacity`, `bank_add_prob`
- Optional `class_whitelist` for `desc` filtering.

Algorithm:
1. **Candidate selection**
   - Iterate `geoms` with `"line"`.
   - Compute total length by summing Euclidean distances along the polyline.
   - Keep candidates whose length is in `[min_line_length, max_line_length]` and whose class passes the whitelist.
2. **Segment extraction**
   - If length ≤ `max_segment_length`, use the full line.
   - Otherwise, sample a start position along the line and walk forward to accumulate `segment_length ∈ [min_segment_length, max_segment_length]`.
   - Interpolate along edges as needed to derive a sub-polyline.
3. **Patch extraction**
   - Compute AABB for the (sub)line and expand by `line_context` pixels.
   - Crop this patch; normalize line coordinates into patch-local frame.
   - Optionally add it to the patch bank (kind `"line"`).
4. **Placement**
   - Use `_try_place_patch` with a **narrow scale range** to keep curvature visually similar.
   - Construct an affine: uniform scale about patch center + translation to `(tx, ty)`.
   - Use `transform_geometry` to get the new line; clipping via `clip_polyline_to_rect` ensures it stays in-bounds.
   - Reject placements whose AABB IoU with existing bbox/poly geoms exceeds `max_iou_with_boxes` (derived from `overlap_threshold`) or whose IoU with existing lines exceeds a separate line-specific threshold.

This respects the design constraint that lines can appear anywhere, but their shape and recognizability are preserved.

## 7. Determinism and performance

- All new behavior is driven by the existing RNG passed into `Compose.apply()`; patch bank mutations and candidate selection must only use this RNG.
- Bank sizes and per-image caps (e.g., `max_total_pastes`, `max_total_clusters`, `max_line_segments`) keep geometry counts bounded.
- Occupancy grids are coarse (e.g., 6–10 cells per dimension) to limit cost; they can be recomputed lazily only when placement_mode is `"background"`.

## 8. Backwards compatibility

- Default parameters for `SmallObjectZoomPaste` produce identical behavior to the current implementation.
- New PatchOps are opt-in via YAML; existing configs and tests remain valid.
- Golden fixtures from previous augmentation changes remain authoritative; new fixtures will cover the new ops.


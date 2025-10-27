## 1. Geometry types & transform entrypoint
- [ ] 1.1 Create `src/datasets/geometry_types.py` with frozen dataclasses: `BBox`, `Quad`, `Polyline` (value objects; no I/O).
- [ ] 1.2 Implement `apply_affine(M)` on each; `BBox.apply_affine` returns `BBox` for axis-aligned, `Quad` for general.
- [ ] 1.3 Add `transform_geometry(geom, M, width, height, policy)` in `src/datasets/geometry.py` (or a new module) that applies: affine → clip → round → clamp.

## 2. Compose integration
- [ ] 2.1 Refactor `Compose._apply_affine_to_geoms` to wrap inputs as value objects and call `transform_geometry`.
- [ ] 2.2 Ensure accumulated `M_total` shares the exact pivot and math with image warp (pixel-center; PIL inverse handling baked).
- [ ] 2.3 Remove per-op bespoke geometry math from `ops.py` where redundant; keep `resize_by_scale` barrier semantics.
- [ ] 2.4 Implement `ExpandToFitAffine` barrier op: compute expanded canvas that encloses transformed image, translate affine accordingly, then optional pad-to-32.

## 3. Debug alignment tooling
- [ ] 3.1 Add optional debug hook: rasterize original shape, warp mask by image transform, compute IoU with transformed geometry; threshold 0.99.
- [ ] 3.2 On failure, export overlay and JSON (matrix, pivot, sizes, metrics) under `vis_out/debug_align/`.
- [ ] 3.3 Extend `vis_tools/vis_augment_compare.py` to toggle this and show IoU per pane.

## 4. Tests
- [ ] 4.1 Unit tests for `geometry_types` (promotion rules, CW ordering, equality).
- [ ] 4.2 Property tests over random boxes/angles/scales for `transform_geometry` IoU >= 0.99.
- [ ] 4.3 Golden tests for rotate→resize and hflip→resize equivalence with accumulated affine vs. per-op.
- [ ] 4.4 Regression tests for border-touching and thin boxes.

## 5. Docs & perf
- [ ] 5.1 Update `openspec` spec to include types and entrypoint semantics (done in this change).
- [ ] 5.2 Update `docs/DATA.md` for `poly` option and rounding policy.
- [ ] 5.3 Micro-benchmark to ensure <5% slowdown; if exceeded, optimize hotspots.


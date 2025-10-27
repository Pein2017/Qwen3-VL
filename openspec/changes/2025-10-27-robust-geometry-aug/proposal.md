---
title: Robust geometry transforms for augmentation (bbox→quad, clipping, CW order)
author: core
created: 2025-10-27
change-id: robust-geometry-aug-2025-10-27
status: draft
---

Problem
- Rotated bboxes are converted back to axis-aligned boxes, losing geometry and causing mismatches after rotate+resize.
- Geometry is “clamped” to the image by per-coordinate saturation, not proper polygon clipping → self-intersections and incorrect extents.
- Quad orientation is not enforced; clockwise/consistent vertex ordering drifts across ops.

Scope
- Augmentation-only geometry handling under `src/datasets/augmentation/{base.py,ops.py}` and helpers in `src/datasets/geometry.py`.
- JSONL schema alignment (DATA.md). Prefer compatibility, allow optional `poly` type if needed.

Decision summary
1) Promote bbox→quad under general affines; keep bbox for axis-aligned transforms.
2) Use exact convex polygon clipping (Sutherland–Hodgman) before rounding/clamping.
3) Enforce clockwise order; dedupe; drop degenerate (<3 unique points) with warnings.
4) Add typed immutable geometry value objects (`BBox`, `Quad`, `Polyline`) with `apply_affine` and pure semantics.
5) Introduce a single `transform_geometry(geom, M, width, height, policy)` used by all augmentation paths.
6) Add debug alignment tooling (mask-warp IoU and overlay export) to identify mismatches.
7) Add an optional canvas expansion barrier that encloses the full transformed image, followed by pad-to-multiple-of-32; never crop, avoid truncation.

Non-goals
- No change to template normalization or training loss; strictly data geometry.

Risks
- Slight performance hit from polygon clipping; mitigated by vectorized pure-Python and small vertex counts.



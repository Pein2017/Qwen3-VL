# Design: Single-image small-object zoom-and-paste

## Goals
- Boost recall for small objects (screws, connectors, cable endpoints) by enlarging and duplicating within the same image.
- Preserve semantics: no cross-image mixing; `desc` unchanged; geometry integrity for bbox/poly/line.
- Respect existing augmentation pipeline: affine accumulation, barrier flushes, 32× padding, pixel-cap safety.

## Approach
- **Op type:** barrier-like geometric op (not pure affine) that:
  1) flushes pending affines, operates in current pixel space,
  2) performs patch crop→scale→translate,
  3) returns updated images and geometries, with unchanged canvas size (unless optional padding is needed—prefer none).
- **Target selection:** filter objects whose area/length below thresholds; optional class whitelist.
- **Patch prep:** crop tight AABB with configurable context margin; allow poly/line masking via AABB (minimal cost) to extract patch.
- **Transform:** sample scale in [lo, hi]; sample translation within image so that patch stays in-bounds; apply same affine (scale+translate) to object geometry.
- **Overlap gating:** reject candidate placement if IoU/coverage with any existing object exceeds threshold (line → buffered polygon using stroke width).
- **Retry policy:** attempt up to `max_attempts` per target; if none valid, skip that target.
- **Failure safety:** if warp pushes geometry out of bounds or degenerates (<2 pts line, <3 pts poly), drop that pasted instance; keep originals always.
- **Probability:** op-level `prob`; limit per-image pasted count (`max_targets`).

## Integration points
- Register in `src/datasets/augmentation/ops.py` with `@register`.
- Use existing geometry helpers (`apply_affine`, `translate_geometry`, clipping/clamping utilities in `geometry.py`).
- Respect padding color (middle gray) only if new canvas ever created (avoid if staying in-bounds).
- Builder: no API change; op exposed via YAML `name: small_object_zoom_paste`.

## Open Questions / Defaults
- Stroke width for line overlap/IoU: default 4px; configurable.
- Context margin around patch: default 4–8px to keep local texture.
- Overlap metric: IoU on buffered polygons (line) or AABB IoU for speed; spec will require low-overlap gating, implementation can start with AABB IoU + buffer for lines.

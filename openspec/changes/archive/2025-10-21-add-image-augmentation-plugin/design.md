## Context
Current augmentation lives in `src/datasets/augment.py` and is called by `AugmentationPreprocessor`. We want a modular plugin system to add/compose image-level ops without editing core dataset logic.

## Goals / Non-Goals
- Goals: plugin registry; config-driven composition; strict validation; reproducible RNG; geometry-consistent transforms.
- Non-Goals: advanced AutoAugment/RandAugment policies; video augmentations; mixed modality transforms.

## Decisions
- Small interface: `ImageAugmenter.apply(images, geoms, rng) -> (images, geoms)`.
- Registry pattern: `register(name)(cls_or_fn)` with `get(name)`; configured via YAML list.
- Composition: `Compose([op1, op2, ...])` executes ops sequentially; each returns updated state.
- Validation: ensure uniform image sizes, update geometry via shared affine utilities from `geometry.py`.
- Determinism: inject `random.Random(seed)` via preprocessor; no module-level RNG state.

## Risks / Trade-offs
- PIL vs. torch: start with PIL for simplicity; consider `torchvision` later if needed.
- Performance: CPU transforms may bottleneck; consider batched/tensor ops if profiling shows issues.

## Migration Plan
- Keep `apply_augmentations` for backward compatibility; implement it via the new plugin pipeline.
- Add config to switch on the plugin path; default off.

## Open Questions
- Do we need per-dataset override hooks? (defer until requested)
- Should we expose probability per op vs. global? (start with per-op optional prob)

## Geometry & Coordinate System
- Coordinate frame: pixel space with origin at top-left; x → right, y → down.
- Record invariants (per `@data_details.md`):
  - Exactly one geometry field per object: `bbox_2d | quad | line`.
  - `width`, `height` are image dimensions and MUST remain accurate.
  - Geometry values on disk are integer pixels; template later normalizes to norm1000.
- Explicitly NOT permitted inside augmentation:
  - Do NOT normalize or rescale coordinates to `0–1000`. Keep absolute pixel coordinates corresponding to `(W, H)`.
  - Normalization to norm1000 is performed later by the ms‑swift/HF processor during encoding.
- Augmentations MUST preserve the original geometry type; no implicit conversion between `bbox_2d`, `quad`, and `line`.

## Affine Composition & Application
- Maintain a single 3×3 homogeneous affine `M` per record; apply the SAME `M` to all images/geometries in the record for consistency.
- Operation order (per config list): left-to-right composition, centered when applicable.
  - Horizontal flip: `(x', y') = (W - 1 - x, y)`
  - Vertical flip: `(x', y') = (x, H - 1 - y)`
  - Rotate θ about center `(cx, cy)=(W/2, H/2)`:
    - `x' = cosθ (x - cx) - sinθ (y - cy) + cx`
    - `y' = sinθ (x - cx) + cosθ (y - cy) + cy`
  - Scale `s` about center: `x' = s (x - cx) + cx`, `y' = s (y - cy) + cy`
- Image canvas size remains `(W, H)`; out-of-bounds areas are filled (e.g., black). No change to `width`/`height`.
- Color-only ops do not affect geometry.

## Geometry Update Procedures
- General: keep float precision during transform, then quantize to int for storage/emission; always clamp to `[0, W-1]` × `[0, H-1]`.

### bbox_2d (axis-aligned)
1) Expand to four corners `(x1,y1),(x2,y1),(x2,y2),(x1,y2)`.
2) Apply `M` to all four points; compute axis-aligned AABB: `x_min = min(xs)`, `x_max = max(xs)`, same for y.
3) Quantize: `x1 = round(x_min)`, `y1 = round(y_min)`, `x2 = round(x_max)`, `y2 = round(y_max)`.
4) Normalize ordering so `x1 ≤ x2`, `y1 ≤ y2`; clamp to bounds.
5) Degeneracy policy: if `(x2 - x1) < min_w_px` or `(y2 - y1) < min_h_px` after clamping ⇒ handle per policy (default: skip augmentation for this record and fall back to original images/geometries).

### quad (4-point polygon)
1) Apply `M` independently to the 4 points in original order; do not reorder.
2) Quantize each point; clamp to bounds.
3) Degeneracy policy: if all 4 points collapse to ≤1 pixel spread in x or y ⇒ default: skip augmentation; alternative (configurable): convert to AABB is NOT allowed by spec (must preserve types), so prefer skip.

### line (polyline)
1) Apply `M` to each `(xi, yi)`; keep vertex order.
2) Optional clipping (per config): perform segment-wise rectangle clipping (Liang–Barsky or Cohen–Sutherland) against `[0,W-1]×[0,H-1]` to avoid distorted clamping near edges; fallback: clamp each point.
3) Quantize to int; remove consecutive duplicate points.
4) Degeneracy policy: if <2 distinct points remain after clipping ⇒ default: skip augmentation for the record.

## Validation & Fail-Fast Policies
- Pre-checks:
  - All images in the record MUST share the same `(W, H)`; otherwise raise `ValueError`.
  - Input geometries MUST be valid per spec (one geometry field per object); otherwise raise.
- Post-transform checks per object:
  - Geometry within bounds (after clamp) and not degenerate per min-size thresholds (bbox) or point-count (line).
  - Preserve geometry type; exactly one geometry field per object remains set.
  - Coordinates remain integer pixel values within `[0, W-1]×[0, H-1]`; no `0–1000` normalization performed here.
- Record-level checks:
  - Number of objects unchanged.
  - `width`, `height` unchanged.
- Failure handling (default): skip augmentation for this record and use original images/geometries; log counter with reason. Optional strict mode: raise on any failure.

## Numeric & Rounding Rules
- Internal math uses float64 for stability; emit integer pixels using `round()` then clamp.
- Configurable `min_w_px`, `min_h_px` (defaults: 1 px); `clip_lines: true|false` (default true); `strict: false` by default.

## Multi-Image Consistency
- Apply identical `M` to all images/geometries in the record to maintain cross-image alignment required by grouping logic.
- If any image fails load/transform, treat as record-level failure per policy.

## Testing Matrix (unit-level)
- bbox: 90° rotation around center yields correct AABB; flips invert x/y as expected.
- quad: identity → no change; 10° rotation keeps ordering and bounds; extreme scale within center maintains in-bounds after clamp.
- line: diagonal line rotated 45° remains valid and clipped correctly at borders; duplicates removed.
- Determinism: fixed seed reproduces identical parameters and outputs.

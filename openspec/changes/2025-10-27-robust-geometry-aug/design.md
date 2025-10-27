## Context
Augmentation geometry drift occurs when image warps and geometry transforms diverge (pivot, matrix, rounding, clipping). We introduce typed value objects and a single transform entrypoint to unify semantics and add debug alignment checks.

## Goals / Non-Goals
- Goals: Immutable geometry types; single transform path; exact alignment with image warp; debug tooling.
- Non-Goals: Changing training loss or template normalization.

## Decisions
- Value Objects: `BBox(x1,y1,x2,y2)`, `Quad([x0,y0,...,x3,y3])`, `Polyline([x0,y0,...])`; `@dataclass(frozen=True)`.
- Promotion: `BBox.apply_affine(M)` returns `BBox` for axis-aligned, `Quad` for general.
- Entrypoint: `transform_geometry(geom, M, width, height, policy={round='nearest', clip='polygon'})`.
- Pivot: Rotation about pixel-center ((W-1)/2,(H-1)/2). Image warp uses inverse-matrix in PIL; same M drives geometry (not Minv).
- Clipping: Sutherland–Hodgman for polygons; Cohen–Sutherland for polylines; round then clamp at end.
- Debug: Optional mask-warp IoU; artifacts saved on failure.
- Canvas expansion: Provide `ExpandToFitAffine` barrier that computes transformed-image quad under M, derives enclosing AABB in float, allocates a larger canvas, composes a translation so pixels and geometry align in expanded coords, then optional pad-to-32 (no crop).

## Alternatives
- Class-free functions only: simpler but weaker typing and easier divergence across call sites.
- Heavy OO with inheritance: rejected; prefer composition and value objects with pure functions.

## Risks / Trade-offs
- Slight overhead from object creation; mitigated by small counts and simple dataclasses.
- Additional debug cost only when enabled.

## Migration Plan
1) Implement types and entrypoint behind existing APIs.
2) Refactor Compose to call entrypoint; keep ops behaviors.
3) Enable debug in visualizer; fix issues flagged by IoU checks.

## Open Questions
- Do we add `Poly` beyond `Quad` as an output type? (Spec allows via policy.)
- Thresholds: default IoU 0.99 acceptable across datasets?



# Design: Unified object ordering policy

## Ordering levels

### Geometry-level (within a polygon)
Polygons are serialized in a canonical form:
- orientation: clockwise
- start vertex: top-left vertex (min y, then min x)

This canonicalization MUST be applied:
- after any affine + clipping path in augmentation (`src/datasets/geometry.py`)
- at builder serialization boundaries (defense-in-depth for legacy JSONLs)

### Geometry-level (within a line / polyline)
Lines/polylines are serialized in a canonical *direction*:
- Preserve the path order (no re-indexing of interior vertices).
- Only allow **full reversal** of the point sequence.
- Canonical start endpoint is chosen by comparing endpoints `(x0, y0)` vs `(xn, yn)` lexicographically by `(x, y)` (leftmost first, tie-break by topmost). If the end endpoint is smaller, reverse the sequence.

This makes line supervision deterministic under augmentation while respecting that polylines are not cyclic.

### Sequence-level (ordering objects within a record)
Objects are ordered TL→BR using their **center** (not first vertex):

1. Compute each object’s AABB:
   - bbox: `(x1, y1, x2, y2)`
   - poly/line: `min_x, min_y, max_x, max_y` over all points
2. Compute `center_x = (min_x + max_x) / 2`, `center_y = (min_y + max_y) / 2`
3. Sort by `(center_y, center_x)` ascending

#### Deterministic tie-breakers
Centers can tie (e.g., duplicated objects or symmetric shapes). To keep ordering deterministic and stable:
- Tertiary: geometry type (bbox < poly < line) or a stable string key
- Quaternary: legacy reference point key (top-left / first vertex / leftmost point), already canonicalized for poly
- Final: stable fallback (e.g., original index) only in strict legacy modes

Tie-breakers should be minimal and documented; tests MUST include at least one tie-case.

## Config wiring

### Proposed config surface (training)
Add an explicit knob (default `center_tlbr`) under `custom`:
- `custom.object_ordering_policy: reference_tlbr | center_tlbr`

The builder uses this knob when enumerating `object_n`.

### Proposed config surface (conversion)
Add an equivalent knob in `DataConversionConfig` / CLI:
- `--object_ordering_policy center_tlbr|reference_tlbr`

This is orthogonal to the existing:
- `--preserve_annotation_order` (skip any reorder for regression diffs)

## Validation hooks
Add a small validation helper used by both conversion and training:
- `validate_object_sequence_ordering(objects, policy) -> None` (raises with first inversion index + key dump)

This can be enabled in debug configs to catch drift early.

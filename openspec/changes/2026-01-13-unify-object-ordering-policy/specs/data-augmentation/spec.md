# data-augmentation Specification (Change Proposal)

## MODIFIED Requirements

### Requirement: TL→BR object ordering after augmentation is policy-driven (center default)
For dense-caption training, the system SHALL restore **top-to-bottom, left-to-right (TL→BR)** ordering of objects after augmentation and before building conversations.

The ordering policy MUST be selectable via a configuration knob shared between conversion and training-time builders:
- `center_tlbr` (**default**): geometry center ordering (AABB center)
- `reference_tlbr`: legacy reference-point ordering

When `reference_tlbr` is active:
- Primary key: reference Y coordinate ascending
- Secondary key: reference X coordinate ascending
- Reference point per geometry type:
  - `bbox_2d`: top-left corner `(x1, y1)`.
  - `poly`: first vertex `(x1, y1)` after canonicalization.
  - `line`: leftmost point among all vertices/points (smallest X; if tie, smallest Y).

When `center_tlbr` is active:
- Primary key: `center_y` ascending (objects higher in the image come first).
- Secondary key: `center_x` ascending (objects further left come first).
- The center is computed from the axis-aligned bounding box (AABB) of the geometry:
  - `bbox_2d`: AABB is `(x1, y1, x2, y2)`.
  - `poly` / `line`: AABB is computed over all vertices/points `(min_x, min_y, max_x, max_y)`.
- Deterministic tie-breakers MUST be applied to keep ordering stable under ties:
  - Tertiary: geometry type rank (`bbox_2d` < `poly` < `line`)
  - Quaternary: legacy reference point `(ref_y, ref_x)` computed as in `reference_tlbr`
  - Rely on stable-sort semantics only after the above explicit keys

The same TL→BR center-based sort rules MUST be used in:
- data conversion ordering, and
- training-time builders that enumerate `object_{n}`
so that object indices remain aligned across stages.

#### Scenario: Center-based TL→BR ordering after rotation
- **WHEN** a record is augmented with rotation, changing object positions and bbox extents
- **AND** the dense-caption builder prepares `object_{n}` entries
- **THEN** under `center_tlbr` the builder re-sorts objects using `(center_y, center_x)` before assigning indices
- **AND** the resulting `object_{n}` enumeration matches the conversion pipeline’s ordering under the same policy

### Requirement: Geometry canonicalization is applied after augmentation outputs
After augmentation has produced geometry outputs (including crop/clipping), the system MUST canonicalize:
- **Polygon vertex lists**:
  - orientation: clockwise
  - start vertex: top-left vertex (min y, then min x)
- **Line/polyline direction**:
  - preserve path order (no re-indexing of interior vertices)
  - only allow full reversal of the point sequence
  - start endpoint is chosen by comparing endpoints `(x0, y0)` and `(xn, yn)` lexicographically by `(x, y)`; reverse if needed

Canonicalization MUST be applied:
- in the augmentation wrapper output path (so downstream preprocessing/builders always see canonical geometry), and
- at builder serialization boundaries (defense-in-depth for legacy JSONLs and mixed provenance sources).

#### Scenario: Line direction is canonicalized after hflip
- **WHEN** a record containing a multi-point `line` object is augmented with horizontal flip
- **THEN** the output `line` MAY be reversed as a whole so that the canonical start endpoint is first
- **AND** interior vertex order remains unchanged (no rotation/re-indexing of the polyline)

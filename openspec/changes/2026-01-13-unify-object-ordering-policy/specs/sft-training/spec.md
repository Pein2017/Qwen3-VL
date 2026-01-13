# sft-training Specification (Change Proposal)

## MODIFIED Requirements

### Requirement: Training uses center-TLBR object ordering by default (legacy supported)
Dense-caption training SHALL enumerate assistant payload keys (`object_1`, `object_2`, ...) using a deterministic TL→BR ordering policy that matches conversion outputs.

The ordering policy MUST be selectable via `custom.object_ordering_policy`:
- `center_tlbr` (**default**): order objects by AABB center TL→BR
- `reference_tlbr`: legacy reference-point TL→BR

The system prompt text for dense mode MUST match the active policy so that "how objects are numbered" matches "how objects are described in labels".

#### Scenario: Center-based ordering is used for object_n enumeration
- **WHEN** training builds a dense-caption conversation from a record with multiple objects
- **THEN** under `center_tlbr` the builder sorts objects by `(center_y, center_x)` (with deterministic tie-breakers) before assigning `object_{n}`
- **AND** the resulting ordering matches the conversion pipeline’s ordering under the same policy.

### Requirement: Builder canonicalizes polygon and line serialization (defense-in-depth)
The training builder MUST canonicalize geometry serialization before rendering the assistant payload:
- **Polygon** vertex lists MUST be clockwise and start at the top-left vertex (min y, then min x).
- **Line/polyline** point sequences MUST preserve path order and may only be normalized by full reversal; the start endpoint is chosen by comparing endpoints `(x0, y0)` vs `(xn, yn)` lexicographically by `(x, y)`.

This canonicalization is required even when records come from mixed provenance sources (old JSONLs, external converters, or post-augmentation outputs).

#### Scenario: Builder fixes legacy line direction drift
- **GIVEN** a record whose `line` object has the same vertices but reversed direction
- **WHEN** building the dense-caption assistant JSON
- **THEN** the builder normalizes the line direction to the canonical endpoint-based start
- **AND** emits a single consistent serialization across runs.

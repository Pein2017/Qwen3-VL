# detection-preprocessor Specification (Change Proposal)

## MODIFIED Requirements

### Requirement: Converter object ordering is center-TLBR by default (legacy supported)
All detection converters (BBU/RRU + public sources) SHALL emit a deterministic object sequence order for `objects` within each record.

Ordering MUST be controlled by a single policy knob shared with training-time builders:
- `center_tlbr` (**default**): order objects by AABB center TL→BR
- `reference_tlbr`: legacy reference-point TL→BR

The repository-level conversion wrapper (`data_conversion/convert_dataset.sh`) SHALL default to `center_tlbr`, while still allowing explicit legacy exports by setting `OBJECT_ORDERING_POLICY=reference_tlbr` or passing `--object_ordering_policy reference_tlbr`.

#### Scenario: Center-based conversion ordering
- **WHEN** a dataset is converted via the conversion wrapper or CLI
- **AND** object ordering is enabled (annotation order is not preserved)
- **THEN** the converter outputs the `objects` list sorted by `(center_y, center_x)` under `center_tlbr`
- **AND** the same policy can be selected explicitly for legacy regression diffs.

### Requirement: Converter canonicalizes polygon and line serialization
Converters SHALL canonicalize geometry serialization to reduce label variance and keep training prompts consistent with targets:
- **Polygon** vertex lists MUST be clockwise and start at the top-left vertex (min y, then min x).
- **Line/polyline** point sequences MUST preserve path order and may only be normalized by full reversal; the start endpoint is chosen by comparing endpoints `(x0, y0)` vs `(xn, yn)` lexicographically by `(x, y)`.

#### Scenario: Line direction canonicalization
- **WHEN** a converter emits a multi-point `line` object
- **THEN** the converter MAY reverse the entire point sequence so the canonical start endpoint is first
- **AND** it MUST NOT rotate/re-index interior vertices or otherwise change the path structure.

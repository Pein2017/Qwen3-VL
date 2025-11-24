## ADDED Requirements

### Requirement: Single-image small-object zoom-and-paste augmentation
The augmentation system SHALL provide a registered operator that enlarges and repositions small objects within the same image while preserving geometry correctness and avoiding excessive overlap with existing annotations.

#### Scenario: Select and paste small objects
- **WHEN** the operator runs with probability `p`
- **AND** targets objects whose bbox/line length is below configurable thresholds (optionally filtered by class whitelist)
- **AND** crops a patch with optional context margin, scales it within a configured range, and translates it to an in-bounds location
- **THEN** the patched image is returned with the chosen objects duplicated at the new location, and the original objects remain unchanged

#### Scenario: Geometry synchronization across types
- **WHEN** a small object is pasted
- **THEN** its geometry (bbox, poly, or line) is transformed by the same scale+translate affine, clipped/clamped to image bounds, and kept in pixel space with a single geometry field
- **AND** degenerate results (<2 points for line, <3 points for poly) are dropped for that pasted instance while keeping the source instance

#### Scenario: Overlap and safety gating
- **WHEN** a candidate paste location would overlap existing annotated objects beyond a configurable IoU/coverage threshold (line treated as a buffered polygon)
- **THEN** the candidate is rejected and another placement is sampled up to a configured attempt limit
- **AND** if no valid placement is found, the operator skips that target without altering the record

#### Scenario: Bounds and canvas invariants
- **WHEN** the operator finishes
- **THEN** the canvas size (`width`, `height`) is unchanged, coordinates remain within `[0..W-1]×[0..H-1]`, and the op remains compatible with downstream padding/alignment (e.g., `expand_to_fit_affine`)

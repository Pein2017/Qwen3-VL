# geometry-eval-metrics Specification

## Purpose
TBD - created by archiving change 2026-01-05-add-geometry-eval-metrics. Update Purpose after archive.
## Requirements
### Requirement: Dump ingestion (norm1000)
The evaluator SHALL consume `gt_vs_pred.jsonl` dumps where GT and predictions are already expressed in `norm1000` coordinates.

#### Scenario: Accept norm1000 gt_vs_pred dumps
- **WHEN** a JSONL record provides `gt_norm1000` and `pred` (or `pred_norm1000`) as lists of `{type, points, desc}`
- **THEN** the evaluator reads both lists as norm1000 geometry objects without requiring access to the original images
- **AND** records with no GT objects and no predicted objects are skipped deterministically (do not affect aggregates)

### Requirement: Label extraction (phase + fine category)
The evaluator SHALL extract two label granularities from `desc` to support category-aware metrics, tolerating both legacy slash-delimited and key=value formats.

#### Scenario: Parse both desc formats
- **WHEN** an object desc contains key=value fields with `类别=...`
- **THEN** the evaluator uses the `类别` value as both the phase/head label and the fine category label
- **WHEN** an object desc is in legacy slash-delimited form (e.g., `phase/...`)
- **THEN** the evaluator uses the prefix before the first `/` as the phase/head label
- **AND** derives a fine category label using the project’s domain mapping rules for umbrella phases (e.g., “螺丝、光纤插头” → “BBU安装螺丝/ODF端光纤插头/...”)

### Requirement: Region overlap ruler (bbox/poly cross-type)
The evaluator SHALL treat `bbox_2d` and convex-quad `poly` as filled regions and compute overlap using area intersection over union, allowing `bbox_2d` ↔ `poly` cross-type comparisons.

#### Scenario: Cross-type region matching (bbox_2d ↔ poly)
- **GIVEN** a GT region encoded as `poly` and a prediction encoded as `bbox_2d` (or the reverse)
- **WHEN** overlap is computed
- **THEN** the evaluator computes filled-shape IoU (area intersection divided by area union)
- **AND** the result is used for matching decisions exactly the same way as same-type region IoU

### Requirement: Line overlap ruler (TubeIoU)
The evaluator SHALL evaluate `line` objects using a mask-wise TubeIoU defined by a configurable buffer tolerance in norm1000 units.

#### Scenario: TubeIoU for polylines
- **GIVEN** a GT polyline and a predicted polyline
- **WHEN** line overlap is computed with tolerance `tol`
- **THEN** each polyline is converted to a tube mask by rasterizing a stroke of width `round(2 * tol)` on the norm1000 grid
- **AND** the overlap score is computed as mask IoU: `|A ∩ B| / |A ∪ B|`

#### Scenario: Tolerance is configurable and reported
- **WHEN** evaluation is executed with a specified TubeIoU tolerance value
- **THEN** that tolerance is used for all line overlap computations
- **AND** the tolerance value is included in the machine-readable evaluation artifact for reproducibility
- **WHEN** evaluation is executed without explicitly specifying a TubeIoU tolerance value
- **THEN** the evaluator selects a default tolerance of `8` (norm1000 units)
- **AND** the selected default tolerance is included in the machine-readable evaluation artifact for reproducibility

### Requirement: Greedy 1-to-1 matching (intersection-over-union)
The evaluator SHALL match GT objects to predictions using greedy 1‑to‑1 assignment based on the overlap score, with deterministic tie-breaking.

#### Scenario: Geometry family compatibility (region vs line)
- **WHEN** the evaluator determines whether two objects are eligible to be compared for overlap
- **THEN** it treats `bbox_2d` and `poly` as members of a single **region family**
- **AND** it treats `line` as the only member of a **line family**
- **AND** region-family objects (`bbox_2d`/`poly`) MAY be compared cross-type using the region overlap ruler
- **AND** line-family objects (`line`) MAY only be compared to other line-family objects using TubeIoU
- **AND** region-family and line-family objects SHALL NOT be compared or matched to each other

#### Scenario: Candidate generation by overlap and label constraints
- **GIVEN** a chosen evaluation mode (localization-only, phase-aware, or category-aware) and an overlap threshold `t`
- **WHEN** candidate pairs are generated for matching
- **THEN** a GT object and a predicted object are eligible to form a candidate pair only when:
  - both objects belong to compatible geometry families
  - the mode’s label constraint is satisfied (if enabled)
  - the overlap score is `>= t`

#### Scenario: Greedy assignment and deterministic tie-breaking
- **GIVEN** a set of candidate pairs `(score, gt_index, pred_index)` for a fixed threshold `t`
- **WHEN** 1‑to‑1 matching is computed
- **THEN** candidate pairs are sorted by:
  - `score` descending
  - `gt_index` ascending
  - `pred_index` ascending
- **AND** pairs are selected greedily in that order, skipping any pair where either index is already matched
- **AND** the evaluator ignores any prediction confidence scores (matching is geometry/label driven only)

### Requirement: Matching modes and metrics
The evaluator SHALL support localization-only and category-aware evaluation modes and report COCO-like threshold sweeps.

#### Scenario: Produce localization and category-aware metrics
- **WHEN** evaluation runs on a dump file
- **THEN** it reports metrics under each mode:
  - localization-only (no desc constraint)
  - phase-aware (same phase label required)
  - category-aware (same fine category required)
- **AND** for each mode it reports Precision/Recall/F1 at IoU thresholds 0.50:0.95 step 0.05
- **AND** reports a mean-F1 across the threshold sweep
- **AND** includes per-geometry breakdowns for `bbox_2d`, `poly`, and `line`

### Requirement: Required outputs and parameter reporting
The evaluator SHALL emit a human-readable console summary and a machine-readable artifact that records both metrics and the parameters used to compute them.

#### Scenario: Primary threshold is configurable and reported
- **WHEN** evaluation is executed with an explicitly specified primary threshold value
- **THEN** the evaluator uses that value as the primary threshold for the console summary
- **AND** the primary threshold value is included in the machine-readable evaluation artifact for reproducibility
- **WHEN** evaluation is executed without explicitly specifying a primary threshold value
- **THEN** the evaluator selects a default primary threshold of `0.50`
- **AND** the selected default primary threshold is included in the machine-readable evaluation artifact for reproducibility

#### Scenario: Emit console summary and JSON artifact
- **WHEN** evaluation runs on a dump file
- **THEN** the evaluator prints a console summary that includes, at minimum:
  - the dump path (or identifier)
  - the number of evaluated images/records
  - aggregate Precision/Recall/F1 for a primary threshold (including the primary threshold value used)
  - mean-F1 over the COCO-like threshold sweep
- **AND** the evaluator produces a machine-readable JSON artifact containing:
  - all reported metrics (including per-geometry breakdowns)
  - the full threshold sweep list used
  - the primary threshold value used for console reporting
  - the TubeIoU tolerance value used for lines
  - the matching algorithm identifier (greedy 1-to-1) and tie-break order
  - the set of evaluation modes executed (localization-only, phase-aware, category-aware)

### Requirement: Deterministic outputs
Evaluation outputs SHALL be deterministic for a fixed input dump and fixed evaluation parameters.

#### Scenario: Deterministic matching and aggregation
- **WHEN** the same dump is evaluated multiple times with identical parameters
- **THEN** the console summary and the machine-readable artifact are bitwise identical
- **AND** matching tie-breaking is deterministic (no randomness, stable sorting)


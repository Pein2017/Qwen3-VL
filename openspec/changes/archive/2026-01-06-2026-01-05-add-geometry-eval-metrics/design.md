# Design: Geometry- and category-aware evaluation for dense captioning

## 1) Inputs

### 1.1 Dump format
Evaluation consumes `gt_vs_pred.jsonl` produced by visualization/inference tooling (norm1000):
- `gt_norm1000`: list of objects `{type, points, desc}`
- `pred` (or `pred_norm1000`): list of objects `{type, points, desc}`

Coordinates are integers in `[0, 1000]` (norm1000). Polygons are convex quads. Lines are polylines with ≥ 2 points.

### 1.2 Desc formats (two variants)
Evaluation must tolerate both:
- **Key=value**: comma-separated `key=value` pairs, `类别` first.
- **Legacy slash levels**: `phase/...` with `,` for same-level fields and `/` for hierarchical levels (see `data_conversion/hierarchical_attribute_mapping.json` separator rules).

## 2) Label normalization (two granularities)

Two label views are computed per object and reported in metrics:

### 2.1 Phase/head label (coarse)
- If `desc` contains `类别=...`, phase label is that value.
- Else, phase label is the substring before the first `/`.

### 2.2 Fine category label (task category)
- Preferred: key=value `类别`.
- Legacy: derived from the hierarchical mapping where “type” values define the fine category for multi-class phases (e.g., “螺丝、光纤插头” → {BBU安装螺丝, ODF端光纤插头, ...}).

This allows both “phase-aware” and “category-aware” evaluation. Both are needed because some legacy phases are umbrella groups.

## 3) Geometry overlap rulers

The evaluation uses **area intersection** as the primary ruler for region-type objects and a **tube area** definition for line-type objects.

### 3.1 Geometry families
- **Region family**: `bbox_2d` and `poly` (convex quad). Cross-type matching is allowed.
- **Line family**: `line` only (matches line-to-line only).

### 3.2 Region IoU (bbox/poly cross-type)
Define filled-shape IoU:
```
IoU = Area(A ∩ B) / Area(A ∪ B)
```
Where:
- `bbox_2d` is a filled axis-aligned rectangle.
- `poly` is a filled convex polygon (quad).

Because polygons are convex quads, intersection can be computed deterministically (convex clipping), or via deterministic rasterization on the norm1000 grid. The key requirement is that the ruler is area IoU over filled regions, not edge distance.

### 3.3 Line TubeIoU (mask-wise, fixed tolerance)
Lines are 1D; area IoU requires a thickness definition.

Define a fixed tolerance `tol` in norm1000 units. Each polyline is converted into a **tube mask** by drawing the line with stroke width:
```
stroke_width = round(2 * tol)
```
Then:
```
TubeIoU(tol) = Area(mask_gt ∩ mask_pred) / Area(mask_gt ∪ mask_pred)
```

Implementation detail requirements (for determinism + performance):
- Rasterization happens on a deterministic grid aligned with norm1000.
- For performance, rasterize inside a tight window computed from the buffered AABBs of both geometries (similar to the approach used by `_geom_mask_iou` in augmentation code paths).

### 3.4 Optional diagnostic: coverage-F1 for lines
Coverage-F1 is useful to interpret “missing vs hallucinated” line cases:
- Recall-like: fraction of GT samples within distance `tol` of prediction segments
- Precision-like: fraction of prediction samples within distance `tol` of GT segments
- Harmonic mean yields an interpretable 0..1 score

This diagnostic is not the primary ruler (area IoU is), but can be emitted for debugging.

## 4) Matching

### 4.1 Candidate generation
For each GT object and prediction object:
- Require compatible geometry family:
  - Region family: allow bbox↔poly cross-type
  - Line family: line↔line only
- Compute overlap score:
  - Region: RegionIoU
  - Line: TubeIoU(tol)

### 4.2 1-to-1 assignment
Greedy assignment by descending overlap score:
- For a chosen threshold `t`, a pair is a candidate if `score >= t`.
- Sort candidates by score desc; assign if both GT and Pred indices are unused.
- Tie-breaking must be deterministic (e.g., stable sort by `(score desc, gt_index asc, pred_index asc)`).

### 4.3 Matching modes
The same matching procedure is executed under three constraints:
1. **Localization-only**: no desc constraint.
2. **Phase-aware**: require same phase/head label.
3. **Category-aware**: require same fine category label.

## 5) Metrics

### 5.1 Threshold sweep
Compute metrics at IoU thresholds:
```
T = {0.50, 0.55, ..., 0.95}
```
Report per `t`:
- `precision(t) = matched(t) / pred_total`
- `recall(t) = matched(t) / gt_total`
- `f1(t) = 2PR / (P+R)` (0 when P+R=0)

Also report:
- `mF1 = mean_t f1(t)` over the sweep
- `mean_overlap_matched@0.50` (tightness on matched pairs)

### 5.2 Breakdowns
Metrics are reported:
- Overall
- By geometry type (`bbox_2d`, `poly`, `line`)
- By label granularity (phase-aware vs category-aware)
- By category (top categories by GT frequency)

### 5.3 Count diagnostics
Report per-image count diagnostics (aggregate as means/rates):
- Count MAE: `mean(|pred_count - gt_count|)`
- Over-generation rate: fraction of images with `pred_count > gt_count`
- Under-generation rate: fraction of images with `pred_count < gt_count`

### 5.4 Optional attribute/OCR correctness (matched-pair)
On matched pairs at a primary threshold (default 0.50), compute:
- key/value micro-F1 excluding free-text keys by default
- OCR/text fields:
  - label `文本`: normalized string similarity (exact match and/or edit similarity)
  - station distance: numeric exact match and MAE

## 6) Outputs
- Console summary: overall + per-type + per-mode metrics.
- JSON artifact: all metrics + parameters + metadata (for regression gating).
- Optional wrong-case JSONL: store per-image misses/FPs for visualization follow-up.


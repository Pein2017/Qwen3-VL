# Change: Dense Augmentation Telemetry & Safeguards

## Context
- Follow-up to `2025-10-27-robust-geometry-aug` to address two regressions identified in `compare/data-augmentation-potentails/final_analysis.md`.
- Observed issues: canvas pixel-cap overshoot after 32× alignment and over-permissive coverage heuristic for elongated/rotated quads.

## Implementation
- Enforced pixel-cap limits post-alignment in `ExpandToFitAffine`, reusing proportional scaling and emitting accurate warnings.
- Introduced polygon-aware coverage via `compute_polygon_coverage` (Sutherland–Hodgman clipping + shoelace area) and wired `RandomCrop` to use it.
- Added structured completeness updates and crop skip telemetry (`last_skip_counters`, padding ratios) surfaced through `augmentation.telemetry`.
- Prevented redundant resampling in `Compose` when affine matrix is identity; ensured metadata propagation of final canvas size.
- Documentation updates in `docs/AUGMENTATION.md`; regression tests in `tests/augmentation/test_crop_coverage.py` and `tests/test_augmentation_geometry.py` cover the new logic.

## Validation
- `conda run -n ms pytest tests/augmentation/test_crop_coverage.py tests/test_augmentation_geometry.py`
  - Result: `24 passed, 3 skipped`

## Follow-ups
- Monitor telemetry during the next dense-caption smoke run to confirm skip distributions and padding ratios.
- Consider extending coverage telemetry to offline analysis (`vis_tools/vis_augment_compare.py`).


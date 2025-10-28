# Proposal: Dense Augmentation Telemetry & Safety

## Problem Statement
Dense-caption training suffered quality regressions traced to augmentation bugs:

1. `ExpandToFitAffine` computed scale factors before rounding canvas dimensions to 32× multiples, leading to overshoot beyond `max_pixels` and excessive neutral padding.
2. `RandomCrop` reused `compute_coverage` (AABB overlap). For elongated/rotated quads this overestimated visibility, retaining nearly invisible objects and degrading supervision.

Secondary gaps included lack of crop telemetry (skip reasons, padding ratios) and reliance on string-only completeness updates.

## Goals
- Guarantee canvas expansion adheres to `max_pixels` even after alignment.
- Replace AABB coverage heuristics with polygon-aware measurement for quads.
- Emit crop telemetry (coverage histograms, skip counters, padding ratios) to support tuning.
- Update structured completeness metadata when available.

## Non-Goals
- No new augmentation operators or config knobs.
- No change to color augmentation or bypass probabilities.

## Proposed Changes
1. **Pixel Cap Enforcement**
   - Adjust `ExpandToFitAffine.pre_flush_hook` to align dimensions first, scale using floor-to-multiple logic, recompute affine scaling, and expose `padding_ratio` telemetry.
2. **Polygon Coverage**
   - Add `compute_polygon_coverage` with Sutherland–Hodgman clipping + shoelace area; wire `RandomCrop` to use it and record `last_object_coverages`.
3. **Completeness Metadata**
   - Extend `AugmentationPreprocessor` to update structured fields (e.g., `attributes.completeness`) besides `desc` replacements.
4. **Telemetry Surface**
   - Track skip reasons (`min_objects`, `line_object`), padding ratios, and final canvas sizes via `Compose` & `apply_augmentations` logging (`augmentation.telemetry`).
5. **Tests & Docs**
   - Add regression tests for polygon coverage and pixel caps.
   - Update `docs/AUGMENTATION.md` to describe telemetry and new safety behavior.

## Validation Plan
- `conda run -n ms pytest tests/augmentation/test_crop_coverage.py tests/test_augmentation_geometry.py`
- Visual spot-check using `vis_tools/vis_augment_compare.py` (optional).
- Review telemetry logs during a 5k-step dense-caption smoke run (follow-up item).


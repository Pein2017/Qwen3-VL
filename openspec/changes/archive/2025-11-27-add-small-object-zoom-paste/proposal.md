# Proposal: Add single-image small-object zoom-and-paste augmentation

## Context
- Current augmentation stack (see `docs/DATA_AUGMENTATION.md`) covers global affines, crops, resize, and color jitter with geometry preservation, but lacks a targeted small-object recall booster.
- Data review (`data/bbu_full_768_poly/all_samples.jsonl`) shows many tiny screws/optical connectors and cable endpoints that are easily missed.
- Existing spec (`data-augmentation`) requires geometry-safe ops but has no requirement for selective small-object duplication/enlargement within the same image.

## Problem
- Missed detections on small objects (screws, fiber connectors) due to low pixel footprint and sparse training exposure.
- Need an augmentation that increases effective object size/occurrence without cross-image copy artifacts.

## Proposal
- Add a **single-image small-object zoom-and-paste** operator:
  - Select small objects by size/length thresholds.
  - Crop patch with light context, scale up (e.g., 1.4–1.8×), and translate within the same image.
  - Reject placements overlapping existing annotations beyond a small IoU/coverage threshold; clamp to bounds; skip if no safe slot after limited attempts.
  - Update geometries (bbox/poly/line) with the same affine; keep `desc` unchanged.
  - Integrate as a registered op in `src/datasets/augmentation`, respecting the affine flush/barrier model and pixel-cap/32× alignment flow.
- Document config knobs and safety rules in `docs/DATA_AUGMENTATION.md`.

## Impact / Scope
- **In scope:** New op registration, geometry-safe transform, overlap checks, YAML configurability, docs/spec update, basic tests/visualization guidance.
- **Out of scope:** Cross-image copy-paste, mixup, synthetic background generation, new training configs.

## Risks / Mitigations
- Over-augmentation or artifacts → use low prob, overlap gating, skip-on-fail.
- Geometry drift → reuse existing affine/geometry helpers and bounding checks.

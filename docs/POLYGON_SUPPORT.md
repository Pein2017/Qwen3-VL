# Polygon Support

Qwen3-VL standardizes on three geometry primitives: `bbox_2d`, `poly`, and `line`. The training pipeline and data converters no longer emit the legacy `quad` field. This document explains the expected behavior and the `poly_fallback` tuning knob.

## Canonical Geometry

- **`bbox_2d`**: Axis-aligned boxes `[x1, y1, x2, y2]` useful for simple detection cases.
- **`poly`**: Flat `[x1, y1, x2, y2, ..., xn, yn]` representing closed polygons (N ≥ 3). Use `poly_points` to make the vertex count explicit (`len(poly) == 2 * poly_points`).
- **`line`**: Open polylines (e.g., cables or fibers) with `line_points` tracking point count.

Each object must include exactly one of these fields plus a non-empty `desc`. Conversion scripts (LVIS, COCO, etc.) now map polygons directly to `poly` and emit `poly_points` for downstream validation. If `poly` is not available or you prefer rectangle supervision, set `poly_fallback: bbox_2d` in your fusion config to automatically convert polygons to their axis-aligned envelopes before augmentation. You can also keep polygons but bound their complexity with `poly_max_points: <int>`.

## Dynamic Fallback

Propagation happens during dataset loading:

1. The JSONL retains its original `poly` shapes for auditing, debugging, and offline tooling.
2. When `Custom.fusion_config` (or the offline `scripts/fuse_datasets.py`) sets `poly_fallback: bbox_2d` for a source, the loader converts **all** polygons to bounding boxes in-memory before augmentation.
3. When `poly_max_points` is set (e.g., `poly_max_points: 12`), the loader keeps polygons whose vertex count ≤ the threshold and downgrades only the oversized ones to `bbox_2d`. This is useful for datasets such as LVIS whose masks can exceed a dozen points.
4. Both knobs work per source and can be toggled without rewriting JSONLs, letting you balance polygon supervision vs. simpler boxes per experiment.

When fallback is disabled, polygons survive through augmentation and are normalized by the prompts (`norm1000`), giving the model rich geometric cues (especially useful for rotated BBU data). Auxiliary datasets often prefer `poly_fallback` to maintain compatibility with the base BBU loss while still correcting for large/non-convex polygons.

## Workflow Notes

- Public dataset scripts now produce `poly` (with `poly_points`) whenever polygon mode is enabled; `bbox_2d` remains available when you disable polygons or when fallback converts shapes.
- Dataset wrappers carry domain information (target vs source) so `MultiSourceFusionDataset` knows whether to attach augmentation/curriculum. Template selection is handled at the wrapper level, so JSONL records themselves no longer require provenance metadata.
- Source-domain wrappers default to clean images; target-domain wrappers automatically attach the configured augmentation pipeline. Override per dataset via `params.augmentation_enabled`/`params.curriculum_enabled` if needed.

For a deeper look, see `docs/DATA_AND_DATASETS.md` (multi-dataset fusion section) and `openspec/changes/update-geometry-poly-fusion/design.md`.

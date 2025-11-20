# Polygon Support

Qwen3-VL standardizes on three geometry primitives: `bbox_2d`, `poly`, and `line`. The training pipeline and data converters no longer emit the legacy `quad` field. This document explains the expected behavior and the `poly_fallback` tuning knob.

## Canonical Geometry

- **`bbox_2d`**: Axis-aligned boxes `[x1, y1, x2, y2]` useful for simple detection cases.
- **`poly`**: Flat `[x1, y1, x2, y2, ..., xn, yn]` representing closed polygons (N ≥ 3). Use `poly_points` to make the vertex count explicit (`len(poly) == 2 * poly_points`).
- **`line`**: Open polylines (e.g., cables or fibers) with `line_points` tracking point count.

Each object must include exactly one of these fields plus a non-empty `desc`. Conversion scripts (LVIS, COCO, etc.) now map polygons directly to `poly` and emit `poly_points` for downstream validation. If `poly` is not available or you prefer rectangle supervision, set `poly_fallback: bbox_2d` in your fusion config to automatically convert polygons to their axis-aligned envelopes before augmentation.

## Dynamic Fallback

Propagation happens during dataset loading:

1. The JSONL retains its original `poly` shapes for auditing, debugging, and offline tooling.
2. When `Custom.fusion_config` (or the offline `scripts/fuse_datasets.py`) sets `poly_fallback: bbox_2d` for a source, the loader computes the bounding box of each polygon using `points_to_xyxy()` and discards the `poly` field before augmentation or builder logic runs.
3. This fallback works per source and can be toggled without rewriting JSONLs, letting you balance polygon supervision vs. simpler boxes per experiment.

When fallback is disabled, polygons survive through augmentation and are normalized by the prompts (`norm1000`), giving the model rich geometric cues (especially useful for rotated BBU data). Auxiliary datasets often prefer `poly_fallback` to maintain compatibility with the base BBU loss while still correcting for large/non-convex polygons.

## Workflow Notes

- Public dataset scripts now produce `poly` (with `poly_points`) whenever polygon mode is enabled; `bbox_2d` remains available when you disable polygons or when fallback converts shapes.
- `metadata.dataset` and `metadata.template` on each record let the augmentation preprocessor gate transforms and help per-source template selection.
- Use `custom.augment_sources` to specify which dataset names (e.g., `["bbu"]`) should undergo geometry-aware augmentation; auxiliary sources can stay clean even though they flow through the same pipeline.

For a deeper look, see `docs/DATA_AND_DATASETS.md` (multi-dataset fusion section) and `openspec/changes/update-geometry-poly-fusion/design.md`.

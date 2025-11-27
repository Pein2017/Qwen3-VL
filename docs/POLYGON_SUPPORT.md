# Polygon Support

Qwen3-VL standardizes on three geometry primitives: `bbox_2d`, `poly`, and `line`. The training pipeline and data converters no longer emit the legacy `quad` field. This document explains the expected behavior and how polygons are handled offline.

## Canonical Geometry

- **`bbox_2d`**: Axis-aligned boxes `[x1, y1, x2, y2]` useful for simple detection cases.
- **`poly`**: Flat `[x1, y1, x2, y2, ..., xn, yn]` representing closed polygons (N ≥ 3). Use `poly_points` to make the vertex count explicit (`len(poly) == 2 * poly_points`).
- **`line`**: Open polylines (e.g., cables or fibers) with `line_points` tracking point count.

Each object must include exactly one of these fields plus a non-empty `desc`. Conversion scripts (LVIS, COCO, etc.) now map polygons directly to `poly` and emit `poly_points` for downstream validation. If you need rectangles instead of polygons, perform the downgrade during conversion (e.g., cap vertices with `--poly-max-points`); the dataloader no longer mutates geometry.

### Vertex Ordering (all domains)

- Offline converters and validators canonicalize polygon vertices to avoid self-crossing: remove duplicated closing points, sort vertices **clockwise around the centroid**, then rotate so the **top-most (then left-most) vertex is first**.
- Lines are left as-is (no cyclic reorder). Boxes are unaffected.
- Visualization tools in `vis_tools/` apply the same canonicalization before drawing, so converter output and on-the-fly views are consistent.

## Geometry handling

Propagation happens during conversion (see `docs/DATA_PREPROCESSING_PIPELINE.md` for the BBU/RRU converter and public converters under `public_data/`):

1. The JSONL retains the geometry produced by offline converters; loaders do not alter `poly`/`bbox_2d` at runtime.
2. To cap polygon complexity, use the conversion scripts (e.g., `public_data/scripts/convert_lvis.py --use-polygon --poly-max-points 12 ...`) so oversized polygons are downgraded before training.
3. If you need all polygons flattened to boxes, add that logic to the converter—runtime fallback has been removed to keep dataloader work minimal.

With runtime fallback removed, polygons survive through augmentation and are normalized by the prompts (`norm1000`), giving the model rich geometric cues (especially useful for rotated BBU data).

## Workflow Notes

- Public dataset scripts now produce `poly` (with `poly_points`) whenever polygon mode is enabled; `bbox_2d` remains available when you disable polygons or when fallback converts shapes.
- Dataset wrappers carry domain information (target vs source) so the fusion loader knows whether to attach augmentation/curriculum. Template selection is handled at the wrapper level, so JSONL records themselves no longer require provenance metadata.
- Source-domain wrappers default to clean images; target-domain wrappers automatically attach the configured augmentation pipeline. Override per dataset via `params.augmentation_enabled`/`params.curriculum_enabled` if needed.

For a deeper look, see `docs/DATA_AND_DATASETS.md` (multi-dataset fusion section) and `openspec/changes/update-geometry-poly-fusion/design.md`.

# Qwen2.5-VL Data Conversion Pipeline

> **Unified Processing Architecture – August 2025**
>
> Streamlined data conversion pipeline using `unified_processor.py` for BBU equipment annotation processing.
> Focuses on rigorous geometry handling, hierarchical Chinese descriptions, and deterministic coordinate
> transformations.

---

## Table of Contents
1. [Overview](#overview)
2. [Inputs](#inputs)
3. [What the Pipeline Does](#what-the-pipeline-does)
4. [Geometry & Coordinate Rules](#geometry--coordinate-rules)
5. [Outputs](#outputs)
6. [Raw Input Example (kept as reference)](#raw-input-example-kept-as-reference)
7. [Output Sample](#output-sample)

---

## Overview

The unified pipeline converts V2 annotations + images into training-ready JSONL with native geometry and
hierarchical descriptions. The processing emphasizes exact coordinate transformations and canonical geometry
ordering for stable learning signals.

- Core modules:
  - `unified_processor.py`: Orchestrates the end-to-end pipeline
  - `coordinate_manager.py`: EXIF orientation → dimension rescaling → smart resize (in that order)
  - `flexible_taxonomy_processor.py`: V2 annotation parsing and hierarchical description generation
  - `validation_manager.py`: Object/sample validation and reporting
  - `vision_process.py`: Smart resize with factor alignment and pixel budget

---

## Inputs

- Dataset directory with paired raw annotation JSON and images (V2 format)
- The pipeline accepts two raw schema variants:
  - `dataList`: rectangle selection stored as 2 points `[x1, y1], [x2, y2]`
  - `markResult.features`: native geometry per feature (e.g., `LineString`, `Quad`, `Polygon`)
- Required metadata: `info.width`, `info.height` in each JSON
- Supported object types (strict): `{bbu, bbu_shield, connect_point, label, fiber, wire}`
  - Geometry constraints:
    - `fiber`, `wire` → line geometries
    - `bbu`, `bbu_shield`, `connect_point`, `label` → quad or bbox geometries
- Optional fixed vocabulary files for teacher pool building (auto-detected):
  - `data_conversion/attribute_taxonomy.json`
  - `data_conversion/hierarchical_attribute_mapping.json`
  - If present, they define a fixed universe of canonical tokens for coverage; otherwise, a simple free-vocabulary fallback is used automatically.

---

## What the Pipeline Does

- Object extraction (V2):
  - Reads `dataList` and/or `markResult.features`
  - Determines `object_type` using Chinese keys and taxonomy mapping
  - Filters to the supported set (unknown types are dropped)

- Hierarchical description construction:
  - Uses exact Chinese keys and a strict mapping
  - Separator rules: comma (`,`) for same-level attributes, slash (`/`) for levels/conditionals

- Geometry normalization and canonicalization:
  - Converts V2 geometries to native formats: `bbox_2d`, `quad`, `line`
  - Applies canonical vertex/point ordering (see next section)
  - Clamps coordinates to image bounds and fixes degeneracies

- Coordinate transformation pipeline (strict order):
  1) Apply EXIF orientation to geometry
  2) Rescale if JSON `info.{width,height}` differ from actual image dimensions
  3) Smart-resize scaling to target dimensions (multiple-of-28 with pixel budget)

- Validation and filtering:
  - Enforces geometry constraints and bounds
  - Minimum object size checks (square/bbox)
  - Requires non-empty `desc`
  - Strips occlusion tokens containing “遮挡” by default (configurable). This property is deprecated and shown to be unhelpful for training, so occlusion words like “有遮挡/无遮挡/挡风板有遮挡” are removed from `desc` during conversion and are excluded from teacher‑pool coverage and selection entirely.
  - Standardizes label descriptions (configurable): any `标签/*` with empty-like or non-informative content (e.g., `空格`, `看不清`, `、`, or missing) is normalized to `标签/无法识别`.
  - Records invalid objects/samples for reporting

- Object ordering and image processing:
  - Sorts objects by top-to-bottom, then left-to-right using their first coordinate pair
  - Processes/copies the image to match final transform (EXIF and smart resize)

- Deterministic splitting and exports:
  - Selects a teacher pool using fixed rules (see below), then splits remaining into train/val
  - Writes flat-format JSONL files and summary artifacts

### Teacher Pool Builder (Rule‑Based)
- Fixed vocabulary mode (default):
  - Builds a coverage universe from `attribute_taxonomy.json` and `hierarchical_attribute_mapping.json` (excludes any `free_text` fields). Occlusion tokens containing “遮挡” are explicitly excluded from the universe and do not affect coverage scores.
  - Greedy set‑cover picks samples to maximize token coverage up to `MAX_TEACHERS` with deterministic tie‑breakers: prefer `line` only if fiber/wire tokens remain → brand balancing → geometry novelty → object_count closest to median → lexicographic by image path.
  - Respects `OBJECT_TYPES` (tokens tied exclusively to filtered‑out types are ignored).
- Free vocabulary fallback:
  - Automatically used only if the fixed files are missing. Builds top‑K frequent tokens per object type and geometry and applies the same greedy selection.
- Emits `teacher_pool.jsonl` and `teacher_pool_stats.json` with coverage metrics (mode, universe size, covered/uncovered units, brand distribution, geometry presence, object_count summary).

---

## Geometry & Coordinate Rules

- BBox (`bbox_2d`): `[x_min, y_min, x_max, y_max]` with `x_min < x_max`, `y_min < y_max`
- Quad (`quad`): 8 integers representing 4 vertices ordered canonically
  - Canonical ordering: start at top-left, then proceed clockwise
- Line (`line`): `[x1, y1, x2, y2, ...]`
  - 2-point lines: ordered lexicographically by `(x, then y)`
  - Multi-point lines: preserve path structure; choose a canonical direction so the first point is the
    topmost-leftmost among endpoints; reverse the whole sequence if needed
- All coordinates are clamped within final image bounds and rounded to integers after transforms

---

## Outputs

Directory for each processed dataset (example shown as `data/{dataset_name}/`):

```
train.jsonl            # Training samples (flat)
val.jsonl              # Validation samples (flat)
teacher_pool.jsonl     # Teacher pool (flat)
teacher_pool_stats.json# Teacher pool coverage statistics
all_samples.jsonl      # Combined flat samples (teacher + train + val)
label_vocabulary.json  # Aggregated labels and statistics
validation_report.json # Summary + counts of invalid objects/samples
invalid_objects.jsonl  # Per-object validation failures (detailed)
invalid_samples.jsonl  # Per-sample skip reasons
images/                # Processed images (EXIF-corrected, smart-resized)
```

Key notes:
- All JSONL files use the same flat format: `{ "images": [..], "objects": [..], "width": W, "height": H }`
- Objects preserve their native geometry key: one of `bbox_2d`, `quad`, `line`, plus a hierarchical `desc`
- `label_vocabulary.json` aggregates object types, properties, and full descriptions for training aids

---

## Raw Input Example (kept as reference)

Minimal examples of the two accepted raw schemas.

- dataList (rectangle → bbox):
```json
{
  "info": {"width": 728, "height": 532},
  "dataList": [
    {
      "coordinates": [[264, 144], [326, 201]],
      "properties": {
        "contentZh": {
          "标签": "螺丝、光纤插头/BBU安装螺丝,显示完整,符合要求"
        }
      }
    }
  ]
}
```

- markResult.features (native geometry):
```json
{
  "info": {"width": 728, "height": 532},
  "markResult": {
    "type": "FeatureCollection",
    "features": [
      {
        "type": "Feature",
        "geometry": {
          "type": "Quad",
          "coordinates": [[704, 487], [670, 554], [973, 644], [993, 590]]
        },
        "properties": {
          "contentZh": {"标签": "标签/4G-RRU3-光纤"}
        }
      },
      {
        "type": "Feature",
        "geometry": {
          "type": "LineString",
          "coordinates": [[614, 1271], [498, 1179], [419, 1216], [280, 1280]]
        },
        "properties": {
          "contentZh": {"标签": "光纤/有遮挡,有保护措施,弯曲半径合理/蛇形管"}
        }
      }
    ]
  }
}
```

---

## Output Sample

Training samples use native multi-geometry with hierarchical descriptions:

```json
{
  "images": ["images/QC-20230217-0000279_19621.jpeg"],
  "objects": [
    {
      "bbox_2d": [264, 144, 326, 201],
      "desc": "螺丝、光纤插头/BBU安装螺丝,显示完整,符合要求"
    },
    {
      "quad": [704, 487, 670, 554, 973, 644, 993, 590],
      "desc": "标签/4G-RRU3-光纤"
    },
    {
      "line": [614, 1271, 498, 1179, 419, 1216, 280, 1280, 117, 1456, 3, 1721],
      "desc": "光纤/有保护措施,弯曲半径合理/蛇形管"
    }
  ],
  "width": 532,
  "height": 728
}
```

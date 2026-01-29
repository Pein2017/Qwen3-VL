# Qwen2.5-VL Data Conversion Pipeline

> **Unified Processing Architecture – August 2025**
>
> Streamlined data conversion pipeline using `data_conversion/pipeline/unified_processor.py`
> for BBU equipment annotation processing.
> Focuses on rigorous geometry handling, hierarchical Chinese descriptions, and deterministic coordinate
> transformations.

---

## Table of Contents
1. [Overview](#overview)
2. [Performance & Parallel Processing](#performance--parallel-processing)
3. [Inputs](#inputs)
4. [What the Pipeline Does](#what-the-pipeline-does)
5. [Validation Architecture](#validation-architecture)
6. [Geometry & Coordinate Rules](#geometry--coordinate-rules)
7. [Outputs](#outputs)
8. [Object Ordering Consistency](#object-ordering-consistency)
9. [Raw Input Example (kept as reference)](#raw-input-example-kept-as-reference)
10. [Output Sample](#output-sample)

---

## Overview

The unified pipeline converts grpo_summary_1024_attr_key_recall annotations + images into training-ready JSONL with native geometry and
hierarchical descriptions. The processing emphasizes exact coordinate transformations and canonical geometry
ordering for stable learning signals.

- Core modules (under `data_conversion/pipeline/`):
  - `unified_processor.py`: Orchestrates the end-to-end pipeline
  - `coordinate_manager.py`: EXIF orientation → dimension rescaling → smart resize (in that order)
  - `flexible_taxonomy_processor.py`: grpo_summary_1024_attr_key_recall annotation parsing and hierarchical description generation
  - `validation_manager.py`: Object/sample validation and reporting
  - `vision_process.py`: Smart resize with factor alignment and pixel budget

### Background: Raw Inputs vs. Desired Outputs

| Aspect | Raw export (`raw_ds/.../*.json`) | Desired converted output (`data/{dataset}/all_samples.jsonl`) |
| --- | --- | --- |
| Geometry encoding | Mix of `dataList` rectangles, `markResult` quads/lines, inconsistent vertex order | Pure `bbox_2d` / `poly` / `line`, canonicalized：bbox TL→BR，poly 起点为最上最左顶点并按质心顺时针排序 |
| Image orientation | EXIF metadata may rotate pixels without updating annotations | Pixel data normalized (EXIF applied), annotations adjusted to match |
| Description text | Annotator-provided Chinese strings with optional “完整/部分” tokens and occlusion notes | Key=value `desc` with sanitized tokens (`可见性=完整/可见性=部分`), remarks preserved (BBU-only) |
| Object ordering | Whatever the annotation platform emitted | Strict “top-to-bottom then left-to-right” ordering for reproducible prompts |
| File format | One JSON per image, nested feature structure | Flat JSONL, one line per sample, ready for training splits |

**Fixed value compression (BBU/RRU)**  
Non‑free‑text attributes are normalized to compact values during conversion (OCR/备注 are untouched):

- **可见性**: `完整` / `部分`
- **挡风板需求**: `免装` / `空间充足需安装`
- **挡风板符合性**: `按要求配备` / `未按要求配备`
- **安装方向**: `方向正确` / `方向错误`
- **符合性**: `符合` / `不符合`
- **保护措施**: `有保护` / `无保护`
- **保护细节**: `蛇形管` / `铠装` / `蛇形管+铠装`
- **弯曲半径**: `半径合理` / `半径不合理<4cm或成环`
- **捆扎**: `整齐` / `散乱`
- **安装状态**（RRU紧固类）: `合格` / `不合格`
- **标签**（RRU尾纤/接地线）: `有标签` / `无标签`
- **套管保护**（RRU尾纤）: `有套管` / `无套管`

Typical workflow:
1. Drop the raw export (e.g., `raw_ds/bbu_scene_2.0/...`) under the repo.
2. Configure `data_conversion/convert_dataset.sh` with paths, resize settings, and sanitization flags.
3. Run the script to produce `data/{dataset_name}/...` where each JSONL line matches the “Desired converted output” column.

---

## Performance & Parallel Processing

The pipeline supports **multiprocessing** for significant speedup on multi-core systems.

### Configuration

Set the number of parallel workers in `convert_dataset.sh`:

```bash
# Performance settings
NUM_WORKERS="8"  # Number of parallel workers (1=sequential, >1=parallel)
```

**Recommendations:**
- **Sequential mode** (`NUM_WORKERS="1"`): Use for debugging or small datasets (<100 images)
- **Parallel mode** (`NUM_WORKERS="4"` to `"8"`): Recommended for production datasets
  - Set to 4-8 for typical multi-core systems
  - Higher values may not improve performance due to I/O bottlenecks
  - Each worker processes images independently with its own validation state

### Expected Speedup

| Workers | Dataset Size | Expected Speedup | Notes |
|---------|--------------|------------------|-------|
| 1 | Any | 1x (baseline) | Sequential processing |
| 4 | 1000+ images | 3-4x | Good balance for most systems |
| 8 | 5000+ images | 5-7x | Optimal for high-core systems |
| 16+ | 10000+ images | 6-8x | Diminishing returns due to I/O |

**Performance factors:**
- Image I/O (reading/writing) is the main bottleneck
- Coordinate transformations and validation are CPU-bound
- Progress bar updates in real-time across all workers
- Memory usage scales linearly with worker count

### Implementation Details

- Uses Python's `multiprocessing.Pool` with `imap_unordered` for efficiency
- **Worker initialization**: Each worker initializes once and reuses its processor instance across all samples
  - Avoids repeated initialization overhead (1886 samples → only 8 initializations with 8 workers)
  - Worker initialization logs are suppressed to reduce console spam (only warnings/errors shown)
- Post-processing validation runs sequentially after parallel processing
- Invalid samples are aggregated from all workers for reporting

---

## Inputs

- Dataset directory with paired raw annotation JSON and images (grpo_summary_1024_attr_key_recall format)
- The pipeline accepts two raw schema variants:
  - `dataList`: rectangle selection stored as 2 points `[x1, y1], [x2, y2]`
  - `markResult.features`: native geometry per feature (e.g., `LineString`, `Quad`, `Polygon`)
- Required metadata: `info.width`, `info.height` in each JSON
- Supported object types (strict): `{bbu, bbu_shield, connect_point, label, fiber, wire}`
  - Geometry constraints:
    - `fiber`, `wire` → line geometries
    - `bbu`, `bbu_shield`, `connect_point`, `label` → poly or bbox geometries
  - RRU 扩展：`station/rru/rru_screw/ground_screw/fastener/lable/fiber_rru/wire_rru`


---

## What the Pipeline Does

- Object extraction (grpo_summary_1024_attr_key_recall):
  - Reads `dataList` and/or `markResult.features`
  - Determines `object_type` using Chinese keys and taxonomy mapping
  - Filters to the supported set (unknown types are dropped)

- Description construction (key=value):
  - Uses exact Chinese keys and a strict mapping
  - Separator rules: comma (`,`) between key=value pairs; multi-values joined with `|`
  - BBU keeps `备注` when present; RRU omits `备注` and may include `组`

- Geometry normalization and canonicalization:
  - Converts grpo_summary_1024_attr_key_recall geometries to native formats: `bbox_2d`, `poly`, `line`
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
  - Strips occlusion tokens containing “遮挡” by default (configurable). This property is deprecated and shown to be unhelpful for training, so occlusion words like “有遮挡/无遮挡/挡风板有遮挡” are removed from `desc` during conversion.
  - Normalizes label OCR: whitespace removed, commas/pipes/equals escaped; unreadable labels emit `可读性=不可读` (no “可以识别/无法识别” rewrite).
  - Records invalid objects/samples for reporting

- Object ordering and image processing:
  - Sorts objects by top-to-bottom, then left-to-right using their first coordinate pair
  - Processes/copies the image to match final transform (EXIF and smart resize)

- Deterministic splitting and exports:
  - Splits all samples into train/val sets using deterministic random split
  - Writes flat-format JSONL files and summary artifacts

---

## Validation Architecture

The pipeline uses a two-layer validation strategy:

### 1. Object/Sample Validation (ValidationManager)
- **When:** During sample processing in `process_all_samples()`
- **What:** Validates individual objects and complete samples
- **How:** `ValidationManager.validate_sample()` and `filter_valid_objects()`
- **Output:** Detailed error reports with suggestions; invalid objects/samples tracked for reporting
- **Behavior:** Strict validation; invalid objects are filtered out, invalid samples can fail-fast or be skipped

### 2. Pipeline-Level Validation (StructureValidator)
- **When:** After splitting into train/val sets, before writing outputs
- **What:** Validates train/val split integrity
- **How:** `StructureValidator.validate_pipeline_output(train_samples, val_samples)`
- **Checks:**
  - Training samples are not empty
  - Validation samples are not empty (unless dataset is very small)
  - All samples have correct structure (images, objects fields)
  - No overlapping images between train and validation sets
- **Behavior:** Lightweight, side-effect free; raises ValueError if validation fails

### Configuration
- `fail_fast: bool` (default: True) - If True, invalid samples cause pipeline to exit; if False, invalid samples are skipped with warning
- Validation is always enabled; there is no "validation_mode" configuration

---

## Geometry & Coordinate Rules

### Geometry Types

- BBox (`bbox_2d`): `[x_min, y_min, x_max, y_max]` with `x_min < x_max`, `y_min < y_max`
- Poly (`poly`): Even number of integers representing polygon vertices (minimum 8 for 4-point polygons, extensible to arbitrary number of points)
  - Canonical ordering: **for all polygons** remove重复收尾→按质心顺时针排序→将“最上、再最左”顶点旋转到首位；输出顺序与可视化一致，避免交叉
- Line (`line`): `[x1, y1, x2, y2, ...]`
  - 2-point lines: ordered lexicographically by `(x, then y)`
  - Multi-point lines: preserve path structure; choose a canonical direction so the first point is the
    topmost-leftmost among endpoints; reverse the whole sequence if needed
- All coordinates are clamped within final image bounds and rounded to integers after transforms

### Coordinate Transformation Pipeline (CoordinateManager)

The pipeline applies three transformation stages in strict order to all geometry types:

**Stage 1: EXIF Orientation Compensation**
- Reads EXIF orientation tag from image file
- Transforms coordinates if image has rotation/flip metadata (orientations 2-8)
- Skipped if JSON dimensions already match oriented dimensions
- Example: Portrait photo stored as landscape with 90° rotation tag

**Stage 2: Dimension Mismatch Rescaling**
- Compares JSON metadata dimensions (`info.width`, `info.height`) with actual image dimensions
- Applies proportional scaling if dimensions differ
- Common when images are pre-processed but annotations are not updated
- Example: JSON says 1920x1080, but actual image is 960x540 (downscaled by 0.5x)

**Stage 3: Smart Resize Scaling**
- Resizes image to fit within `max_pixels` constraint (e.g., 786432 for 768*32*32)
- Maintains aspect ratio to avoid distortion
- Ensures dimensions are multiples of `image_factor` (e.g., 32 for model requirements)
- Scales coordinates proportionally to match resized image
- Example: 1920x1080 → 896x512 (within pixel budget, factor=32)

**API Usage:**
```python
# Primary entry point - applies all three stages
bbox, transformed_geom, final_w, final_h = CoordinateManager.transform_geometry_complete(
    geometry_input=obj["poly"],  # or obj["bbox_2d"], obj["line"]
    image_path=Path("image.jpg"),
    json_width=1920,
    json_height=1080,
    smart_resize_factor=32,
    max_pixels=786432,
    enable_smart_resize=True
)

# Validate after transformation
if not CoordinateManager.validate_geometry_bounds(transformed_geom, final_w, final_h):
    logger.warning("Invalid geometry - out of bounds")
```

**Key Properties:**
- All transformations are deterministic and stateless
- Coordinates are clamped to image bounds after each stage
- Works with all geometry types (bbox, poly, line, GeoJSON)
- Individual stages can be used separately for advanced use cases

---

## Object Ordering Consistency

- 预处理与 prompts 使用同一排序规则：先按 **Y 坐标升序**（自上到下），再按 **X 坐标升序**（自左到右）。
- 默认策略（`reference_tlbr`，兼容历史导出）参考点：
  - `bbox_2d`: 左上角 `(x1, y1)`
  - `poly`: 第一个顶点 `(x1, y1)`
  - `line`: 最左端点（X 最小；如 X 相同取 Y 最小）
- 可选策略（`center_tlbr`）：使用几何外接 AABB 的中心点 `(cx, cy)` 作为排序参考点（仍按 Y→X 排序）。
- 默认执行 TLBR 重排；如需与历史导出保持字节级一致，可用 `--preserve_annotation_order` 或在 `DataConversionConfig` 设置 `preserve_annotation_order=True` 跳过重排。
- 通过 `--object_ordering_policy {reference_tlbr|center_tlbr}` 或环境变量 `OBJECT_ORDERING_POLICY` 选择策略。
- 实现：`data_conversion/utils/sorting.py`；调用点：`unified_processor.py` 在写出前统一排序；prompt 描述位于 `src/config/prompts.py`。
- 验证脚本：`verify_ordering_consistency.py`。

---

## Outputs

Directory for each processed dataset (example shown as `data/{dataset_name}/`):

```
train.jsonl            # Training samples (flat)
val.jsonl              # Validation samples (flat)
all_samples.jsonl      # Combined flat samples (train + val)
label_vocabulary.json  # Aggregated labels and statistics
validation_report.json # Summary + counts of invalid objects/samples
invalid_objects.jsonl  # Per-object validation failures (detailed)
invalid_samples.jsonl  # Per-sample skip reasons
images/                # Processed images (EXIF-corrected, smart-resized)
```

Key notes:
- All JSONL files use the same flat format: `{ "images": [..], "objects": [..], "width": W, "height": H }`
- Objects preserve their native geometry key: one of `bbox_2d`, `poly`, `line`, plus a hierarchical `desc`
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
          "标签": "螺丝、光纤插头/BBU安装螺丝,完整,符合"
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
          "contentZh": {"标签": "光纤/有遮挡,有保护,半径合理/蛇形管"}
        }
      }
    ]
  }
}
```

---

## Output Sample

Training samples use native multi-geometry with key=value descriptions:

```json
{
  "images": ["images/QC-20230217-0000279_19621.jpeg"],
  "objects": [
    {
      "bbox_2d": [264, 144, 326, 201],
      "desc": "类别=BBU安装螺丝,可见性=完整,符合性=符合"
    },
    {
      "poly": [704, 487, 670, 554, 973, 644, 993, 590],
      "desc": "类别=标签,文本=4G-RRU3-光纤"
    },
    {
      "line": [614, 1271, 498, 1179, 419, 1216, 280, 1280, 117, 1456, 3, 1721],
      "desc": "类别=光纤,保护措施=有保护,保护细节=蛇形管,弯曲半径=半径合理"
    }
  ],
  "width": 532,
  "height": 728
}
```

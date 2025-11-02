# Data & Datasets

Comprehensive guide to data format, schema, dataset builders, and preprocessing pipeline.

**Source of truth**: `src/datasets/`, `src/datasets/data_details.md`, `src/datasets/geometry.py`

---

## Table of Contents
- [Data Format](#data-format)
- [Dataset Pipeline](#dataset-pipeline)
- [Builders](#builders)
- [Preprocessors](#preprocessors)
- [Best Practices](#best-practices)

---

## Data Format

### JSONL Schema

Each record in your training data follows this structure:

```json
{
  "images": ["path/to/img1.jpg", "path/to/img2.jpg"],
  "objects": [
    {"bbox_2d": [x1, y1, x2, y2], "desc": "物体描述"},
    {"quad": [x1, y1, x2, y2, x3, y3, x4, y4], "desc": "旋转框描述"},
    {"line": [x1, y1, ..., xn, yn], "line_points": N, "desc": "线段描述"}
  ],
  "width": 1920,
  "height": 1080,
  "summary": "可选: 单行中文汇总"
}
```

**Key Rules**:
- Image paths resolve relative to JSONL file directory (absolute paths also allowed)
- Exactly ONE geometry field per object (`bbox_2d`, `quad`, or `line`)
- For lines: `line_points` must equal number of coords ÷ 2
- Coordinates are in pixel space with original `width`/`height`

### Geometry Types

| Type | Format | Use Case |
|------|--------|----------|
| **bbox_2d** | `[x1, y1, x2, y2]` | Axis-aligned boxes |
| **quad** | `[x1,y1, x2,y2, x3,y3, x4,y4]` | Rotated rectangles (4 corners, clockwise) |
| **line** | `[x1,y1, ..., xn,yn]` + `line_points: N` | Polylines (cables, fibers) |

### Coordinate Normalization

**On Disk** (source data):
- Pixel coordinates with `width`/`height` metadata
- Keep original resolution for maximum precision

**During Training**:
- `custom.emit_norm` controls assistant text format:
  - `none`: Pixel coordinates in JSON
  - `norm100`: Normalized to [0, 100]
  - `norm1000`: Normalized to [0, 1000]
- Template automatically normalizes top-level `objects` to `norm1000` for grounding

### Modes: Dense vs Summary vs Mixed

**Dense Mode** (default):
```json
{
  "object_1": {"bbox_2d": [100, 200, 300, 400], "desc": "..."},
  "object_2": {"line_points": 4, "line": [50, 60, 80, 120, 130, 180, 180, 220], "desc": "..."}
}
```

**Summary Mode**:
```
"单行汇总文本"
```
Requires `summary` field in every record.

**Mixed Mode**:
- Set `summary_ratio` (e.g., 0.3 = 30% summary mode)
- Deterministic per epoch (seeded by epoch number)
- Useful for balancing detailed annotation and overview

### Summary Field Standard

Format requirements (aligned with training/inference prompts):
- Use Chinese comma '，' between items (object entries)
- Group identical items with `×N` (full-width ×, no space before N)
- Preserve visual order: 自上到下、再从左到右（线对象以最左端点为起点）
- Inside an object entry, keep taxonomy with slashes: 类型/属性[,属性]/[条件属性]
- Remarks only once at the end: prepend with `，备注: ...`
- Single sentence per image: no newlines, no trailing '。', no extra spaces
- No geometry or coordinate arrays in summary strings

**Example**: `光模块×3，线缆×2，BBU设备×1，备注: 顶部有灰尘`

---

## Dataset Pipeline

### Architecture Overview

```
JSONL → DenseCaptionDataset → Collator → Trainer
```

**Key Components**:
1. **DenseCaptionDataset**: Mode selection (dense/summary), augmentation config, per-item orchestration
2. **Preprocessors**: Validation, augmentation (plugged into the dataset)
3. **Builder**: Message formatting (JSONLinesBuilder)
4. **Collator**: Tensor preparation, optional packing

### Visual Feature Distillation (optional)

- Enable via `custom.visual_kd` when you want to lock the vision/aligner stack to a teacher while giving the language tower more room.
- The dataset already supplies `pixel_values` and `image_grid_thw`; as long as a record contains images, the trainer captures and distills the corresponding activations automatically.
- Batches without images (e.g., summary-only validation groups) skip the extra loss—no action required.

### DenseCaptionDataset

**Role**:
- Selects dense vs summary mode per sample
- Applies augmentation/preprocessing
- Attaches metadata for downstream processing and template encoding

**Configuration**:
```yaml
custom:
  train_jsonl: /path/to/train.jsonl
  val_jsonl: /path/to/val.jsonl
  summary_ratio: 0.0               # 0=dense, 1=summary, 0.3=30% summary
  emit_norm: norm1000              # Coordinate format in text
```

---

## Builders

### JSONLinesBuilder

**Purpose**: Formats single-image records into single-turn conversation messages

**Dense Mode**:
```python
# User message: embed the image followed by the prompt
[
  {"type": "image", "image": "path"},
  {"type": "text", "text": prompt}
]

# Assistant message: minimal object hierarchy (no per-image wrapper)
{
  "object_1": {"bbox_2d": [...], "desc": "类型/属性/..."},
  "object_2": {"line_points": 4, "line": [...], "desc": "..."}
}
```

**Summary Mode**:
```python
# Assistant message: single summary string
"标签×3，BBU设备×1，挡风板×1"
```

**Key Behavior**:
- Attaches top-level `objects` with pixel coords (for template normalization)
- Geometries normalized based on `emit_norm` setting
- Deterministic ordering of object indices (`object_1`, `object_2`, ...)
- Consumes validated `ConversationRecord` objects and exposes augmentation telemetry (`pipeline.last_summary`) for downstream health checks.

---

## Preprocessors

### DenseCaptionPreprocessor

**Purpose**: Validation and light filtering

**Checks**:
- Schema validity (required fields present)
- Geometry field uniqueness (exactly one per object)
- Line point count matches `line_points`
- Image paths resolve correctly

**Action**: Raises `ValueError` on invalid records (fail-fast)

### AugmentationPreprocessor

**Purpose**: Apply geometry-aware augmentations

**Features**:
- Atomic updates (image + geometries transformed together)
- Preserves coordinate alignment
- See [AUGMENTATION.md](AUGMENTATION.md) for details
- Reads standardized telemetry (`AugmentationTelemetry`) with crop coverage, kept indices, and skip reasons to audit augmentation pipelines.

**Example**:
```yaml
custom:
  augmentation:
    enabled: true
    bypass_prob: 0.1              # 10% clean samples
    ops:
      - name: hflip
        params: { prob: 0.5 }
      - name: rotate
        params: { max_deg: 25.0, prob: 0.4 }
      - name: expand_to_fit_affine
        params: { multiple: 32 }
      - name: color_jitter
        params: { brightness: [0.75, 1.25], prob: 0.5 }
```

### Domain Context: BBU Installation Inspection

This corpus covers telecom cabinet inspections, focused on **BBU (Baseband Unit) installation quality**. Understanding the domain helps keep dense annotations, summaries, and production flows aligned.

- **Dense captioning** (training) enumerates every inspected item. Object types map to the attribute taxonomy in `data_conversion/hierarchical_attribute_mapping.json` and `data_conversion/attribute_taxonomy.json` (ignore the `occlusion` block). Core categories include:
  - `bbu` (BBU设备) — expects attributes such as brand (`bbu_brand`), completeness (`bbu_stituation`), and windshield requirement (`bbu_equipment`).
  - `bbu_shield` (挡风板) — brand, completeness, obstruction status, and installation direction.
  - `connect_point` (接地/接线端子), `label` (标签), plus `fiber` / `wire` lines for cabling.
  Geometry constraints from the mapping JSON enforce quads/bboxes for hardware and polylines for cabling.
- **Summary / Stage-A / Stage-B** (production) reuse the same records:
  - Stage-A generates per-image summaries (requires every record to carry a `summary`).
  - Stage-B consumes multiple images per site to emit a final deployment verdict.
  - Keep taxonomies in sync so Stage pipelines and dense training stay compatible.
- **Hierarchy semantics**: Attribute templates are slash-delimited (`brand/completeness/...`). Conditional levels (e.g., windshield conformity) only appear when parent attributes require them. Free-text `special_situation` fields append at the end.

When adding new inspection criteria, update both conversion JSON files and regenerate summaries before training to avoid drift between dense captions and production outputs.

---

## Best Practices

### Data Preparation

✅ **Do**:
- Keep pixel coords on disk (template normalizes during training)
- Use relative image paths when possible
- Validate schema before training
- Include `width`/`height` metadata
- Test with small dataset first

❌ **Don't**:
- Mix coordinate systems in same file
- Omit required fields (`width`, `height`)
- Use absolute paths unnecessarily
- Skip validation (fail early is better)

### Schema Validation

```bash
# Recommended: validate before training
python -m src.datasets.validate_jsonl --input train.jsonl --verbose
```

**Common Issues**:
- Missing `line_points` for line geometries
- Multiple geometry fields per object
- Path resolution failures
- Width/height mismatch

### Performance Tips

1. **Image Loading**: Use relative paths from JSONL directory for portability
2. **Augmentation**: Enable only needed ops (each adds overhead)
3. **Packing**: Set `training.packing: true` for 20-30% speedup

### Debugging

**Enable debug mode**:
```bash
python -m src.sft --config config.yaml --debug
```

**Check first batch**:
```python
from src.datasets import DenseCaptionDataset
ds = DenseCaptionDataset(config)
item = ds[0]
print(item.keys())  # input_ids, labels, pixel_values, ...
```

---

## Collation & Packing

### Standard Collation

**Output Tensors**:
- `input_ids`: Token IDs (includes vision placeholders)
- `labels`: Target tokens (-100 for non-target positions)
- `pixel_values`: Preprocessed images
- `image_grid_thw`: Grid dimensions per image
- `objects`: Top-level geometry metadata (norm1000)

### Packing Mode

**When**: `training.packing: true`

**Benefits**:
- Eliminates padding waste
- 20-30% faster training
- Better GPU utilization

**Limitations**:
- Incompatible with `lazy_tokenize`
- Requires Qwen3-VL (Flash Attention 2+)

---

## Verification Checklist

Before training:

- [ ] JSONL schema valid (all required fields present)
- [ ] Geometry fields correct (one per object)
- [ ] Line objects have `line_points` matching coord count
- [ ] Image paths resolve correctly
- [ ] Width/height metadata present
- [ ] Summary field present (if using summary mode)
- [ ] Coordinates in pixel space (not normalized)
- [ ] No duplicate objects or malformed geometries
- [ ] Test with `--debug` flag first

---

## See Also

- **Augmentation**: [AUGMENTATION.md](AUGMENTATION.md) - Geometry-aware transforms
- **Training**: [REFERENCE.md](REFERENCE.md#training) - Full training guide
- **Architecture**: [README.md](README.md#architecture) - End-to-end pipeline
- **Upstream Models**: [UPSTREAM_DEPENDENCIES.md](UPSTREAM_DEPENDENCIES.md) - HF Qwen3-VL + ms-swift background

---

**Last Updated**: 2025-10-28 (v1.1.2)

# Data Formats

Complete reference for JSONL data schemas, geometry formats, and data preparation for Qwen3-VL training.

## Table of Contents

- [JSONL Schema](#jsonl-schema)
- [Dense vs Summary Modes](#dense-vs-summary-modes)
- [Geometry Formats](#geometry-formats)
- [Coordinate Normalization](#coordinate-normalization)
- [Summary Field Format](#summary-field-format)
- [Data Verification](#data-verification)

## JSONL Schema

Each line in the training/validation JSONL file represents one sample.

### Basic Schema

```json
{
  "images": ["path/to/img1.jpg", "path/to/img2.jpg"],
  "objects": [
    {
      "bbox_2d": [x1, y1, x2, y2],
      "desc": "BBU设备/华为,显示完整,安装牢固"
    },
    {
      "quad": [x1, y1, x2, y2, x3, y3, x4, y4],
      "desc": "标签/5G-BBU光纤"
    },
    {
      "line": [x1, y1, x2, y2, ..., xn, yn],
      "line_points": 4,
      "desc": "光纤/有保护措施,弯曲半径合理/蛇形管"
    }
  ],
  "width": 1920,
  "height": 1080,
  "summary": "BBU设备/华为/显示完整×1，螺丝、光纤插头×2，标签/可以识别×1"
}
```

### Field Descriptions

**Required fields:**
- `images` (List[str]): Image paths, resolved relative to JSONL file's directory
- `objects` (List[Dict]): Object annotations with geometry + description
- `width` (int): Original image width in pixels
- `height` (int): Original image height in pixels

**Optional fields:**
- `summary` (str): One-line Chinese summary (required if `summary_ratio > 0` in training config)

### Image Paths

Paths in `images` are resolved relative to the JSONL file's directory:

```python
# Training runner auto-sets ROOT_IMAGE_DIR
ROOT_IMAGE_DIR = os.path.dirname(jsonl_path)

# Relative paths are resolved:
"images": ["img/001.jpg"]  # → {ROOT_IMAGE_DIR}/img/001.jpg

# Absolute paths work too:
"images": ["/abs/path/to/img.jpg"]
```

### Object Descriptions

Use hierarchical slash-separated format:

```
类型/属性[,属性]/[条件属性]/[备注(前缀"备注:")]
```

**Examples:**
- `BBU设备/华为,显示完整,安装牢固`
- `螺丝、光纤插头/BBU安装螺丝,只显示部分,符合要求`
- `标签/5G-AAU1光纤`
- `光纤/有保护措施,弯曲半径合理/蛇形管`
- `挡风板/安装方向正确,备注:部分遮挡`

## Dense vs Summary Modes

Training supports dynamic selection between dense captioning (with geometry) and summary mode (text-only) per pairing group.

### Dense Mode (Default)

Model learns to output grouped JSON with geometry:

```json
{
  "图片_1": {
    "object_1": {
      "bbox_2d": [264, 144, 326, 201],
      "desc": "BBU设备/华为/显示完整"
    },
    "object_2": {
      "quad": [100, 50, 200, 55, 195, 150, 95, 145],
      "desc": "标签/可以识别"
    }
  },
  "图片_2": { ...}
}
```

**Configuration:**
```yaml
custom:
  summary_ratio: 0.0  # or omit entirely
```

### Summary Mode

Model learns to output one-line summaries per image:

```json
{
  "图片_1": "BBU设备/华为/显示完整×1，螺丝、光纤插头×2，标签/可以识别×1",
  "图片_2": "光纤/有保护措施/蛇形管×3，标签/无法识别×1"
}
```

**Configuration:**
```yaml
custom:
  summary_ratio: 1.0  # All groups use summary mode
```

### Mixed Mode

Randomly select dense or summary mode per pairing group:

```yaml
custom:
  summary_ratio: 0.5  # 50% summary, 50% dense per group
```

**Key insight:** All samples in one pairing group see the same mode, ensuring coherent JSON shapes per batch.

### Mode Selection Behavior

With `summary_ratio`:
- `0.0`: Always dense (default)
- `1.0`: Always summary
- `0.5`: Each group has 50% chance of being summary, 50% dense
- Selection is deterministic per epoch (seeded RNG)

**Data requirement:** When `summary_ratio > 0`, all records **must** have a valid `summary` field.

## Geometry Formats

### Bounding Box (bbox_2d)

Axis-aligned rectangle with top-left and bottom-right corners:

```json
{
  "bbox_2d": [x1, y1, x2, y2],
  "desc": "BBU设备/华为"
}
```

- `x1, y1`: Top-left corner
- `x2, y2`: Bottom-right corner
- Coordinates in pixel space (relative to original image dims)
- Must satisfy: `x1 < x2` and `y1 < y2`

### Quadrilateral (quad)

Four-corner polygon for rotated or perspective-distorted objects:

```json
{
  "quad": [x1, y1, x2, y2, x3, y3, x4, y4],
  "desc": "标签/5G-BBU光纤"
}
```

- 8 values: 4 corner points in order
- Typical ordering: top-left, top-right, bottom-right, bottom-left
- Coordinates in pixel space

### Line/Polyline (line)

Multi-segment line for cables, wires, or elongated objects:

```json
{
  "line": [x1, y1, x2, y2, ..., xn, yn],
  "line_points": 4,
  "desc": "光纤/有保护措施/蛇形管"
}
```

- Even number of values (2N for N points)
- `line_points` (int): Number of point pairs (N)
- Must have: `len(line) == 2 * line_points`
- Minimum 2 points (4 values)

## Coordinate Normalization

### Storage Format (JSONL)

Geometry is stored in **pixel coordinates** relative to original image dimensions:

```json
{
  "bbox_2d": [264, 144, 326, 201],  // Pixel coordinates
  "width": 1920,
  "height": 1080
}
```

### Text Output Format (Training Target)

The `emit_norm` setting controls what coordinate space appears in the assistant's JSON text:

```yaml
custom:
  emit_norm: norm1000  # Options: none, norm100, norm1000
```

- `none`: Pixel coordinates (e.g., `[264, 144, 326, 201]`)
- `norm100`: Normalized to [0, 100] (e.g., `[13, 13, 16, 18]`)
- `norm1000`: Normalized to [0, 1000] (e.g., `[137, 133, 169, 186]`)

**Recommended:** Use `norm1000` for consistency with Qwen2-VL conventions.

### Template Encoding

During encoding, the template **always** normalizes top-level `objects.bbox` to `norm1000` regardless of `emit_norm`:

```python
# Top-level objects for template
merged = {
    "messages": [...],
    "objects": {
        "bbox": [[264, 144, 326, 201]],  # Pixel coords
        "ref": ["BBU设备"],
        "image_id": [0]
    }
}

# Template encoding step:
# objects.bbox normalized to norm1000 automatically
```

This dual representation ensures:
- **Text targets** use `emit_norm` for readability/consistency
- **Grounding metadata** uses `norm1000` for template internals

## Summary Field Format

### Standardized All-Slash Format

Summary fields follow a unified format with no mixed comma/slash separators:

**Rules:**
1. Replace commas with slashes in descriptions
2. Categorize labels: `标签/可以识别` or `标签/无法识别`
3. Group identical items and count: `item×N`
4. Preserve object list order (no reordering)
5. Separate groups with Chinese comma `，`

### Examples

**Input objects:**
```json
[
  {"desc": "BBU设备/华为,显示完整,无需安装"},
  {"desc": "螺丝、光纤插头/BBU安装螺丝,只显示部分,符合要求"},
  {"desc": "螺丝、光纤插头/BBU安装螺丝,只显示部分,符合要求"},
  {"desc": "标签/5G-BBU光纤"}
]
```

**Output summary:**
```
BBU设备/华为/显示完整/无需安装×1，螺丝、光纤插头/BBU安装螺丝/只显示部分/符合要求×2，标签/可以识别×1
```

### Transformation Rules

**1. Flatten hierarchical levels:**
```
Input:  "BBU设备/华为,显示完整,无需安装"
Output: "BBU设备/华为/显示完整/无需安装"
```

**2. Categorize labels:**
```
Input:  "标签/5G-AAU1光纤"  → Output: "标签/可以识别"
Input:  "标签/无法识别"     → Output: "标签/无法识别"  (unchanged)
```

**3. Group and count:**
```
Input:  [item A, item A, item B]
Output: "item A×2，item B×1"
```

**4. Preserve order:**
```
# Items appear in the order they were in objects list
# No alphabetization or reordering
```

## Data Verification

Use the verification toolkit to validate your data:

### Verify Summary Mode Behavior

Show what the model will see with `summary_ratio=1.0`:

```bash
python scripts/verify_data.py summary-mode --jsonl data/train.jsonl --num-samples 3
```

Output shows:
- Images (ignored in summary mode)
- Objects (ignored in summary mode)
- Summary (what model actually sees)
- Target output format comparison

### Validate Summary Format

Check that summary fields follow standardized all-slash format:

```bash
python scripts/verify_data.py summary-format data/train.jsonl
```

Expected output:
```
Summary Format Validation: data/train.jsonl
Total samples:            2196
Correct summaries:        2196 ✅
Invalid summaries:        0
Accuracy:                 100.0%
```

### Validate Geometry

Check that geometry fields have correct dimensions and structure:

```bash
python scripts/verify_data.py geometry data/train.jsonl --verbose
```

Validates:
- Each object has exactly one geometry field (bbox_2d, quad, or line)
- Correct number of coordinates (4 for bbox, 8 for quad, even ≥4 for line)
- `line_points` matches line coordinate count

See also: REFERENCE → Upstream internals (ms‑swift media extraction) for the strict `{ "type": X, X: value }` contract in `swift/llm/template/template_inputs.py`.

### Common Validation Errors

**Missing summary field:**
```
ValueError: Missing or invalid 'summary' for record index 5
```
Solution: Add summary field to all records when using `summary_ratio > 0`.

**Invalid geometry dimensions:**
```
Record 10, Object 2: Invalid quad (expected 8 values, got 6)
```
Solution: Fix coordinate count in JSONL.

**Mismatched line_points:**
```
Record 15, Object 3: line_points mismatch (line_points=4 but line has 10 values)
```
Solution: Update `line_points` to match actual point count or fix line coordinates.

## Best Practices

### Data Preparation

1. **Use relative paths** for images (resolved relative to JSONL directory)
2. **Provide summary fields** if you plan to use mixed/summary training modes
3. **Validate early** with verification tools before training
4. **Keep original dimensions** in width/height fields
5. **Use pixel coordinates** in geometry (normalization happens at encode time)

### Quality Checks

Run all verification tools before starting training:

```bash
# Check geometry validity
python scripts/verify_data.py geometry data/train.jsonl

# Check summary format (if using summary mode)
python scripts/verify_data.py summary-format data/train.jsonl

# Understand mode behavior
python scripts/verify_data.py summary-mode --jsonl data/train.jsonl
```

### Split Creation

Create balanced train/val splits:

```python
import json
import random

# Load all records
with open('all_data.jsonl') as f:
    records = [json.loads(line) for line in f]

# Shuffle and split
random.shuffle(records)
split_idx = int(0.8 * len(records))
train = records[:split_idx]
val = records[split_idx:]

# Write splits
with open('train.jsonl', 'w') as f:
    for rec in train:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')

with open('val.jsonl', 'w') as f:
    for rec in val:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')
```

## Additional Resources

- **Training workflows**: See [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
- **Inference examples**: See [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md)
- **Advanced topics**: See [REFERENCE.md](REFERENCE.md)

---

**Last Updated**: October 24, 2025


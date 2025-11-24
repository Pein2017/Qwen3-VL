# Public Datasets for Qwen3-VL

This directory provides a **complete LVIS → Qwen3-VL JSONL pipeline** plus helpers for future public datasets (Objects365, Open Images, ...).

Use this README as the operational guide for:
- Downloading LVIS data
- Converting to Qwen3-VL JSONL (bbox-only or polygon)
- Sampling and validating subsets
- Integrating the outputs into Qwen3-VL training / fusion

> For how these public datasets are fused with BBU training, see
> `openspec/changes/update-geometry-poly-fusion/design.md` ("`public_data/` Integration" section).

## Environment

- Repo root: `/data/Qwen3-VL`
- This module: `/data/Qwen3-VL/public_data`
- Conda environment: `ms` (always run commands with `conda run -n ms`)
- Disk: ~25 GB for LVIS + COCO images, plus converted JSONL

## Directory Structure

```text
./
├── lvis/                      # LVIS dataset (1203 categories, long-tail)
│   ├── raw/                   # Original annotations + COCO images
│   ├── processed/             # Converted JSONL files and samples
│   ├── metadata/              # Category mappings, label lists
│   └── stats/                 # Conversion statistics
├── scripts/                   # Executable scripts
│   ├── download_lvis.py       # Download LVIS annotations + COCO images
│   ├── convert_lvis.py        # Convert LVIS → Qwen3-VL JSONL
│   ├── sample_dataset.py      # Sampling utilities
│   └── validate_jsonl.py      # JSONL schema / image checks
├── converters/                # Converter modules
│   ├── base.py                # Base converter interface
│   ├── lvis_converter.py      # LVIS-specific converter (bbox + polygons)
│   └── geometry.py            # Geometry helpers (bbox, bounds)
├── configs/                   # Conversion configs (e.g., lvis.yaml)
├── tests/                     # Converter tests
└── vis_tools/                 # Visualization helpers (bbox + polygon overlays)
```

---

## End-to-End LVIS Pipeline

All commands assume:

```bash
cd /data/Qwen3-VL/public_data
```

### 1. Run tests (optional but recommended)

```bash
conda run -n ms bash tests/run_tests.sh
```

### 2. Download LVIS annotations + COCO 2017 images

```bash
conda run -n ms python scripts/download_lvis.py
```

If you already have COCO images, you can skip downloading them:

```bash
conda run -n ms python scripts/download_lvis.py --skip_images
mkdir -p lvis/raw/images
ln -s /path/to/your/coco/train2017 lvis/raw/images/train2017
ln -s /path/to/your/coco/val2017   lvis/raw/images/val2017
```

### 3. (Optional) Visualize a few samples

```bash
conda run -n ms python vis_tools/visualize_lvis.py \
  --num_samples 5 \
  --save \
  --mode both
```

Outputs go to `vis_tools/output/*.png`.

### 4. Convert LVIS to Qwen3-VL JSONL

**BBox-only mode (faster):**

```bash
conda run -n ms python scripts/convert_lvis.py --split train
```

**Polygon mode (N-point polygons as `quad`):**

```bash
conda run -n ms python scripts/convert_lvis.py --split train --use-polygon
```

You can also run a small test conversion first:

```bash
conda run -n ms python scripts/convert_lvis.py --split train --use-polygon --test
```

### 5. Create sampled subsets (optional)

```bash
conda run -n ms python scripts/sample_dataset.py \
  --input lvis/processed/train.jsonl \
  --output lvis/processed/samples/train_5k_stratified.jsonl \
  --num_samples 5000 \
  --strategy stratified
```

Other strategies: `uniform`, `top_k` (see `sample_dataset.py`).

### 6. Validate JSONL output

```bash
conda run -n ms python scripts/validate_jsonl.py \
  lvis/processed/samples/train_5k_stratified.jsonl
```

Validation checks schema, image paths, and geometry.

---

## JSONL Output Schema

Each line in the output JSONL matches the Qwen3-VL dense-caption contract:

```json
{
  "images": ["relative/path/to/image.jpg"],
  "objects": [
    {"bbox_2d": [x1, y1, x2, y2], "desc": "person"},
    {"quad": [x1, y1, ..., xn, yn], "quad_points": n, "desc": "car"}
  ],
  "width": 640,
  "height": 480
}
```

Key points:
- Coordinates are **pixel values** (not normalized).
- `bbox_2d`: `[x1, y1, x2, y2]` derived from COCO `[x, y, w, h]`.
- `quad`: generic N-point closed polygon (N ≥ 3) with `quad_points = N`.
- Image paths are relative to the JSONL file if `relative_image_paths` is enabled.

---

## Polygon Support (LVIS Segmentation)

When `--use-polygon` is enabled:

- LVIS `segmentation` entries (one or more `[x1,y1,...]` parts) are converted to **polygon objects**:
  - Each part becomes one object with `quad` + `quad_points`.
  - At least 3 points (6 coords) and an even number of coords are required.
  - Coordinates must lie within a small margin of the image bounds.
- BBoxes are still available via `bbox_2d` when polygon conversion is disabled or when polygons are invalid.

To consume these polygons in Qwen3-VL, the dense-caption preprocessor must accept N-point `quad` geometries (not just 4 points), e.g.:

```python
# Pseudocode (Qwen3-VL side)
quad = obj["quad"]
assert len(quad) >= 6 and len(quad) % 2 == 0
assert "quad_points" in obj and len(quad) == obj["quad_points"] * 2
```

---

## Integration with Qwen3-VL Training

From the repo root (`/data/Qwen3-VL`), you can point a training config to the converted LVIS JSONL:

```yaml
# Example: configs/lvis_stage1.yaml
custom:
  train_jsonl: ./public_data/lvis/processed/samples/train_5k_stratified.jsonl
  val_jsonl: ./public_data/lvis/processed/val.jsonl
  emit_norm: norm1000
  images_per_user_turn: 1
```

- Templates in Qwen3-VL are responsible for converting pixel coords → `norm1000`.
- For multi-dataset fusion with BBU data, follow the design in
  `openspec/changes/update-geometry-poly-fusion/design.md`.

---

## Troubleshooting

- **Images not found during conversion**: check that `lvis/raw/images/train2017/` and `val2017/` exist or symlinks are correct.
- **Conversion slow**: test with `--max_samples` or `--test` before full runs.
- **Validation failures**: inspect errors from `scripts/validate_jsonl.py` and check `lvis/stats/` for details.
- **Disk / memory issues**: use sampled subsets (1k/5k) instead of full LVIS for quick experiments.

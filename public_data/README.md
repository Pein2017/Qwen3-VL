# Public Data (LVIS) Pipeline - Single Source

This README now contains **all guidance for `public_data/`**. The earlier files
`LVIS_QUICKSTART.md`, `NEXT_STEPS.md`, `POLYGON_SUPPORT.md`, `STATUS.md`, and
`SUMMARY.md` have been folded in here (stubs remain only to avoid dead links).

## Scope & Prereqs
- Focus: LVIS v1.0 -> Qwen3-VL JSONL (bbox or polygon).
- Repo root: `/data/Qwen3-VL`; module root: `/data/Qwen3-VL/public_data`.
- Conda env: `ms` (run commands with `conda run -n ms ...`).
- Tools: `wget`, `unzip`; Disk: ~25 GB for LVIS+COCO plus converted JSONL.

## Layout
public_data/
|-- lvis/            # raw -> processed data
|-- scripts/         # download/convert/sample/validate CLIs
|-- converters/      # conversion logic (bbox + polygon)
|-- vis_tools/       # visualization helpers
`-- tests/           # converter tests (no images required)

## Quick Recipes
Run from repo root unless noted.

**Smoke test (10 samples, polygon on, no smart-resize)**
```bash
cd public_data
conda run -n ms python scripts/download_lvis.py --skip_images   # if annotations only
conda run -n ms python scripts/convert_lvis.py --split train --use-polygon --test
# For bbox-only smoke test: drop --use-polygon
```

**Full conversion (train + val, bbox)**
```bash
cd public_data
conda run -n ms python scripts/download_lvis.py
conda run -n ms python scripts/convert_lvis.py --split train
conda run -n ms python scripts/convert_lvis.py --split val
```

**Polygon conversion**
```bash
conda run -n ms python scripts/convert_lvis.py --split train --use-polygon
```

**Sampling (stratified 5k from full train)**
```bash
conda run -n ms python scripts/sample_dataset.py \
  --input lvis/processed/train.jsonl \
  --output lvis/processed/samples/train_5k_stratified.jsonl \
  --num_samples 5000 \
  --strategy stratified \
  --stats
```

**Validate bbox JSONL** (validator currently checks bbox-only)
```bash
conda run -n ms python scripts/validate_jsonl.py \
  lvis/processed/train.jsonl
```

**Visualize a few samples**
```bash
conda run -n ms python vis_tools/visualize_lvis.py --num_samples 3 --mode both --save
```

## Conversion Behavior (from code)
- Input: LVIS COCO-format JSON (`lvis_v1_[split].json`) + COCO images under `lvis/raw/images/{train2017,val2017}`.
- Output JSONL: one line per image with `images`, `objects`, `width`, `height`; stats saved as `<output>_stats.json`.
- Bounding boxes: COCO `[x,y,w,h]` -> `[x1,y1,x2,y2]`; boxes are clipped unless `--no-clip-boxes`; min area/dim default 1 px.
- Crowd: skipped by default (`--keep-crowd` to keep).
- Relative paths: default; use `--absolute-paths` to keep absolute.
- Polygons: `--use-polygon` converts each segmentation part to a `poly` (N-point closed polygon). If no valid polygon, it falls back to bbox.
- Smart resize: `--smart-resize` uses `src/datasets/preprocessors/resize.SmartResizePreprocessor` to resize images + geometry. **Caveat:** that preprocessor currently scales `bbox_2d` and `poly` fields; `quad` is no longer accepted.
- Stats counters: total/skipped images & objects; polygon mode adds `poly_converted` and `polygon_skipped`.
- Known flag gap: `--poly-max-points` is parsed but not wired into `LVISConverter`; using it will raise a constructor error. Skip this flag for now.

## JSONL Schema (produced here)
```json
{
  "images": ["train2017/000000000001.jpg"],
  "objects": [
    {"bbox_2d": [x1, y1, x2, y2], "desc": "person"},
    {"poly": [x1, y1, ..., xn, yn], "poly_points": n, "desc": "car"}
  ],
  "width": 640,
  "height": 480
}
```
- Pixel coordinates; not normalized.
- `poly` is the generic N-point polygon; `poly_points` = N.
- One image per record is expected; validator warns otherwise.

## Sampling Strategies (scripts/sample_dataset.py)
- `stratified` (default): preserves object frequency distribution; draws per-category targets proportional to object counts.
- `uniform`: equal samples per category.
- `random`: dataset-wide random without replacement.
- `top_k`: sample from top-K most frequent categories (prints top list).
All strategies keep unique image lines; category stats printed when `--stats` is set.

## Validation & Tests
- `scripts/validate_jsonl.py`: checks JSON shape, bbox format, image existence (unless `--skip-image-check`). **Polygon objects are not yet validated** because the script requires `bbox_2d`; use bbox mode or add a custom check for polygons.
- Test suite (no images needed): `cd public_data && conda run -n ms bash tests/run_tests.sh`. It exercises annotation loading, bbox conversion, polygon extraction, and format compliance.
- Test-mode validation inside `convert_lvis.py` is lightweight and may report JSON errors because of a double-parse bug; rely on `validate_jsonl.py` for ground truth.

## Training Integration (example)
```yaml
custom:
  train_jsonl: ./public_data/lvis/processed/samples/train_5k_stratified.jsonl
  val_jsonl:   ./public_data/lvis/processed/val.jsonl
  emit_norm: norm1000
  images_per_user_turn: 1
```
- Templates in Qwen3-VL convert pixel coords to `norm1000`.
- For multi-dataset fusion, see `openspec/changes/update-geometry-poly-fusion/design.md`.

## Dataset Facts
- LVIS v1.0: 1203 categories (long-tail: frequent/common/rare), ~100k train images, ~20k val, ~1.27M annotations.
- Polygons are common and detailed (many 10-40 point contours); converter accepts any N >= 3.

## Troubleshooting
- Missing images: ensure `lvis/raw/images/train2017` and `val2017` exist (copy from your COCO install if needed).
- Slow conversion: use `--max_samples` or `--test` first.
- Validator failures on polygons: run bbox mode or add a temporary bbox-only pass; polygon validation is a TODO.
- Disk pressure: sample down with `sample_dataset.py` before training.

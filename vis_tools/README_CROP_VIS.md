# Crop Augmentation Visualization Guide

This repo no longer supports the legacy crop ops `random_crop` / `small_object_zoom_paste`.
The supported zoom-in mechanism is **device-anchored ROI crop** (`roi_crop`).

The visualization entrypoint is:
- `vis_tools/vis_augment_compare.py`

---

## Quick Start

### 1) Visualize the full augmentation curriculum (recommended)

This mode mirrors a training preset (including curriculum phases), and renders a few samples at several curriculum checkpoints.

```bash
conda run -n ms python vis_tools/vis_augment_compare.py \
  --mode curriculum \
  --config-yaml configs/train/sft/dense_1024.yaml \
  --jsonl data_new_schema/bbu_full_1024/train.jsonl \
  --out-dir vis_out/augment_curriculum_dense_1024 \
  --num-samples 12 --variants 3 --seed 2026
```

### 2) Focus on ROI crop only

This mode forces `roi_crop.prob=1.0` (ignores curriculum/bypass), which is useful for debugging anchor matching and crop geometry.

```bash
conda run -n ms python vis_tools/vis_augment_compare.py \
  --mode roi_crop \
  --config-yaml configs/train/sft/dense_1024.yaml \
  --jsonl data_new_schema/bbu_full_1024/train.jsonl \
  --out-dir vis_out/augment_roi_crop_focus \
  --num-samples 12 --variants 3 --seed 2026
```

---

## What To Look For

- **Applied ops list**: each saved image includes the op sequence in the title.
- **Object filtering**: object count deltas indicate drops due to coverage thresholds.
- **Partial visibility tokens**: cropped objects should only update the structured token:
  - `可见性=完整 → 可见性=部分`
  - `可见性=显示完整 → 可见性=只显示部分`

---

## Troubleshooting

### `roi_crop: no_anchor`

`roi_crop` matches anchors by **exact** `desc` token `类别=...` (no substring, no fallback).

- Ensure your dataset `desc` is comma-separated tokens and includes `类别=<category>`.
- Ensure `roi_crop.anchor_classes` in the YAML config uses the exact `类别` values present in your data.

---

## Related Docs

- `docs/data/DATA_AUGMENTATION.md` (operator semantics + tuning)
- `configs/train/sft/dense_1024.yaml` (reference preset; shows recommended op ordering)

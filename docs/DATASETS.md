# Datasets

Source of truth: `src/datasets/dense_caption.py`, `src/datasets/dynamic_pair.py`, `src/datasets/builders/jsonlines.py`, `src/datasets/preprocessors/*`, `src/datasets/augmentation/*`, `src/datasets/collators.py`

## Overview
- `DenseCaptionDataset`: selects dense vs summary mode per pairing group; configures augmentation
- `DynamicPairDataset`: epoch-seeded pairing and per-item orchestration
- `JSONLinesBuilder`: builds one-turn chat; user embeds all images; assistant returns grouped JSON (dense) or per-image lines (summary)

## Builders
- `JSONLinesBuilder` (dense): grouped JSON per 图片_i with object entries
- `JSONLinesBuilder` (summary): `{ 图片_i: "<one-line>" }` only; geometries omitted from text
- Both attach top-level `objects` with pixel coords for template normalization

## Preprocessors
- `DenseCaptionPreprocessor`: validation and light filtering
- `AugmentationPreprocessor`: geometry-aware affine ops (image + points updated atomically)

## Augmentation (examples)
```yaml
custom:
  augmentation:
    enabled: true
    ops:
      - name: hflip
        params: { prob: 0.3 }
      - name: rotate
        params: { max_deg: 15.0, prob: 0.3 }
      - name: color_jitter
        params: { brightness: [0.85, 1.15], contrast: [0.85, 1.15], prob: 0.3 }
      - name: pad_to_multiple
        params: { multiple: 32 }
```

## Collation & packing
- Collators emit: `input_ids`, `labels`, `pixel_values`, `image_grid_thw`, `objects`
- Optional packing (`training.packing: true`) concatenates samples to eliminate padding waste

## See also
- Data schema and verification: `docs/DATA.md`
- End-to-end flow and health checks: `docs/ARCHITECTURE.md`

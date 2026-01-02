# Data Docs Index (`docs/data/`)

This folder documents the **data contract + conversion pipeline + domain knowledge** for Qwen3‑VL (BBU/RRU + public auxiliary datasets).

## Quick Navigation

### Core Contract
- `DATA_JSONL_CONTRACT.md` — **global JSONL contract**: record keys, geometry rules, and `summary` requirements.
- `POLYGON_SUPPORT.md` — polygon canonicalization + augmentation/visualization expectations.

### Intake / Conversion
- `DATA_PREPROCESSING_PIPELINE.md` — annotation export → validated train/val JSONL (+ QA artifacts).
- `DATA_AND_DATASETS.md` — dataset builders, modes (dense vs summary), and how conversion/fusion plug into training.

### Domain Knowledge (BBU / RRU)
- `BBU_RRU_BUSINESS_KNOWLEDGE.md` — BBU/RRU **类别/属性/分组规则**速查（面向 Prompt / Stage‑A/B / 数据质检快速索引）。

### Fusion / Public Data
- `UNIFIED_FUSION_DATASET.md` — fusion dataset design (single template, per-source policies, telemetry).
- `PUBLIC_DATA.md` — `public_data/` responsibilities, where to find converters/tests/ops guides.

### Augmentation
- `DATA_AUGMENTATION.md` — augmentation configuration and operator details (rotation-safe, crop filtering, pixel safety).

## Suggested Reading Order
1. `DATA_JSONL_CONTRACT.md`
2. `DATA_PREPROCESSING_PIPELINE.md`
3. `DATA_AND_DATASETS.md`
4. `BBU_RRU_BUSINESS_KNOWLEDGE.md` (when working on BBU/RRU)
5. `DATA_AUGMENTATION.md` / `POLYGON_SUPPORT.md` (when touching geometry or augmentation)
6. `UNIFIED_FUSION_DATASET.md` / `PUBLIC_DATA.md` (when mixing datasets)


# Architecture Overview

Status: Active
Scope: High-level architecture and data flow for Qwen3-VL training and inference.
Owners: Training + Runtime
Last updated: 2026-01-02
Related: [training/REFERENCE.md](../training/REFERENCE.md), [runtime/STAGE_A_STAGE_B.md](../runtime/STAGE_A_STAGE_B.md), [data/DATA_JSONL_CONTRACT.md](../data/DATA_JSONL_CONTRACT.md)

**Source anchors**: `src/sft.py`, `src/datasets/`, `src/utils/`

## End-to-End Pipeline (Config-Driven)

```
YAML → ConfigLoader → SwiftSft → DenseCaptionDataset → Trainer
```

**Key design principles**:
- Single length knob: `global_max_length` (proxies both model & template).
- Over-length safety: when the template raises `MaxLengthError` (e.g., `truncation_strategy: raise`), `DenseCaptionDataset` raises a hard error so training stops instead of truncating or silently skipping.
- Adapters applied before trainer: `sft.prepare_model(...)`.
- Config-only surface (avoid CLI flags beyond `--config`).
- Typed configuration contracts (`src/config/schema.py`) validate YAML before training launches.
- **Required**: `data.dataset: ["dummy"]` in all configs (ms-swift validation requirement; see [data/DATA_AND_DATASETS.md](../data/DATA_AND_DATASETS.md)).

## Model Components & Token Flow

```
Vision Encoder (ViT) → Aligner (Projector) → LLM
```

**Key points**:
- Chat template inserts image placeholders automatically.
- Do NOT hand-craft `<|image_pad|>` tokens.
- Placeholder count scales with `image_grid_thw`.
- Vision embeddings replace placeholders at runtime.

## Data Flow at a Glance

1. **Load** JSONL record (images, objects, width, height, optional summary)
2. **Group/Pair** with epoch-seeded RNG → optional augmentation
3. **Build** one-turn messages (user embeds all images; assistant returns JSON)
4. **Encode** via template (adds vision tokens, normalizes coords to norm1000, tokenizes)
5. **Train** consumes tensors: `input_ids`, `labels`, `pixel_values`, `image_grid_thw`, `objects`

## Health Checks (Fail-Fast)

- ✅ Image count in user turn matches placeholders
- ✅ `image_grid_thw` aligns with `pixel_values`
- ✅ Assistant spans end correctly; image tokens masked
- ✅ Non-target tokens labeled −100
- ✅ Geometry kept at top level in pixel space; template normalizes to norm1000

# Qwen3‑VL Docs

Status: Active — Internal Engineering

## Quick Navigation
- **What's New** → `CHANGELOG.md` 🆕
- **Data & Datasets** → `DATA_AND_DATASETS.md` - Schema, builders, preprocessing
- **Augmentation** → `AUGMENTATION.md` - Geometry transforms, smart cropping
- **Training & Inference** → `REFERENCE.md` - Full guide, recipes, FAQ
- **Stage-A & Stage-B** → `STAGE_A_STAGE_B.md` - Quality control pipeline, reflection loop
- **Upstream Dependencies** → `UPSTREAM_DEPENDENCIES.md` - HF Qwen3-VL + ms-swift context
- **Experiments** → `experiments/` - Training comparisons, ablations

## Recent Updates

### v1.1.2 - Config & Telemetry Contracts (Oct 2025) 📐
- YAML loader now builds frozen dataclasses (`TrainingConfig`, `CustomConfig`, `SaveDelayConfig`, `VisualKDConfig`) with early validation and deterministic merging.
- Datasets adopt shared contracts for records/geometry; augmentation telemetry is a typed dataclass surfaced to preprocessors and debug logging.
- Stage-A CLI wraps runtime flags in `StageAConfig`, catching invalid missions/paths before inference launches.

### v1.1.1 - Quad Truncation Refinement (Oct 2025) 🔧
- Fixed rotate+crop quad handling: rotated quads now maintain rotation after crop
- Added polygon simplification to preserve true quad corners
- Perfect boundary truncation without spurious AABB conversion

### v1.1.0 - Smart Cropping with Label Filtering (Oct 2025) 🆕
- **RandomCrop** operator with automatic label filtering and geometry truncation
- Perfect visual-label alignment for dense detection captioning
- Completeness field tracking: `显示完整` ↔ `只显示部分` updates
- Quad rotation fix + redundancy cleanup (removed CenterCrop, Equalize)
- See [CHANGELOG.md](CHANGELOG.md) for full details

---

## Architecture Overview

**Source**: `src/sft.py`, `src/datasets/`, `src/utils/`

### End-to-End Pipeline (Config-Driven)

```
YAML → ConfigLoader → SwiftSft → DenseCaptionDataset → Trainer
```

**Key Design Principles**:
- Single length knob: `global_max_length` (proxies both model & template)
- Adapters applied before trainer: `sft.prepare_model(...)`
- Config-only surface (avoid CLI flags beyond `--config`)
- Typed configuration contracts (`src/config/schema.py`) validate YAML before training launches.

### Model Components & Token Flow

```
Vision Encoder (ViT) → Aligner (Projector) → LLM
```

**Key Points**:
- Chat template inserts image placeholders automatically
- Do NOT hand-craft `<|image_pad|>` tokens
- Placeholder count scales with `image_grid_thw`
- Vision embeddings replace placeholders at runtime

### Data Flow at a Glance

1. **Load** JSONL record (images, objects, width, height, optional summary)
2. **Group/Pair** with epoch-seeded RNG → optional augmentation
3. **Build** one-turn messages (user embeds all images; assistant returns JSON)
4. **Encode** via template (adds vision tokens, normalizes coords to norm1000, tokenizes)
5. **Train** consumes tensors: `input_ids`, `labels`, `pixel_values`, `image_grid_thw`, `objects`

### Health Checks (Fail-Fast)

- ✅ Image count in user turn matches placeholders
- ✅ `image_grid_thw` aligns with `pixel_values`
- ✅ Assistant spans end correctly; image tokens masked
- ✅ Non-target tokens labeled −100
- ✅ Geometry kept at top level in pixel space; template normalizes to norm1000

---

## Doc ↔ Code Map

| Documentation | Source Code |
|---------------|-------------|
| Data & Datasets | `src/datasets/`, `src/datasets/data_details.md` |
| Augmentation | `src/datasets/augmentation/`, `src/datasets/geometry.py` |
| Training & Inference | `src/sft.py`, `src/stage_a/`, `src/stage_b/`, `configs/` |
| Stage-A & Stage-B | `src/stage_a/`, `src/stage_b/`, `configs/stage_b/` |
| Utils & Logging | `src/utils/`, `src/callbacks/` |

---

**Last Updated**: 2025-10-27 (v1.1.1)

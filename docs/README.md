# Qwen3‑VL Docs

Status: Active — Internal Engineering

## Quick Navigation
- **Intake & preprocessing** → `data/DATA_PREPROCESSING_PIPELINE.md` (annotation → JSONL) → `data/DATA_JSONL_CONTRACT.md`
- **Data & datasets** → `data/DATA_AND_DATASETS.md` (schema/builders/conversion), `data/DATA_AUGMENTATION.md`, `data/POLYGON_SUPPORT.md`
- **Fusion & public data** → `data/UNIFIED_FUSION_DATASET.md`, `data/PUBLIC_DATA.md`
- **Training & reference** → `training/TRAINING_PLAYBOOK.md`, `training/REFERENCE.md`
- **Stage‑1 runtime** → `runtime/STAGE_A_RUNTIME.md`
- **Stage‑2 runtime** → `runtime/STAGE_B_RUNTIME.md`
- **Business pipeline & guidance** → `runtime/STAGE_A_STAGE_B.md`, `stage-B-knowledge-Chinese.md`
- **Upstream dependencies** → `platform/UPSTREAM_DEPENDENCIES.md`
- **Specs & governance** → `openspec/AGENTS.md`, `openspec/project.md`

### Suggested Reading Order
1. **Intake → schema** — `data/DATA_PREPROCESSING_PIPELINE.md`, `data/DATA_JSONL_CONTRACT.md`, `data/DATA_AND_DATASETS.md`
2. **Augmentation** — `data/DATA_AUGMENTATION.md`
3. **Training & fusion** — `training/TRAINING_PLAYBOOK.md`, `data/UNIFIED_FUSION_DATASET.md`, `training/REFERENCE.md`
4. **Stage‑1 runtime** — `runtime/STAGE_A_RUNTIME.md`
5. **Stage‑2 runtime** — `runtime/STAGE_B_RUNTIME.md`, `runtime/STAGE_A_STAGE_B.md`
6. **Ecosystem** — `data/PUBLIC_DATA.md`, `platform/UPSTREAM_DEPENDENCIES.md`

### Documentation Ownership & Directory Map

| Directory | Primary doc(s) | Scope |
|-----------|----------------|-------|
| `src/` | `training/REFERENCE.md`, `training/TRAINING_PLAYBOOK.md` | Core training/inference implementation (`src/sft.py`, datasets, trainers). |
| `src/stage_a/` | `runtime/STAGE_A_RUNTIME.md`, `runtime/STAGE_A_STAGE_B.md` | Stage‑1 per-image object recognition and summary emission. |
| `src/stage_b/` | `runtime/STAGE_B_RUNTIME.md`, `runtime/STAGE_A_STAGE_B.md` | Stage‑2 verdict loop（rule_search）。 |
| `data_conversion/` | `data/DATA_PREPROCESSING_PIPELINE.md`, `data/DATA_AND_DATASETS.md` (Conversion section) | Optional offline preprocessing from annotation exports; taxonomy, resize, validation. |
| `public_data/` | `data/PUBLIC_DATA.md` | LVIS and auxiliary datasets (download, convert, sample, validate, visualize). |
| `scripts/` | `scripts/README.md`, `training/TRAINING_PLAYBOOK.md`, `runtime/STAGE_B_RUNTIME.md` | Canonical entrypoints: training, inference, Stage‑A/B launchers, dataset fusion. |
| `openspec/` | `openspec/AGENTS.md`, `openspec/project.md` | Change-management specs and proposal workflow. |
| `vis_tools/` | `data/DATA_AUGMENTATION.md`, `vis_tools/README_CROP_VIS.md` | Visualization/debug scripts for augmentation and QA spot checks. |

Whenever you add or modify code in the directories above, update the associated doc in the same PR to keep the handbook current.

### Script & Tooling Inventory

| Script | Location | Description |
|--------|----------|-------------|
| `train.sh` | `scripts/` | Conda-aware launcher for `python -m src.sft` / `torchrun` with config auto-resolution and debug toggles. |
| `fuse_datasets.py` | `scripts/` | Offline builder for `src/datasets/fusion.py` configs; pre-mixes BBU + auxiliary JSONL with deterministic ratios. |
| `download.py` | `scripts/` | Download helper for internal/raw corpora (mirrors instructions in `docs/data/DATA_AND_DATASETS.md`). |
| `stage_a.sh` | `scripts/` | Mission-aware wrapper around `src.stage_a.cli` with guardrails for checkpoint/input directories. |
| `stage_b.sh` | `scripts/` | Stage-B rule-search launcher; wires configs to `src.stage_b.runner` (supports `smoke` no-model audit). |
| `stage_b_smoke.py` | `scripts/` | No-model Stage-B smoke/audit (config+ingest+guidance+prompt+parse+export). |
| `validate_sft_config.py` | `scripts/` | Fast YAML validation for SFT configs (no model weights). |
| `validate_dense_jsonl_contract.py` | `scripts/` | Fast JSONL contract validation for dense-caption records. |
| `build_irrelevant_summary_jsonl.py` | `scripts/` | Builds `data/irrelevant_summary/train.jsonl` from `data/irrelevant_summary/images/*.jpg|*.jpeg` with `summary: 无关图片` (dummy full-frame bbox for contract compatibility). |
| `debug_fusion_template_clone.py` | `scripts/` | Smoke-test for template reuse in fused datasets (regression guard for cloning bugs). |
| `merge_stage2_lora.sh` | `scripts/` | Utility to merge staged LoRA checkpoints for deployment. |

Use these scripts instead of ad-hoc commands so telemetry, logging, and environment setup stay consistent across teams.

## Recent Updates

### v1.1.3 - RRU Support & Canonical Polygons (Nov 2025) 🛰️
- Unified converter now handles RRU raw annotations: taxonomy additions (`ground_screw`, 尾纤/接地线标签与套管保护, 站点距离=数字), group membership encoded in `desc` via `组=<id>` (no `groups` field). Summaries are JSON strings with per-category stats (no ×N aggregation).
- Polygon vertices are canonicalized offline (clockwise, top-most then left-most first) and `vis_tools` mirrors the ordering to avoid self-crossing during visualization.

### v1.1.2 - Config & Telemetry Contracts (Oct 2025) 📐
- YAML loader now builds frozen dataclasses (`TrainingConfig`, `CustomConfig`, `SaveDelayConfig`, `VisualKDConfig`) with early validation and deterministic merging.
- Datasets adopt shared contracts for records/geometry; augmentation telemetry is a typed dataclass surfaced to preprocessors and debug logging.
- Stage-A CLI wraps runtime flags in `StageAConfig`, catching invalid missions/paths before inference launches.

### v1.1.1 - Quad Truncation Refinement (Oct 2025) 🔧
- Fixed rotate+crop polygon handling: rotated polygons now maintain rotation after crop
- Added polygon simplification to preserve true polygon corners
- Perfect boundary truncation without spurious AABB conversion

### v1.1.0 - Smart Cropping with Label Filtering (Oct 2025) 🆕
- **RandomCrop** operator with automatic label filtering and geometry truncation
- Perfect visual-label alignment for dense detection captioning
- Completeness field tracking: `可见性=完整` ↔ `可见性=部分` updates
- Quad rotation fix + redundancy cleanup (removed CenterCrop, Equalize)

---

## Major Change: Geometry Schema Overhaul (Nov 2025) 🚨

- **Big change**: the pipeline now publishes `poly` geometry entries everywhere (replacing the previous 4-point geometry key). Internally we still emit 4-point polygons today, but the schema and prompts are ready to hold arbitrary vertex counts going forward.
- All documentation, builders, and augmentation ops now expect `poly` (even-length list ≥6 values / ≥3 points) as one of the three canonical geometry keys (`bbox_2d`, `poly`, `line`).
- This change affects data conversion, dataset builders, augmentation telemetry, Stage-A/B workflows, and training prompts. Please regenerate derived artifacts and re-validate dataset probes if you re-run conversion scripts.

## Architecture Overview

**Source**: `src/sft.py`, `src/datasets/`, `src/utils/`

### End-to-End Pipeline (Config-Driven)

```
YAML → ConfigLoader → SwiftSft → DenseCaptionDataset → Trainer
```

**Key Design Principles**:
- Single length knob: `global_max_length` (proxies both model & template)
- Over-length safety: when the template raises `MaxLengthError` (e.g., `truncation_strategy: raise`), `DenseCaptionDataset` now raises a hard error so training stops instead of truncating or silently skipping.
- Adapters applied before trainer: `sft.prepare_model(...)`
- Config-only surface (avoid CLI flags beyond `--config`)
- Typed configuration contracts (`src/config/schema.py`) validate YAML before training launches.
- **Required**: `data.dataset: ["dummy"]` in all configs (ms-swift validation requirement; see `DATA_AND_DATASETS.md`)

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
| Data & Datasets | `src/datasets/`, `data_conversion/`, `scripts/fuse_datasets.py` |
| Augmentation | `src/datasets/augmentation/`, `src/datasets/geometry.py`, `vis_tools/` |
| Training Playbook & Reference | `src/sft.py`, `scripts/train.sh`, `configs/`, `src/callbacks/`, `src/trainers/` |
| Stage-A Runtime | `src/stage_a/`, `scripts/stage_a.sh`, `src/utils/logger.py` |
| Stage-B Runtime | `src/stage_b/`, `scripts/stage_b.sh`, `configs/stage_b/` |
| Public Data | `public_data/`, `public_data/scripts/`, `public_data/tests/` |
| Utils & Logging | `src/utils/`, `src/callbacks/` |

---

**Last Updated**: 2025-11-21 (Doc ownership refresh)

# Qwen3‑VL Docs

Status: Active — Internal Engineering

## Quick Navigation
- **What's New** → `CHANGELOG.md` 🆕
- **Data & Datasets** → `DATA_AND_DATASETS.md` - Schema, builders, conversion pipeline
- **Augmentation** → `DATA_AUGMENTATION.md` - Geometry transforms, telemetry, visualization hooks
- **Training** → `TRAINING_PLAYBOOK.md`
- **Inference & Stage-A** → `INFERENCE_AND_STAGEA.md`
- **Stage-B Runtime** → `STAGE_B_RUNTIME.md`
- **Reference overview** → `REFERENCE.md` - Architecture plus doc index
- **Stage-A & Stage-B (business)** → `STAGE_A_STAGE_B.md`, `stage-B-knowledge-Chinese.md`
- **Public datasets** → `PUBLIC_DATA.md`
- **Upstream Dependencies** → `UPSTREAM_DEPENDENCIES.md` - HF Qwen3-VL + ms-swift context
- **Experiments** → `docs/experiments/` - Training comparisons, ablations
- **Specs & governance** → `openspec/AGENTS.md`, `openspec/project.md`

### Documentation Ownership & Directory Map

| Directory | Primary doc(s) | Scope |
|-----------|----------------|-------|
| `src/` | `TRAINING_PLAYBOOK.md`, `INFERENCE_AND_STAGEA.md`, `REFERENCE.md` | Core training/inference implementation (`src/sft.py`, datasets, trainers). |
| `data_conversion/` | `DATA_AND_DATASETS.md` (Conversion section) | Unified processor for BBU annotations, taxonomy JSONs, resize/validation helpers. |
| `public_data/` | `PUBLIC_DATA.md` | LVIS and future auxiliary datasets (download, convert, sample, validate, visualize). |
| `vis_tools/` | `DATA_AUGMENTATION.md`, `vis_tools/README_CROP_VIS.md` | Visualization/debug scripts for augmentation, eval dumps, Qwen3-VL outputs. |
| `scripts/` | `TRAINING_PLAYBOOK.md`, `INFERENCE_AND_STAGEA.md`, `STAGE_B_RUNTIME.md` | Canonical entrypoints: training, inference, Stage-A/B launchers, dataset fusion. |
| `openspec/` | `openspec/AGENTS.md`, `openspec/project.md` | Change-management specs and proposal workflow. |

Whenever you add or modify code in the directories above, update the associated doc in the same PR to keep the handbook current.

### Script & Tooling Inventory

| Script | Location | Description |
|--------|----------|-------------|
| `train.sh` | `scripts/` | Conda-aware launcher for `python -m src.sft` / `torchrun` with config auto-resolution and debug toggles. |
| `fuse_datasets.py` | `scripts/` | Offline builder for `src/datasets/fusion.py` configs; pre-mixes BBU + auxiliary JSONL with deterministic ratios. |
| `download.py` | `scripts/` | Download helper for internal/raw corpora (mirrors instructions in `docs/DATA_AND_DATASETS.md`). |
| `stage_a_infer.sh` | `scripts/` | Mission-aware wrapper around `src.stage_a.cli` with guardrails for checkpoint/input directories. |
| `stage_b_run.sh` | `scripts/` | Stage-B reflection loop launcher; wires configs to `src.stage_b.runner`. |
| `run_grpo.py` | `scripts/` | Experimental Stage-B GRPO launcher (LLM-only LoRA fine-tuning with rewards in `src/stage_b/rewards.py`). |
| `inspect_lora_ckpts.py` | `scripts/` | Static inspector for adapter checkpoints to verify `modules_to_save` and LoRA coverage. |

Use these scripts instead of ad-hoc commands so telemetry, logging, and environment setup stay consistent across teams.

## Recent Updates

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
- Completeness field tracking: `显示完整` ↔ `只显示部分` updates
- Quad rotation fix + redundancy cleanup (removed CenterCrop, Equalize)
- See [CHANGELOG.md](CHANGELOG.md) for full details

---

## Major Change: Geometry Schema Overhaul (Nov 2025) 🚨

- **Big change**: the pipeline now publishes `poly` geometry entries everywhere (replacing the previous 4-point geometry key). Internally we still emit 4-point polygons today, but the schema and prompts are ready to hold arbitrary vertex counts going forward.
- All documentation, builders, and augmentation ops now expect `poly` (even-length list ≥8 values) as one of the three canonical geometry keys (`bbox_2d`, `poly`, `line`).
- This change affects data conversion, dataset builders, augmentation telemetry, Stage-A/B workflows, and training prompts. Please regenerate derived artifacts and re-validate dataset probes if you re-run conversion scripts.

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
| Data & Datasets | `src/datasets/`, `data_conversion/`, `src/datasets/data_details.md`, `scripts/fuse_datasets.py` |
| Augmentation | `src/datasets/augmentation/`, `src/datasets/geometry.py`, `vis_tools/` |
| Training Playbook & Advanced FAQ | `src/sft.py`, `scripts/train.sh`, `configs/`, `src/callbacks/`, `src/trainers/` |
| Inference & Stage-A | `src/stage_a/`, `scripts/stage_a_infer.sh`, `scripts/inspect_lora_ckpts.py`, `src/utils/logger.py` |
| Stage-B Runtime | `src/stage_b/`, `scripts/stage_b_run.sh`, `scripts/run_grpo.py`, `configs/stage_b/` |
| Public Data | `public_data/`, `public_data/scripts/`, `public_data/tests/` |
| Utils & Logging | `src/utils/`, `src/callbacks/` |

---

**Last Updated**: 2025-11-21 (Doc ownership refresh)

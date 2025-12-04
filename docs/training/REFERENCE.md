# Training & Inference Reference

Comprehensive guide for training, inference, deployment, and advanced topics.

**Source**: `src/sft.py`, `src/stage_a/`, `src/stage_b/`, `scripts/`, `configs/`

---

## Table of Contents

- [Architecture & Implementation](#architecture--implementation)
- [Training](#training)
- [Inference](#inference)
- [Advanced Topics & FAQ](#advanced-topics--faq)
- [Additional Resources](#additional-resources)

---

## Architecture & Implementation

### Inspection Pipeline (Stage‑1 → Stage‑2)
- **Stage‑1 / Stage‑A (Basic Object Recognition)**: `src/stage_a/` emits per-image evidence/rare-object summaries used as inputs to Stage‑2. Runbook: `docs/runtime/STAGE_A_RUNTIME.md`.
- **Stage‑2 / Stage‑B (Group Ticket Verification)**: `src/stage_b/` ingests Stage‑A JSONL + labels and returns `pass|fail` verdicts with **prompt-only rollouts** plus optional reflection updates. No CriticEngine; manual-review and failure queues live under each mission run dir. Runbook: `docs/runtime/STAGE_B_RUNTIME.md` and business context in `docs/runtime/STAGE_A_STAGE_B.md`.
- **Offline preprocessing (optional)**: `data_conversion/` normalizes annotation exports into train/val/tiny JSONL and QA reports. Guide: `docs/data/DATA_PREPROCESSING_PIPELINE.md`.

### Source Code Layout

**Training Pipeline**:
- `src/sft.py` - Main entry point, `SwiftSft` integration
- `src/config/` - YAML loading, config merging, `TrainArguments` assembly
- `src/README.md` - Source-level architecture documentation

**Dataset Components**:
- `src/datasets/dense_caption.py` - `DenseCaptionDataset` (mode selection, augmentation config)
- `src/datasets/builders/jsonlines.py` - `JSONLinesBuilder` (message formatting)
- `src/datasets/preprocessors/` - Validation, augmentation preprocessing
- `src/datasets/collators.py` - Tensor preparation (padding)
- Canonical schema doc: `docs/data/DATA_JSONL_CONTRACT.md`

**Geometry & Augmentation**:
- `src/datasets/geometry.py` - Core geometry transforms (bbox, poly, line)
- `src/datasets/augmentation/base.py` - `Compose` pipeline, `ImageAugmenter` protocol
- `src/datasets/augmentation/ops.py` - All augmentation operators (Rotate, Flip, Crop, etc.)

**Utilities & Infrastructure**:
- `src/utils/logger.py` - Rank-aware logging (DDP-safe)
- `src/utils/README.md` - Utilities documentation
- `src/callbacks/save_delay_callback.py` - `SaveDelayCallback` (checkpoint throttling)

### Doc ↔ Code Cross-References
- **Stage‑1 inference**: `src/stage_a/` ↔ `docs/runtime/STAGE_A_RUNTIME.md`
- **Stage‑2 verdict loop**: `src/stage_b/` ↔ `docs/runtime/STAGE_B_RUNTIME.md`, `docs/runtime/STAGE_A_STAGE_B.md`
- **Data preprocessing**: `data_conversion/` ↔ `docs/data/DATA_PREPROCESSING_PIPELINE.md`, `docs/data/DATA_AND_DATASETS.md` (conversion section)
- **Fusion dataset**: `src/datasets/unified_fusion_dataset.py` ↔ `docs/data/UNIFIED_FUSION_DATASET.md`
- **Training & config**: `src/sft.py`, `src/config/` ↔ `docs/training/TRAINING_PLAYBOOK.md`

### Key Components Deep Dive

**ConfigLoader** (`src/config/loader.py`):
- Loads YAML and materializes frozen dataclasses (`TrainingConfig`, `CustomConfig`, `SaveDelayConfig`, `VisualKDConfig`)
- Resolves `global_max_length` → `model.max_model_len` + `template.max_length`
- Attaches typed runtime toggles to `TrainArguments` (e.g., `save_delay_config`, `visual_kd_config`)
- Fails fast with informative errors when schemas or inheritance are invalid
- Source of truth for all configuration behavior
- Over-length policy: when `template.truncation_strategy` is set to `raise`, downstream datasets catch `MaxLengthError` and drop the offending sample instead of truncating, then retry another record (see `DenseCaptionDataset`).

**DenseCaptionDataset** (`src/datasets/dense_caption.py`):
```python
# What it does:
# 1. Wraps JSONL data with preprocessors
# 2. Selects dense vs summary mode per sample (epoch-seeded)
# 3. Configures augmentation pipeline (bypass_prob, ops)
# 4. Handles both train and validation splits
```

**JSONLinesBuilder** (`src/datasets/builders/jsonlines.py`):
```python
# What it does:
# 1. Formats single-image records → single-turn messages
# 2. User message: [image, prompt]
# 3. Assistant message: {"object_1": {...}, "object_2": {...}}
# 4. Attaches top-level "objects" with pixel coords (for template normalization)
# 5. Handles dense/summary modes differently
```

**Geometry Transforms** (`src/datasets/geometry.py`):
```python
# Core function: transform_geometry(geom, M, width, height)
# - Applies affine matrix M to bbox/poly/line
# - Clips to image bounds (with polygon/line clipping algorithms)
# - Preserves degenerate geometries (fallback to clamping)
# - Handles rotation without clipping for fully-inside polygons
```

**Augmentation Pipeline** (`src/datasets/augmentation/`):
```python
# Compose pipeline:
# 1. Accumulates affine ops (rotate, flip, scale) → single matrix M
# 2. Flushes on barriers (resize, crop, expand) → applies M
# 3. Color ops deferred (applied after geometric ops)
# 4. Propagates crop metadata (kept_indices, coverages)
```

### Token Flow Details

**Vision Token Insertion**:
```
Image (PIL)
  → Processor (resizes to multiple of patch_size)
  → Vision Encoder (ViT) → [batch, num_patches, hidden_dim]
  → Aligner (MLP projector) → [batch, num_tokens, llm_dim]
  → Replace <|image_pad|> placeholders in LLM input
```

**Aligner Components** (in Qwen3-VL model):
- `model.visual.merger` - Main MLP projector
- `model.visual.deepstack_merger_list.{0,1,2}` - Additional projection layers
- Located in `model.visual.*` (HuggingFace model structure)

**Chat Template Mechanics**:
- Template automatically inserts `<|image_pad|>` tokens
- Placeholder count = `image_grid_thw.prod()` per image
- Do NOT manually insert placeholders in text
- Template handles vision token expansion automatically

### Logging & Callbacks

**Rank-Aware Logging** (`src/utils/logger.py`):
```python
from src.utils.logger import get_logger

logger = get_logger("my_module")
logger.info("Rank 0 only")           # Only logged on main process
logger.debug("Debug info")            # Controlled by --debug flag
logger.warning("Warning (all ranks)") # Logged on all ranks if severe
```

**SaveDelayCallback** (`src/callbacks/save_delay_callback.py`):
```python
# Prevents early checkpoints before model has learned anything
# Config: custom.save_delay_steps / custom.save_delay_epochs (via SaveDelayConfig)
# Example: save_delay_steps: 100 → no saves until step 100
```

### Health Check Implementation

**Validation Points** (enforced in code):

1. **Image Placeholder Count** (`src/datasets/builders/jsonlines.py`):
   - User message image count matches `len(images)`
   - Template inserts correct number of `<|image_pad|>` tokens

2. **Grid Alignment** (`src/datasets/collators.py`):
   - `image_grid_thw` shape matches `pixel_values` dimensions
   - Each image has valid T×H×W grid

3. **Label Masking** (template encoding):
   - Image tokens in `input_ids` have `labels = -100`
   - Assistant tokens have `labels = input_ids[pos]`
   - User tokens have `labels = -100`

4. **Geometry Normalization** (`JSONLinesBuilder`):
   - Top-level `objects` kept in pixel space
   - Template normalizes `bbox_2d` to norm1000 during encoding
   - Assistant text uses `emit_norm` setting

### Extension Points

**Add New Preprocessor**:
1. Create `src/datasets/preprocessors/my_preprocessor.py`
2. Implement `BasePreprocessor` protocol
3. Register in `DenseCaptionDataset.__init__`

**Add New Augmentation Op**:
1. Add class to `src/datasets/augmentation/ops.py`
2. Implement `ImageAugmenter` protocol (`affine()` or `apply()`)
3. Use in YAML: `- name: my_op`

**Add New Builder**:
1. Create `src/datasets/builders/my_builder.py`
2. Implement builder protocol (`build_messages()`)
3. Configure in dataset initialization

### Hard-Sample Mining
- Deprecated as of 2025-11-27. Any config containing `custom.hard_sample_mining` will fail validation with guidance to remove the block and train with standard padded settings.

### Critical Implementation Details

**Adapter Preparation** (`src/sft.py`):
```python
# MUST be called before creating trainer
sft = SwiftSft(args)
sft.prepare_model()  # Configures LoRA, freezes, modules_to_save
trainer = sft.create_trainer()  # Now trainer has correct config
```

**Packing**: Removed. Training now uses padded batches only; any config enabling `training.packing` or packing knobs fails fast. Legacy implementation is archived under `archive/packing/`.

**Dataset-specific metrics (fusion)**: Padded batches carry `dataset_labels` from fusion metadata. The trainer logs per-dataset `*_loss` / `*_token_acc` during training for all datasets and skips source domains during eval when `dataset_domains` marks them (e.g., targets `bbu/rru` vs sources `lvis/lang_chat`). No extra config is required beyond fusion dataset wiring.

**Freeze Logic** (`src/sft.py` + `SwiftSft`):
```python
# freeze_llm: true → model.model.layers[*].requires_grad = False
# freeze_vit: true → model.visual.*.requires_grad = False
# freeze_aligner: false → model.visual.merger*.requires_grad = True
```

---

## Training

Core SFT/LoRA recipes, KL anchoring overlays, augmentation telemetry, and troubleshooting now live in [TRAINING_PLAYBOOK.md](TRAINING_PLAYBOOK.md). Use that document for:
- YAML scaffolding for single- and multi-stage training
- LoRA/freezing setups and SaveDelay guidance (packing removed)
- Telemetry expectations plus health-check checklists
- Advanced topics (LR schedulers, DeepSpeed configs, augmentation FAQs, common issues, aligner tuning)

**Token-type telemetry (optional)**  
- Config: `custom.token_type_metrics.enabled` (default `false`), `include` (default `['target','lvis']`), `exclude` (default `['coig_lang_chat']`).  
- Behavior: collator reconstructs assistant JSON, tokenizes with the active template, aligns token types (1=desc, 2=coord numbers, 3=format), and pads/truncates to supervised positions; rows outside `include` get IGNORE.  
- Metrics: per dataset label, logs `{label}_token_acc` (all supervised tokens, naturally weighted), type-sliced accuracies/entropy `{label}_{desc|coord|format}_token_acc|entropy`, plus token-count companions `{label}_token_count` and `{label}_{type}_token_count` to make weighting explicit.  
- Validation: smoke run on 2025-12-04 with `configs/smoke/group_metrics.yaml` (4B checkpoint, tiny fusion `configs/fusion/bbu_rru_lvis_coig_tiny.yaml`, `logging_steps=1`, `eval_steps=1`, `save_strategy=no`, `max_steps=20`) produced the expected token-type metrics; see `output/smoke/group_metrics/v0-20251204-062817/smoke_group_metrics_4b/logging.jsonl`.

Keep configs under `configs/` in sync with the playbook when making behavioral changes.

## Inference

Runtime/deployment instructions for Stage-A summaries and the Stage-B verdict loop live in [STAGE_B_RUNTIME.md](../runtime/STAGE_B_RUNTIME.md):
- Adapter vs merged checkpoints, export commands, and decoding tips
- Dense captioning usage examples
- Stage-A CLI guardrails and output schemas
- Stage-B sampler/selection/manual-review/reflection flow (prompt-only, no CriticEngine)

## Advanced Topics & FAQ

Operational FAQs (LR schedulers, DeepSpeed presets, augmentation pipelines, template mechanics, and troubleshooting) are consolidated into [TRAINING_PLAYBOOK.md](TRAINING_PLAYBOOK.md). Refer to that doc whenever you tweak trainers, infrastructure knobs, or template logic.

## Additional Resources

- **Training**: [TRAINING_PLAYBOOK.md](TRAINING_PLAYBOOK.md)
- **Stage-A & Stage-B runtime**: [STAGE_B_RUNTIME.md](../runtime/STAGE_B_RUNTIME.md)
- **Data preprocessing & contract**: [DATA_PREPROCESSING_PIPELINE.md](DATA_PREPROCESSING_PIPELINE.md), [DATA_JSONL_CONTRACT.md](DATA_JSONL_CONTRACT.md)
- **Data formats & augmentation**: [DATA_AND_DATASETS.md](DATA_AND_DATASETS.md), [DATA_AUGMENTATION.md](DATA_AUGMENTATION.md)
- **Archived docs**: `docs/archive/` (historical references, detailed technical guides)

---

**Last Updated**: 2025-12-04 (Token-type telemetry validated via smoke run)

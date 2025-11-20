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

### Source Code Layout

**Training Pipeline**:
- `src/sft.py` - Main entry point, `SwiftSft` integration
- `src/config/` - YAML loading, config merging, `TrainArguments` assembly
- `src/README.md` - Source-level architecture documentation

**Dataset Components**:
- `src/datasets/dense_caption.py` - `DenseCaptionDataset` (mode selection, augmentation config)
- `src/datasets/builders/jsonlines.py` - `JSONLinesBuilder` (message formatting)
- `src/datasets/preprocessors/` - Validation, augmentation preprocessing
- `src/datasets/collators.py` - Tensor preparation, packing logic
- `src/datasets/data_details.md` - Data schema documentation

**Geometry & Augmentation**:
- `src/datasets/geometry.py` - Core geometry transforms (bbox, poly, line)
- `src/datasets/augmentation/base.py` - `Compose` pipeline, `ImageAugmenter` protocol
- `src/datasets/augmentation/ops.py` - All augmentation operators (Rotate, Flip, Crop, etc.)

**Utilities & Infrastructure**:
- `src/utils/logger.py` - Rank-aware logging (DDP-safe)
- `src/utils/README.md` - Utilities documentation
- `src/callbacks/save_delay_callback.py` - `SaveDelayCallback` (checkpoint throttling)

### Key Components Deep Dive

**ConfigLoader** (`src/config/loader.py`):
- Loads YAML and materializes frozen dataclasses (`TrainingConfig`, `CustomConfig`, `SaveDelayConfig`, `VisualKDConfig`)
- Resolves `global_max_length` → `model.max_model_len` + `template.max_length`
- Attaches typed runtime toggles to `TrainArguments` (e.g., `save_delay_config`, `visual_kd_config`)
- Fails fast with informative errors when schemas or inheritance are invalid
- Source of truth for all configuration behavior

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

### Critical Implementation Details

**Adapter Preparation** (`src/sft.py`):
```python
# MUST be called before creating trainer
sft = SwiftSft(args)
sft.prepare_model()  # Configures LoRA, freezes, modules_to_save
trainer = sft.create_trainer()  # Now trainer has correct config
```

**Packing Implementation**:
- Enabled via `training.packing: true`
- Collator concatenates samples to `max_length`
- Requires Flash Attention 2+ (Qwen3-VL native)
- Incompatible with `lazy_tokenize`

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
- LoRA/freezing setups, packing, and SaveDelay guidance
- Telemetry expectations plus health-check checklists
- Advanced topics (LR schedulers, DeepSpeed configs, augmentation FAQs, common issues, aligner tuning)

Keep configs under `configs/` in sync with the playbook when making behavioral changes.

## Inference

All deployment instructions moved to [INFERENCE_AND_STAGEA.md](INFERENCE_AND_STAGEA.md):
- Adapter vs merged checkpoints, export commands, and decoding tips
- Dense captioning usage examples
- Stage-A CLI guardrails and output schemas
- Additional Stage-A implementation notes and dense/summary mixed-mode design

For Stage-B rollout details (sampler config, critic/manual review, reflection flow, and GRPO experiments) see [STAGE_B_RUNTIME.md](STAGE_B_RUNTIME.md).

## Advanced Topics & FAQ

Operational FAQs (LR schedulers, DeepSpeed presets, augmentation pipelines, template mechanics, and troubleshooting) are consolidated into [TRAINING_PLAYBOOK.md](TRAINING_PLAYBOOK.md). Refer to that doc whenever you tweak trainers, infrastructure knobs, or template logic.

## Additional Resources

- **Training**: [TRAINING_PLAYBOOK.md](TRAINING_PLAYBOOK.md)
- **Inference & Stage-A**: [INFERENCE_AND_STAGEA.md](INFERENCE_AND_STAGEA.md)
- **Stage-B Runtime**: [STAGE_B_RUNTIME.md](STAGE_B_RUNTIME.md)
- **Data formats & augmentation**: [DATA_AND_DATASETS.md](DATA_AND_DATASETS.md), [DATA_AUGMENTATION.md](DATA_AUGMENTATION.md)
- **Archived docs**: `docs/archive/` (historical references, detailed technical guides)

---

**Last Updated**: 2025-11-21 (Doc split)

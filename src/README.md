## Overview

Modular, YAML-driven pipeline for fine-tuning Qwen3-VL on dense captioning tasks with structured geometry annotations (bbox/poly/line).

**Key Features**:
- **Composable**: Pluggable preprocessors, builders, augmentation strategies
- **Reproducible**: Epoch-based seeding for dynamic multi-image pairing
- **Geometry-aware**: Affine transformations preserve spatial accuracy
- **JSON-lines first**: Grouped JSON output with geometry preserved

**Pipeline**: YAML Config → ConfigLoader → SwiftSft → BaseCaptionDataset (alias DenseCaptionDataset) → Training Loop

## Table of Contents
- [Overview](#overview)
- [Architecture at a Glance](#architecture-at-a-glance)
- [Directory Structure](#directory-structure)
- [Core Workflow](#core-workflow)
  - [Runner specifics (sft.py)](#runner-specifics-sftpy)
  - [Required custom.* keys](#required-custom-keys)
- [Data Contract (JSONL)](#data-contract-jsonl)
  - [Per-Sample Pipeline](#per-sample-pipeline)
  - [Geometry Handling](#geometry-handling)
  - [Message Formats](#message-formats)
- [Configuration (YAML)](#yaml-structure)
  - [Config inheritance rules](#config-inheritance-rules)
  - [Two-stage training (recommended)](#two-stage-training-recommended)
  - [Mixed-mode: LoRA on LLM and Vision, full-tune Aligner](#mixed-mode-lora-on-llm-and-vision-full-tune-aligner)
  - [Dynamic Per-Group Prompt Selection](#dynamic-per-group-prompt-selection-optional)
- [Inference and Adapter Management](#inference-and-adapter-management)
- [Utilities](#utilities)
- [Dynamic grouping & augmentation (dataset)](#dynamic-grouping--augmentation-dataset)
- [Health checks (fail-fast)](#health-checks-fail-fast)
- [Troubleshooting](#troubleshooting)
- [Quick Start](#quick-start)

## Architecture at a Glance

- YAML config is the single source of truth → `ConfigLoader` merges/validates and instantiates ms-swift `TrainArguments`.
- `SwiftSft` initializes model and template (uses the model's native chat_template); adapters applied via `sft.prepare_model(...)`.
- `BaseCaptionDataset` (alias DenseCaptionDataset) orchestrates preprocessing, mode selection, and `JSONLinesBuilder` per sample (no pairing).
- Template encodes messages, adds vision tokens, and normalizes coordinates (top-level objects → norm1000) during encoding.
- Trainer from ms-swift runs training; checkpoints save adapters (LoRA) or merged weights depending on workflow.

## Directory Structure

```
src/
├── config/                  # YAML config loading & prompt management
│   ├── loader.py           # ConfigLoader: merge configs, resolve prompts
│   └── prompts.py          # Centralized prompt templates
├── datasets/
│   ├── preprocessors/      # Row-level transformations
│   │   ├── base.py        # BasePreprocessor interface
│   │   ├── dense_caption.py  # Validation & filtering
│   │   └── augmentation.py   # Geometry-aware augmentation
│   ├── dense_caption.py   # BaseCaptionDataset (single-image orchestration)
│   ├── builders/           # Message format builders
│   │   ├── base.py        # BaseBuilder interface
│   │   └── jsonlines.py   # Minimal object-hierarchy output
│   ├── utils.py           # load_jsonl, extract_geometry
│   ├── geometry.py        # Affine transforms, normalization
│   ├── augment.py         # Image + geometry augmentation
│   ├── collators.py       # Data collators (padding-free, etc.)
│   └── data_details.md    # JSONL schema specification
├── utils/
│   └── auto_detect_aligners.py  # Utility to list aligner modules
├── sft.py                  # Training entry point (YAML-driven)
```


## Core Workflow

### Configuration & Setup
**Modules**: `config/loader.py`, `sft.py`

1. **ConfigLoader** loads YAML, resolves `extends`/`inherit` chains, merges base/experiment configs, resolves prompts → `TrainArguments`
2. **SwiftSft** initializes model, template, trainer with config
3. **BaseCaptionDataset** constructs train/eval datasets with selected builder and augmentation

### Runner specifics (sft.py)
- Pure YAML-driven: CLI only accepts `--config`, optional `--base_config`, and `--debug`.
- Inherits and merges configs via `extends`/`inherit`; last wins, cycles fail fast.
- Auto-sets `ROOT_IMAGE_DIR` to the directory of `custom.train_jsonl` when not provided.
- Applies tuner/adapters with `sft.prepare_model(...)` before trainer creation.
- Supports optional sample limiting: `custom.sample_limit`, `custom.train_sample_limit`, `custom.val_sample_limit`.

### Required custom.* keys
- **custom.user_prompt**: string used by `JSONLinesBuilder` for the user turn.
- **custom.emit_norm**: one of `none|norm100|norm1000`; affects text geometry only.
  - Top-level `objects.bbox` remains in pixel space and is normalized by the template at encode time.

### LoRA Adapter Preparation (Critical!)
**⚠️ REQUIRED for custom training scripts**

When writing custom training scripts (outside of `SwiftSft`), you **must** call `sft.prepare_model()` before creating the `Trainer`:

```python
from swift.llm import SwiftSft, TrainArguments

# Load config
train_args = TrainArguments.from_config(...)

# Initialize SFT pipeline
sft = SwiftSft(train_args)

# ⚠️ CRITICAL: Apply LoRA adapter before creating trainer
# Without this call, the model will NOT be wrapped by SwiftModel/PeftModel
# and full model weights will be saved instead of adapter weights!
sft.model = sft.prepare_model(train_args, sft.model, template=sft.template, train_dataset=dataset)

# Now create trainer
trainer = Trainer(model=sft.model, ...)
```

**What `prepare_model()` does**:
1. **Freezes modules** based on `freeze_llm`, `freeze_vit`, `freeze_aligner`
2. **Applies LoRA adapters** to `target_modules` (e.g., `all-linear`)
3. **Marks `modules_to_save`** for full fine-tuning (e.g., aligner MLP layers)
4. **Wraps model** in `SwiftModel` or `PeftModel` (depending on backend)
5. **Enables adapter saving** instead of full checkpoint

**Without this call**:
- ❌ Model remains unwrapped (raw `Qwen3VLModel`)
- ❌ LoRA not applied; all parameters trainable (OOM risk)
- ❌ Checkpoint saves ~9.6GB full weights instead of ~240MB adapter
- ❌ `adapter_config.json` missing or empty `modules_to_save`

**Verification**:
```python
# After prepare_model, check model type
print(f"Model type: {type(sft.model).__name__}")
# Expected: SwiftModel or PeftModel (not Qwen3VLModel)

# Check if adapter config exists after training
import json
adapter_cfg = json.load(open("checkpoint/adapter_config.json"))
print(adapter_cfg["modules_to_save"])  # Should list your aligner modules
```

## Data Contract (JSONL)
JSONL records (see `data_details.md`):
- `images`: List[str] — paths resolved via `ROOT_IMAGE_DIR`
- `objects`: List — each has one geometry (`bbox_2d`/`poly`/`line`) + `desc`
- `width`, `height`: image dimensions
- `summary`: **standardized all-slash format** (required for summary modes, optional otherwise)

### Per-Sample Pipeline
**Executed in `DenseCaptionDataset.__getitem__(index)`**

```
Index → Epoch-seeded permutation
     → Deep copy single record
     → Preprocessing (optional augmentation)
     → Mode Selection (dense or summary)
     → Message Building (JSONLines, minimal object map)
     → Template Encoding (tokenization, bbox norm1000)
     → Training Sample
```

**Step Details**:
1. **Record Selection**: Epoch-based permutation ensures deterministic shuffling across workers while staying single-image.
2. **Preprocessing**: `AugmentationPreprocessor` applies affine transforms to images + geometries atomically.
3. **Mode Selection**: `custom.use_summary` sets the default (dense/summary); fusion configs may override per target/source via `mode: dense|summary`.
4. **Message Building**:
   - `JSONLinesBuilder`: User prompt embeds the image; assistant returns `{ "object_1": {...}, ... }` (no per-image wrapper).
   - Summary mode yields a single formatted string. Top-level `objects` retain exact point arrays for template normalization.
5. **Template Encoding**: ms-swift adds `<image>` tokens, normalizes bbox to norm1000, and tokenizes text.

### Processing Stages (concise)

1) JSONL record loaded (images, objects, width/height, optional summary)
2) Optional preprocessing (augmentation, validation)
3) Mode select (dense or summary) in `DenseCaptionDataset` → instantiate `JSONLinesBuilder(mode=...)`
4) Builder assembles one-turn chat: user embeds image + prompt; assistant returns minimal object hierarchy or summary string
5) Template encodes: inserts vision tokens, normalizes top-level `objects.bbox` to norm1000, tokenizes text
6) DataLoader yields tensors: `input_ids`, `attention_mask`, `labels`, `pixel_values`, `image_grid_thw`, `objects`

**Key Transformations**:
- **Geometry**: Exact `poly`/`line`/`bbox_2d` point arrays are preserved and used for grounding.
- **Coordinates**: Original pixel → norm1000 (based on original dims; no runtime resizing in training path).
- **Text**: Original geometries preserved in JSON-lines (training target); `emit_norm` affects text only.
- **Images**: No HF smart-resize in training path (`do_resize=false`).
- **Tokens**: `<|image_pad|>` expanded to match vision token count

### Geometry Handling
**Modules**: `datasets/builders/jsonlines.py`, `datasets/geometry.py`, `datasets/augment.py`

- Supported types: `bbox_2d` (4), `poly` (even-length list, currently 8), `line` (2N)
- Default: exact points preserved in top-level `objects.bbox`; template scales any even-length list to norm1000.
- Augmentation: affine transforms update points atomically; text unchanged; spatial accuracy preserved.
- Useful ops: `normalize_points()`, `apply_affine()`

### Message Formats

**JSONLinesBuilder**（单图输出）:
```json
{
  "object_1": {
    "poly": [x1,y1,...,x4,y4],
    "desc": "BBU设备/华为/完整/..."
  },
  "object_2": {
    "line_points": 4,
    "line": [x1,y1,...,x4,y4],
    "desc": "光纤/有保护/..."
  }
}
```

**User turn**: `content` 先嵌入单张图片，再追加文本指令；无需额外的编号标签。

**⚠️ CRITICAL: ms-swift Image Format Convention**

ms-swift uses a **strict key-value convention** for multimodal content where the **value key must match the type key**:

```python
# ✅ CORRECT (ms-swift convention)
{"type": "image", "image": "<path/url/PIL.Image>"}
{"type": "audio", "audio": "<path>"}
{"type": "video", "video": "<path>"}

# ❌ WRONG (OpenAI-style; will silently fail)
{"type": "image", "url": "<path>"}         # ← image key missing!
{"type": "image", "image_url": "<path>"}   # ← wrong key name
```

**Why this matters:**
- ms-swift extracts media via `item.get(item['type'])` (e.g., `item.get('image')`)
- If you use `"url"` instead of `"image"`, extraction returns `None`
- Template will **not load images** → no `pixel_values`/`image_grid_thw` in tensors
- Vision encoder + aligner **never execute** → **zero gradients** for vision components
- Training appears to run but vision modules don't learn anything

**Where to check:**
- Message builders (`src/datasets/builders/*.py`): Ensure `{"type": "image", "image": path}`
- Custom preprocessing: Verify content dict keys match the type
- Debug: Print `sample.keys()` after encoding; must include `pixel_values` and `image_grid_thw`

**Reference:** See `/data/ms-swift/swift/llm/template/template_inputs.py:241` for extraction logic.

**Configuration**:
- `global_max_length`: Single knob for full conversation length (prompt + completion); overrides `model.max_model_len` and `template.max_length`
- `emit_norm`: 控制文本输出的坐标空间（none/norm100/norm1000）
输出直接使用单图对象映射，`group_key_prefix` 配置项保持移除。
- 顶层 `objects.ref/bbox/image_id` 保留为原始像素坐标，模板自动归一化为 norm1000
- 不再有 section headers 或 `image_index` 字段
- **Packing**: Removed. Training now always uses standard padding; `training.packing` and related knobs are rejected. Legacy code lives in `archive/packing/` for reference.

**Dual Representation Strategy**:
1. **Assistant 文本**: 使用 object-index JSON（`object_{n}`），几何字段直接暴露（bbox_2d/poly/line）。
2. **顶层 objects**: 精确像素坐标供模板在编码阶段转换为 norm1000。
3. 增广后的几何与文本保持一致。


## Architecture Principles

**Separation of Concerns**: Data (JSONL) → Preprocessing (row-level) → Building (pair-level) → Encoding (template-level)

**Pluggability**: Abstract base classes enable swapping preprocessors, builders, and pair selectors independently

**Reproducibility**: Epoch-based seeding + config-driven (zero CLI hyperparameters)

**Geometry Preservation**: Affine transforms maintain validity; text unchanged during augmentation; pixel xyxy preserved for template normalization

### Key Components

| Component | Role | Examples |
|-----------|------|----------|
| **Preprocessors** | Row-level transformations | `AugmentationPreprocessor`, `DenseCaptionPreprocessor` |
| **Builders** | Pair → message conversion | `JSONLinesBuilder` |
| **ConfigLoader** | YAML management | Merge configs, resolve prompts → `TrainArguments` |
| **DenseCaptionDataset** | High-level wrapper | Selects builder, configures augmentation |


## Quick start

```bash
python -m src.sft --config /abs/path/to/your_config.yaml
```
For config structure and inheritance, see the YAML examples below and `docs/MS_SWIFT_TRAINING_GUIDE.md`.

**YAML Structure** (explicit values required; you can factor shared fields into `configs/base.yaml`):
```yaml
# Inheritance (optional)
extends: base.yaml            # or a list: ["base.yaml", "more.yaml"]

model:
  model: path/to/Qwen3-VL-4B-Instruct

template:
  template: qwen3_vl             # use model's chat_template.json (vision tokens handled automatically)
  max_length: 4096
  truncation_strategy: raise     # fail fast if a sample reaches max_length
  max_pixels: 589824             # e.g., up to 1024x1024

training:
  num_train_epochs: 3
  per_device_train_batch_size: 2
  learning_rate: 1e-4

custom:
  train_jsonl: data/bbu_full_768/train.jsonl
  val_jsonl: data/bbu_full_768/val.jsonl
  emit_norm: norm1000               # none | norm100 | norm1000
  # 无需配置 group_key_prefix；输出已是单图 object_{n} 结构

# prompts section is no longer required - default system prompt is used automatically
# Custom prompts can be specified via:
# prompts:
#   system: |
#     你是图像密集标注助手。只返回原始 JSON-lines…
#     输出仅包含 object_{n} 键，无需在文本中手动分段
#   user: 描述所有对象
```

### Config inheritance rules

- Top-level keys: `extends` or `inherit` (alias). Accepts a string or list.
- Paths are resolved relative to the current YAML file. Absolute paths also work.
- Merge order: earlier bases have lower precedence; current file wins.
- Cycles are detected and raise an error.

### Two-stage training (recommended)

- Stage 1 (Aligner-only LoRA): start from base model, freeze LLM+ViT → checkpoint A
  ```yaml
  tuner:
    train_type: lora
    target_modules: [all-linear]
    freeze_llm: true
    freeze_vit: true
    freeze_aligner: false
  ```
- Stage 2 (LLM+Aligner LoRA): start from checkpoint A, freeze ViT → checkpoint B
  ```yaml
  model:
    model: path/to/checkpoint-A
  tuner:
    train_type: lora
    target_modules: [all-linear]
    freeze_llm: false
    freeze_vit: true
    freeze_aligner: false
  ```
Deploy checkpoint B for inference with the base processor/template.

### Mixed-mode: LoRA on LLM and Vision, full-tune Aligner

To apply LoRA to the language model and vision tower while fully fine-tuning the aligner (no LoRA on the aligner), use:

```yaml
tuner:
  train_type: lora
  target_modules: [all-linear]
  freeze_llm: false        # LoRA on LLM
  freeze_vit: false        # LoRA on Vision
  freeze_aligner: true     # do NOT inject LoRA into aligner
  modules_to_save:
    - model.visual.merger
    - model.visual.deepstack_merger_list.0
    - model.visual.deepstack_merger_list.1
    - model.visual.deepstack_merger_list.2
```

Tip: keep `training.aligner_lr` (and optionally `training.vit_lr`) to control per-module learning rates via the multimodal optimizer.

### Summary Output Toggle

**Feature**: Switch between dense JSON captions and single-line summaries. `custom.use_summary` sets the default mode, and fusion configs can override per target/source with `mode: dense|summary`, enabling mixed summary+dense sampling.

**Supported modes**:
- **Dense default**: grouped JSON with geometry + description (`custom.use_summary: false` or omitted); fusion datasets inherit unless overridden.
- **Summary**: single-line summaries (`custom.use_summary: true` or fusion dataset `mode: summary`).

**Usage**:
```yaml
# Dense default (non-fusion or fusion fallback)
custom:
  train_jsonl: data/bbu_full_768/train.jsonl
  use_summary: false
```

```yaml
# Summary default (all datasets unless fusion overrides)
custom:
  train_jsonl: data/bbu_full_768/train.jsonl
  use_summary: true
```

```yaml
# Fusion mixed modes (summary target, dense source)
custom:
  fusion_config: configs/fusion/bbu_rru_lvis_coig.yaml
  use_summary: false   # default dense; target overrides below
...
targets:
  - dataset: bbu
    train_jsonl: /abs/path/to/summary_target.jsonl
    mode: summary
sources:
  - dataset: coco
    train_jsonl: /abs/path/to/coco.jsonl
    ratio: 0.1
    mode: dense
```

For details, see `docs/data/DATA_JSONL_CONTRACT.md#top-level-record`.


### Further reading

- **Training workflows**: See `docs/TRAINING_GUIDE.md` for complete training guide
- **Inference & deployment**: See `docs/INFERENCE_GUIDE.md` for inference and adapter merging
- **Data preparation**: See `docs/data/DATA_JSONL_CONTRACT.md` for JSONL schemas and validation
- **Advanced topics**: See `docs/training/REFERENCE.md` for performance tuning and troubleshooting

## Utilities

```bash
# List aligner modules present in a model/checkpoint (helps set modules_to_save)
python -m src.utils.auto_detect_aligners Qwen/Qwen3-VL-4B-Instruct
python -m src.utils.auto_detect_aligners output/stage_1_full_aligner_only/best/checkpoint-200

# Merge LoRA adapter into base for deployment (recommended)
CUDA_VISIBLE_DEVICES=0 conda run -n ms swift export \
  --model /abs/path/to/base/Qwen3-VL \
  --adapters /abs/path/to/output/stage_2/checkpoint-XXX \
  --merge_lora true \
  --output_dir /abs/path/to/output/merged/checkpoint-XXX \
  --safe_serialization true \
  --max_shard_size 5GB
```

## Dataset knobs

- use_summary: set default mode (dense/summary) for runs; fusion datasets can override with per-dataset `mode`
- augmentation.*: geometry-aware augmentation pipeline (Compose config)
- dump_conversation_text: write one decoded conversation sample to disk for inspection

These live under the `custom` section in YAML and are consumed by `src/sft.py` and `datasets/dense_caption.py`.

## Health checks (fail-fast)
- **Chat template & images**: User turn image count must match placeholders; mismatches error.
- **Vision tokens**: `image_grid_thw` aligns with `pixel_values` and placeholder expansion.
- **Assistant spans**: End-of-turn included; non-target tokens labeled −100; image tokens masked.
- **Geometry**: Top-level `objects.bbox` are normalized to norm1000 during encoding; emit_norm only affects assistant text.
- **Config**: Missing required `custom.*` keys or cyclic `extends` raise errors; invalid DeepSpeed flags fail early.

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| **Zero gradient norm (grad_norm=0) for vision/aligner in full training** | **Wrong image key in message content**: using `"url"` instead of `"image"` | ms-swift expects `{"type": "image", "image": path}` not `{"type": "image", "url": path}`. Fix in message builder (e.g., `jsonlines.py` lines 50, 88). See "Message Formats" section for details. |
| **Missing `pixel_values`/`image_grid_thw` in encoded samples** | Same as above: image key mismatch | Template can't extract images → no vision tensors → aligner never used. Verify builder uses `"image"` key; debug with `print(sample.keys())` after `dataset[0]` |
| **Full model saved instead of LoRA adapter** | Missing `sft.prepare_model()` call | See "LoRA Adapter Preparation (Critical!)" section; must call before `Trainer()` |
| **`adapter_config.json` missing or `modules_to_save` empty** | LoRA not applied to model | Verify model is wrapped: `isinstance(model, (SwiftModel, PeftModel))` after `prepare_model()` |
| **`TypeError: modules_to_save cannot be applied to ModuleList`** | Invalid `modules_to_save` path | Specify individual elements: `model.visual.deepstack_merger_list.0`, `.1`, `.2` instead of the container |
| **FileNotFoundError** | Image paths must be relative | Runner auto-sets `ROOT_IMAGE_DIR` to JSONL dir; verify paths exist |
| **MaxLengthError/OOM** | Long JSON-lines or many objects | Increase `global_max_length` (single knob) or lower `template.max_length`; keep `truncation_strategy=raise` to avoid truncation |
| **Points misalignment** | Augmentation bug | `AugmentationPreprocessor` updates images+geometries atomically; print sample to verify |
| **Memory/performance** | Suboptimal settings | Use `attn_impl=flash_attn`, `torch_dtype=bfloat16`, `gradient_checkpointing=true`; adjust batch size; configure DeepSpeed for multi-GPU |

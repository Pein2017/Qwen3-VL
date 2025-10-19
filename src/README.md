## Overview

Modular, YAML-driven pipeline for fine-tuning Qwen3-VL on dense captioning tasks with structured geometry annotations (bbox/quad/line).

**Key Features**:
- **Composable**: Pluggable preprocessors, builders, augmentation strategies
- **Reproducible**: Epoch-based seeding for dynamic multi-image pairing
- **Geometry-aware**: Affine transformations preserve spatial accuracy
- **Format-flexible**: JSON-lines, readable, or simple output modes

**Pipeline**: YAML Config → ConfigLoader → SwiftSft → DenseCaptionDataset → DynamicPairDataset → Training Loop


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
│   ├── builders/           # Message format builders
│   │   ├── base.py        # BaseBuilder interface
│   │   ├── jsonlines.py   # Structured JSON-lines output
│   │   └── readable.py    # Human-readable formats
│   ├── utils.py           # load_jsonl, extract_geometry
│   ├── geometry.py        # Affine transforms, normalization
│   ├── augment.py         # Image + geometry augmentation
│   ├── dynamic_pair.py    # DynamicPairDataset (core engine)
│   └── dense_caption.py   # DenseCaptionDataset (wrapper)
├── sft.py                  # Training entry point (YAML-driven)
└── data_details.md         # JSONL schema specification
```


## Core Workflow

### Configuration & Setup
**Modules**: `config/loader.py`, `sft.py`

1. **ConfigLoader** loads YAML, resolves `extends`/`inherit` chains, merges base/experiment configs, resolves prompts → `TrainArguments`
2. **SwiftSft** initializes model, template, trainer with config
3. **DenseCaptionDataset** constructs train/eval datasets with selected builder and augmentation

### Data Format
JSONL records (see `data_details.md`):
- `images`: List[str] — paths resolved via `ROOT_IMAGE_DIR`
- `objects`: List — each has one geometry (`bbox_2d`/`quad`/`line`) + `desc`
- `width`, `height`: image dimensions
- `summary`: optional per-image summary

### Per-Sample Pipeline
**Executed in `DynamicPairDataset.__getitem__(index)`**

```
Index → Pair Selection (epoch-seeded RNG) 
     → Deep Copy (rec_a, rec_b)
     → Preprocessing (optional augmentation)
     → Message Building (JSONLines/Readable/Simple)
     → Template Encoding (tokenization, bbox norm1000)
     → Training Sample
```

**Step Details**:
1. **Pair Selection**: Random pairing with epoch-based seeding for reproducibility
2. **Preprocessing**: `AugmentationPreprocessor` applies affine transforms to images + geometries atomically
3. **Message Building**: 
   - `JSONLinesBuilder`: User prompt embeds all images; assistant returns grouped JSON per 图片_N
   - Creates top-level `objects` with exact point arrays for template normalization
4. **Template Encoding**: ms-swift adds `<image>` tokens, normalizes bbox to norm1000, tokenizes

### Complete Data Transformation Flow

From raw JSONL to model tensors:

```python
# ═══════════════════════════════════════════════════════════════
# STAGE 1: Raw JSONL Records (Your Dataset)
# ═══════════════════════════════════════════════════════════════
record = {
    "images": ["images/example1.jpeg"],
    "objects": [
        {"quad": [10, 100, 200, 110, 190, 300, 5, 290], 
         "desc": "BBU设备/华为,显示完整,机柜空间充足需要安装/这个BBU设备按要求配备了挡风板"},
        {"line": [50, 150, 100, 200, 150, 250], 
         "desc": "光纤/有保护措施,弯曲半径合理/蛇形管"},
        {"bbox_2d": [80, 400, 120, 450], 
         "desc": "螺丝、光纤插头/BBU安装螺丝,显示完整,符合要求"}
    ],
    "width": 420,
    "height": 896,
    "summary": "BBU设备×1，光纤×1，螺丝、光纤插头×1"
}

# ═══════════════════════════════════════════════════════════════
# STAGE 2: After Pairing (DynamicPairDataset)
# ═══════════════════════════════════════════════════════════════
# Pairs record_a with record_b (may be same for single-image training)
# After optional augmentation (geometries transformed, types preserved)

# ═══════════════════════════════════════════════════════════════
# STAGE 3: After Message Building (JSONLinesBuilder)
# ═══════════════════════════════════════════════════════════════
{
    "messages": [
        {
            "role": "user",
            "content": "describe all objects"
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": (
                        '{"section": "image_1"}\n'
                        '{"type": "BBU设备", "geometry": {"quad": [10,100,200,110,190,300,5,290]}, '
                        '"desc": "BBU设备/华为,显示完整,机柜空间充足需要安装/这个BBU设备按要求配备了挡风板"}\n'
                        '{"type": "光纤", "geometry": {"line": [50,150,100,200,150,250]}, '
                        '"desc": "光纤/有保护措施,弯曲半径合理/蛇形管"}\n'
                        '{"type": "螺丝、光纤插头", "geometry": {"bbox_2d": [80,400,120,450]}, '
                        '"desc": "螺丝、光纤插头/BBU安装螺丝,显示完整,符合要求"}\n'
                        '{"summary": "BBU设备×1，光纤×1，螺丝、光纤插头×1"}'
                    )
                }
            ]
        }
    ],
    "images": ["images/example1.jpeg"],
    "objects": {
        "ref": ["BBU设备", "光纤", "螺丝、光纤插头"],
        "bbox": [
            [10,100,200,110,190,300,5,290],   # quad exact points
            [50,150,100,200,150,250],         # line exact points
            [80,400,120,450]                   # bbox_2d
        ],
        "image_id": [0, 0, 0]
    }
}

# ═══════════════════════════════════════════════════════════════
# STAGE 4: After Template Preprocessing (base.py)
# ═══════════════════════════════════════════════════════════════
# Template records original width/height, loads images
# objects.width = [420], objects.height = [896]

# ═══════════════════════════════════════════════════════════════
# STAGE 5: Processor Call (Qwen3VLProcessor)
# ═══════════════════════════════════════════════════════════════
# No runtime smart-resize during training (do_resize=False)

# Text with image tokens:
text = "<|image_pad|>describe all objects"

# After processor calculates grid and replaces tokens:
text_with_tokens = (
    "<|vision_start|><|image_pad|><|image_pad|>...<|image_pad|><|vision_end|>"  # N tokens
    "describe all objects"
)

# ═══════════════════════════════════════════════════════════════
# STAGE 6: Points Normalization (template.normalize_bbox)
# ═══════════════════════════════════════════════════════════════
# Converts pixel coords to norm1000 based on ORIGINAL dimensions
for pts in objects['bbox']:
    for i, (x, y) in enumerate(zip(pts[::2], pts[1::2])):
        pts[2*i]   = round(x / 420 * 1000)
        pts[2*i+1] = round(y / 896 * 1000)

# objects.bbox now in [0, 1000] space

# ═══════════════════════════════════════════════════════════════
# STAGE 7: Tokenization (tokenizer)
# ═══════════════════════════════════════════════════════════════
# Text → token IDs
input_ids = [151644, 151649, ...]  # <|image_pad|> tokens + text tokens

# Assistant text → labels
labels = [-100, -100, ..., 49360, 784, ...]  # -100 for input, token IDs for target

# ═══════════════════════════════════════════════════════════════
# STAGE 8: Final Batch (DataLoader)
# ═══════════════════════════════════════════════════════════════
{
    "input_ids": torch.tensor([[151644, 151649, ...]]),        # [batch, seq_len]
    "attention_mask": torch.tensor([[1, 1, 1, ...]]),          # [batch, seq_len]
    "labels": torch.tensor([[-100, -100, ..., 49360, ...]]),   # [batch, seq_len]
    "pixel_values": torch.tensor([[[[...]]],                   # [batch, channels, temporal, H, W]
    "image_grid_thw": torch.tensor([[1, 34, 16]]),            # [num_images, 3] (t, h, w grids)
    "objects": {
        "ref": ["BBU设备", "光纤", "螺丝、光纤插头"],
        "bbox": [
            [24,112,476,123,452,335,12,323],   # quad → norm1000 (example)
            [119,167,238,223,357,279],         # line → norm1000
            [190,446,286,502]                   # bbox_2d → norm1000
        ],
        "image_id": [0, 0, 0]
    }
}

# ═══════════════════════════════════════════════════════════════
# STAGE 9: Model Forward Pass
# ═══════════════════════════════════════════════════════════════
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    pixel_values=pixel_values,
    image_grid_thw=image_grid_thw,
    labels=labels  # For training loss
)

# Model internally:
# 1. Vision encoder processes pixel_values using image_grid_thw
# 2. Embeds grounding boxes from objects.bbox (norm1000 space)
# 3. Language model attends to vision tokens + text tokens
# 4. Computes cross-entropy loss between logits and labels
```

**Key Transformations**:
- **Geometry**: Exact `quad`/`line`/`bbox_2d` point arrays are preserved and used for grounding.
- **Coordinates**: Original pixel → norm1000 (based on original dims; no runtime resizing in training path).
- **Text**: Original geometries preserved in JSON-lines (training target); `emit_norm` affects text only.
- **Images**: No HF smart-resize in training path (`do_resize=false`).
- **Tokens**: `<|image_pad|>` expanded to match vision token count

### Geometry Handling
**Modules**: `datasets/builders/jsonlines.py`, `datasets/geometry.py`, `datasets/augment.py`

- Supported types: `bbox_2d` (4), `quad` (8), `line` (2N)
- Default: exact points preserved in top-level `objects.bbox`; template scales any even-length list to norm1000.
- Augmentation: affine transforms update points atomically; text unchanged; spatial accuracy preserved.
- Useful ops: `normalize_points()`, `apply_affine()`

### Message Formats

**JSONLinesBuilder**（Grouped JSON 场景）:
```json
{
  "图片_1": {
    "object_1": {
      "quad": [x1,y1,...,x4,y4],
      "desc": "BBU设备/华为/显示完整/..."
    },
    "object_2": {
      "line": [x1,y1,...,xn,yn],
      "desc": "光纤/有保护措施/..."
    }
  },
  "图片_2": {
    "object_1": {
      "bbox_2d": [x1,y1,x2,y2],
      "desc": "螺丝、光纤插头/..."
    }
  }
}
```

**User turn**: 所有图片放在 `content` 列表中，末尾附上一段文字指令。

**Configuration**:
- `global_max_length`: Single knob for full conversation length (prompt + completion); overrides `model.max_model_len` and `template.max_length`
- `emit_norm`: 控制文本输出的坐标空间（none/norm100/norm1000）
（已移除）模板负责插入 图片_{i} 分隔，代码不再提供 `group_key_prefix` 配置项。
- 顶层 `objects.ref/bbox/image_id` 保留为原始像素坐标，模板自动归一化为 norm1000
- 不再有 section headers 或 `image_index` 字段
- `prompts.scheme`: A (minimal/prior-free) or B (informative, adds ordering/taxonomy hints)
- **Packing** (optional): `training.packing: true` to concatenate samples to `global_max_length`, eliminating padding waste
  - ✅ Compatible with Qwen3-VL (`support_padding_free=True`)
  - ⚠️ Incompatible with `lazy_tokenize`; requires bin-packing preprocessing
  - Best for variable-length samples; ~90-95% GPU utilization

**Dual Representation Strategy**:
1. **Assistant 文本**: 使用 grouped JSON，几何字段直接暴露（bbox_2d/quad/line）。
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
| **DynamicPairDataset** | Core engine | Epoch-seeded pairing, orchestrates pipeline |
| **DenseCaptionDataset** | High-level wrapper | Selects builder, configures augmentation |


## Quick Start

```bash
# Basic training
python -m src.sft --config configs/qwen3vl_lora.yaml

# With base config inheritance (two ways)
# 1) Inline in YAML via `extends`
python -m src.sft --config configs/standard.yaml

# 2) CLI-provided base (lowest precedence)
python -m src.sft --config configs/experiment.yaml --base_config configs/base.yaml

# Debug mode
python -m src.sft --config configs/debug.yaml --debug
```

**YAML Structure** (explicit values required; you can factor shared fields into `configs/base.yaml`):
```yaml
# Inheritance (optional)
extends: base.yaml            # or a list: ["base.yaml", "more.yaml"]

model:
  model: path/to/Qwen3-VL-4B-Instruct

template:
  template: qwen3_vl             # use model's chat_template.json (图片_{i} auto-injected)
  max_length: 4096
  truncation_strategy: right
  max_pixels: 401408             # e.g., up to 1024x1024

training:
  num_train_epochs: 3
  per_device_train_batch_size: 2
  learning_rate: 1e-4

custom:
  train_jsonl: data/ds_v2_full/train.jsonl
  val_jsonl: data/ds_v2_full/val.jsonl
  emit_norm: norm1000               # none | norm100 | norm1000
  # 无需配置 group_key_prefix；模板自动插入 图片_{i}

prompts:
  scheme: A | B   # A: 极简格式约束；B: 薄领域提示
  system: |
    你是图像密集标注助手。只返回原始 JSON-lines…（B 可额外包含对象类型与排序提示）
    模板自动插入 图片_{i} 分隔，无需在文本中手动分段
  user: 描述所有对象
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
    - model.visual.deepstack_merger_list
```

Tip: keep `training.aligner_lr` (and optionally `training.vit_lr`) to control per-module learning rates via the multimodal optimizer.

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| **FileNotFoundError** | Image paths must be relative | Runner auto-sets `ROOT_IMAGE_DIR` to JSONL dir; verify paths exist |
| **MaxLengthError/OOM** | Long JSON-lines or many objects | Prefer `global_max_length` (single knob) or lower `template.max_length`; `truncation_strategy=right` (auto) |
| **Points misalignment** | Augmentation bug | `AugmentationPreprocessor` updates images+geometries atomically; print sample to verify |
| **Memory/performance** | Suboptimal settings | Use `attn_impl=flash_attn`, `torch_dtype=bfloat16`, `gradient_checkpointing=true`; adjust batch size; configure DeepSpeed for multi-GPU |



# Training Guide

Status: Active — Internal Engineering

Complete guide to training Qwen3-VL models for dense captioning and quality control tasks.

## Table of Contents

- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Training Modes](#training-modes)
- [Two-Stage Workflow](#two-stage-workflow)
- [Advanced Topics](#advanced-topics)
- [Troubleshooting](#troubleshooting)

## Quick Start

```bash
# 1. Prepare configuration
cp configs/base.yaml configs/my_experiment.yaml
# Edit my_experiment.yaml with your paths and hyperparameters

# 2. Launch training
python -m src.sft --config configs/my_experiment.yaml

# 3. Monitor progress
tensorboard --logdir tb/my_experiment
```

## Configuration

### YAML Structure

All training is controlled via YAML configuration files. The config loader supports inheritance and merging:

```yaml
# Inherit from base config
extends: base.yaml

# Model settings
model:
  model: /path/to/Qwen3-VL-4B-Instruct  # Base model or checkpoint
  torch_dtype: bfloat16
  attn_impl: flash_attention_2

# Template settings (uses model's native chat_template)
template:
  template: qwen3_vl
  max_length: 4096
  truncation_strategy: right
  max_pixels: 589824  # Up to 1024x1024

# Training hyperparameters
training:
  num_train_epochs: 3
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
  learning_rate: 1.0e-4
  warmup_ratio: 0.1
  weight_decay: 0.1
  lr_scheduler_type: cosine_warmup_with_min_lr
  lr_scheduler_kwargs:
    min_lr: 1.0e-6
  # Performance
  packing: true          # Padding-free training
  gradient_checkpointing: true
  bf16: true

# Custom dataset settings
custom:
  train_jsonl: data/bbu_full_768/train.jsonl
  val_jsonl: data/bbu_full_768/val.jsonl
  emit_norm: norm1000
  images_per_user_turn: 2
  summary_ratio: 0.0  # 0=dense only, 1=summary only, 0.5=mixed

# Prompts
prompts:
  scheme: B  # A: minimal, B: informative

# DeepSpeed (multi-GPU)
deepspeed:
  enabled: true
  config: zero2
```

### Config Inheritance

Configs can inherit from base files to avoid repetition:

```yaml
# Inheritance examples
extends: base.yaml              # Single base
extends: [base.yaml, gpu8.yaml] # Multiple bases (later wins)

# Paths are relative to the current YAML file
extends: ../base/common.yaml    # Relative path
extends: /abs/path/base.yaml    # Absolute path
```

**Merge rules:**
- Earlier bases have lower precedence
- Current file always wins
- Cycles are detected and raise errors
- Deep merge for nested dicts

### Global Length Control

Use `global_max_length` as a single knob for full conversation length (prompt + completion):

```yaml
global_max_length: 20000  # Sets both model.max_model_len and template.max_length
```

This is preferred over setting `model.max_model_len` and `template.max_length` separately.

## Training Modes

### Full Fine-Tuning

Train all parameters. High memory usage but maximum flexibility.

```yaml
tuner:
  train_type: full
```

**Use when:**
- Small model size
- Plenty of GPU memory
- Need maximum capacity

**Memory:** ~40GB for Qwen3-VL-4B (bfloat16)

### LoRA (Low-Rank Adaptation)

Most efficient mode. Trains small adapter weights (~240MB) instead of full model (~9.6GB).

```yaml
tuner:
  train_type: lora
  target_modules: [all-linear]  # Apply to all linear layers
  lora_rank: 16
  lora_alpha: 32
  lora_dropout: 0.1
  lora_bias: none
```

**Use when:**
- Limited GPU memory
- Fast iteration
- Deployment requires small checkpoints

**Memory:** ~12GB for Qwen3-VL-4B (LoRA rank 16)

### Selective Freezing

Freeze specific components while training others:

```yaml
tuner:
  train_type: lora
  freeze_llm: false   # Train LLM with LoRA
  freeze_vit: true    # Freeze vision tower
  freeze_aligner: false  # Train aligner with LoRA
```

### Mixed Mode: LoRA + Full Fine-Tuning

Apply LoRA to some modules, full fine-tuning to others (e.g., aligner):

```yaml
tuner:
  train_type: lora
  target_modules: [all-linear]
  freeze_llm: false
  freeze_vit: false
  freeze_aligner: true  # Don't apply LoRA to aligner
  modules_to_save:      # Full fine-tune these modules
    - model.visual.merger
    - model.visual.deepstack_merger_list.0
    - model.visual.deepstack_merger_list.1
    - model.visual.deepstack_merger_list.2
```

**Critical:** Use regex targeting for precise control:

```yaml
tuner:
  target_regex: '^(?:model\.)?(?:language_model\.layers\.(?:34|35)\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj)))$'
```

This targets only LLM last-2 layers for LoRA.

### Multimodal Optimizer

Use different learning rates for different components:

```yaml
training:
  optimizer: multimodal  # Enable per-component LR
  learning_rate: 1.0e-4   # LLM learning rate
  vit_lr: 8.0e-5          # Vision tower LR (if not frozen)
  aligner_lr: 1.0e-4      # Aligner LR
```

**Use when:**
- Vision tower needs slower updates
- Aligner needs faster convergence
- Fine-tuning from pre-aligned checkpoint

## Two-Stage Workflow

Recommended training strategy for Qwen3-VL:

### Stage 1: Aligner-Only

Start from base model, freeze LLM + Vision, train only aligner.

**Goal:** Learn vision-to-text alignment without disturbing pre-trained components.

```yaml
# configs/stage_1_aligner.yaml
extends: base.yaml

model:
  model: /path/to/Qwen3-VL-4B-Instruct

tuner:
  train_type: lora
  target_modules: [all-linear]
  freeze_llm: true
  freeze_vit: true
  freeze_aligner: false  # Train aligner with LoRA

training:
  num_train_epochs: 5
  learning_rate: 1.0e-4
  output_dir: output/stage_1
```

**Run:**
```bash
python -m src.sft --config configs/stage_1_aligner.yaml
```

**Output:** `output/stage_1/checkpoint-XXX/` (adapter weights only)

### Stage 2: LLM + Aligner

Load Stage-1 adapter, unfreeze LLM, continue training.

**Goal:** Fine-tune language understanding while preserving alignment.

```yaml
# configs/stage_2_llm_aligner.yaml
extends: base.yaml

model:
  model: /path/to/Qwen3-VL-4B-Instruct  # Base model (NOT stage-1 checkpoint)

tuner:
  train_type: lora
  target_modules: [all-linear]
  freeze_llm: false   # Now train LLM
  freeze_vit: true    # Keep vision frozen
  freeze_aligner: false
  resume_from_checkpoint: output/stage_1/checkpoint-XXX  # Load stage-1 adapter

training:
  num_train_epochs: 3
  learning_rate: 5.0e-5  # Lower LR for stability
  output_dir: output/stage_2
```

**Run:**
```bash
python -m src.sft --config configs/stage_2_llm_aligner.yaml
```

**Output:** `output/stage_2/checkpoint-YYY/` (combined adapter)

### Best Checkpoint Selection

Training saves adapters based on evaluation loss:

```yaml
training:
  eval_strategy: steps
  save_strategy: best
  eval_steps: 20
  save_steps: 40
  load_best_model_at_end: true
  metric_for_best_model: eval_loss
  greater_is_better: false
  save_total_limit: 2  # Keep only best 2 checkpoints
```

## Advanced Topics

### LoRA Adapter Preparation

**Critical:** When writing custom training scripts, you **must** call `sft.prepare_model()` before creating the trainer:

```python
from swift.llm import SwiftSft, TrainArguments

# Load config
train_args = TrainArguments.from_config(...)

# Initialize SFT pipeline
sft = SwiftSft(train_args)

# ⚠️ CRITICAL: Apply LoRA adapter before creating trainer
sft.model = sft.prepare_model(
    train_args,
    sft.model,
    template=sft.template,
    train_dataset=dataset
)

# Now create trainer
trainer = Trainer(model=sft.model, ...)
```

**What `prepare_model()` does:**
1. Freezes modules based on `freeze_llm`, `freeze_vit`, `freeze_aligner`
2. Applies LoRA adapters to `target_modules`
3. Marks `modules_to_save` for full fine-tuning
4. Wraps model in `SwiftModel` or `PeftModel`
5. Enables adapter saving instead of full checkpoint

**Without this call:**
- ❌ Model remains unwrapped (raw `Qwen3VLModel`)
- ❌ LoRA not applied; all parameters trainable (OOM risk)
- ❌ Checkpoint saves ~9.6GB full weights instead of ~240MB adapter
- ❌ `adapter_config.json` missing or empty `modules_to_save`

**Verification:**
```python
# After prepare_model, check model type
print(f"Model type: {type(sft.model).__name__}")
# Expected: SwiftModel or PeftModel (not Qwen3VLModel)

# Check if adapter config exists after training
import json
adapter_cfg = json.load(open("checkpoint/adapter_config.json"))
print(adapter_cfg["modules_to_save"])  # Should list your aligner modules
```

See also: REFERENCE → Upstream internals (ms‑swift SFT/LoRA).

### Dynamic Per-Group Prompt Selection

Train with multiple output formats simultaneously by selecting prompts per pairing group.

**Supported modes:**
- **Dense-only (default)**: `prompts.scheme: B` → grouped JSON with geometry
- **Summary-only**: `summary_ratio: 1.0` → one-line summaries per image
- **Mixed**: `summary_ratio: 0.5` → randomly alternate per group

**Key insight:** All samples in one pairing group see the same system prompt and produce the same output format (all dense or all summary), ensuring coherent JSON shapes.

```yaml
# Dense-only (existing behavior)
prompts:
  scheme: B
custom:
  train_jsonl: data/train.jsonl
  # No summary_ratio → always dense

# Summary-only
prompts:
  scheme: B
custom:
  train_jsonl: data/train.jsonl
  summary_ratio: 1.0  # All groups use summary mode

# Mixed (50% summary, 50% dense per group)
prompts:
  scheme: B
custom:
  train_jsonl: data/train.jsonl
  summary_ratio: 0.5
```

**Data requirement:** When `summary_ratio > 0`, all records must have a valid `summary` field.

For details, see [DATA_FORMATS.md](DATA_FORMATS.md#dynamic-mode-selection).

### Augmentation

Enable geometry-aware augmentation to improve robustness:

```yaml
custom:
  augmentation:
    enabled: true
    ops:
      # Geometric ops
      - name: hflip
        params: { prob: 0.3 }
      - name: vflip
        params: { prob: 0.1 }
      - name: rotate
        params: { max_deg: 15.0, prob: 0.3 }
      - name: scale
        params: { lo: 1.05, hi: 1.25, prob: 0.2 }
      # Color ops
      - name: color_jitter
        params: { brightness: [0.85, 1.15], contrast: [0.85, 1.15], prob: 0.3 }
      # Enforce size multiple
      - name: pad_to_multiple
        params: { multiple: 32 }
```

Augmentation applies affine transforms to both images and geometries atomically, preserving spatial accuracy.

### Packing (Padding-Free Training)

Concatenate samples to `global_max_length`, eliminating padding waste:

```yaml
training:
  packing: true
```

**Benefits:**
- ~90-95% GPU utilization (vs ~60-70% with padding)
- Faster training on variable-length samples

**Limitations:**
- Incompatible with `lazy_tokenize`
- Requires bin-packing preprocessing
- Only compatible with models that support padding-free (Qwen3-VL does)

## Troubleshooting

### Zero Gradient Norm for Vision/Aligner

**Symptom:** `grad_norm=0` in logs; vision components not learning.

**Cause:** Wrong image key in message content.

**Solution:** Message builders must use `{"type": "image", "image": path}` not `{"type": "image", "url": path}`.

ms-swift extracts media via `item.get(item['type'])`, so using `"url"` instead of `"image"` returns `None` → no vision tensors → aligner never executes.

**Check:** Verify `pixel_values` and `image_grid_thw` in encoded samples:
```python
sample = dataset[0]
print(sample.keys())  # Must include 'pixel_values' and 'image_grid_thw'
```

### Full Model Saved Instead of LoRA Adapter

**Symptom:** Checkpoint is ~9.6GB instead of ~240MB.

**Cause:** Missing `sft.prepare_model()` call.

**Solution:** See [LoRA Adapter Preparation](#lora-adapter-preparation) section above.

**Verification:**
```bash
# Check checkpoint size
du -sh output/checkpoint-XXX/
# Expected: ~240MB for LoRA, ~9.6GB for full model

# Check adapter config
cat output/checkpoint-XXX/adapter_config.json
# Should have non-empty "modules_to_save" list
```

### ModuleList Error in modules_to_save

**Symptom:** `TypeError: modules_to_save cannot be applied to ModuleList`.

**Cause:** Invalid `modules_to_save` path pointing to a container instead of individual modules.

**Solution:** Specify individual elements:
```yaml
# ❌ Wrong
modules_to_save:
  - model.visual.deepstack_merger_list  # This is a container

# ✅ Correct
modules_to_save:
  - model.visual.deepstack_merger_list.0
  - model.visual.deepstack_merger_list.1
  - model.visual.deepstack_merger_list.2
```

### Out of Memory (OOM)

**Symptom:** CUDA OOM during training.

**Solutions:**
1. Lower batch size:
   ```yaml
   training:
     per_device_train_batch_size: 1
     gradient_accumulation_steps: 8  # Maintain effective batch size
   ```

2. Reduce sequence length:
   ```yaml
   global_max_length: 8192  # Down from 20000
   ```

3. Enable gradient checkpointing:
   ```yaml
   training:
     gradient_checkpointing: true
   ```

4. Use DeepSpeed ZeRO:
   ```yaml
   deepspeed:
     enabled: true
     config: zero3  # Shard optimizer + gradients + parameters
   ```

5. Reduce LoRA rank:
   ```yaml
   tuner:
     lora_rank: 8  # Down from 16
   ```

### Missing pixel_values / image_grid_thw

**Symptom:** Encoded samples lack vision tensors.

**Cause:** Same as zero gradient norm — wrong image key in messages.

**Solution:** Fix message builder to use `"image"` key (see above).

### FileNotFoundError for Images

**Symptom:** Can't find image files during training.

**Cause:** Image paths in JSONL must be relative to the JSONL file's directory.

**Solution:** 
- Use relative paths in JSONL: `"images": ["images/img1.jpg"]`
- Runner auto-sets `ROOT_IMAGE_DIR` to JSONL directory
- Or use absolute paths: `"images": ["/abs/path/to/img1.jpg"]`

### Points Misalignment After Augmentation

**Symptom:** Geometry coordinates don't match augmented images.

**Cause:** Augmentation bug or invalid affine transform.

**Solution:** `AugmentationPreprocessor` updates images + geometries atomically. Print sample to verify:
```python
sample = dataset[0]
print(json.loads(sample["messages"][1]["content"][0]["text"]))  # Check coords
```

### Loss is NaN

**Symptom:** Training loss becomes NaN after a few steps.

**Solutions:**
1. Lower learning rate:
   ```yaml
   training:
     learning_rate: 5.0e-5  # Down from 1.0e-4
   ```

2. Enable gradient clipping:
   ```yaml
   training:
     max_grad_norm: 1.0  # Default: 0.5; try 1.0
   ```

3. Check for corrupted data:
   ```python
   from src.datasets.utils import load_jsonl
   records = load_jsonl('data/train.jsonl')[:10]
   for i, r in enumerate(records):
       assert all(k in r for k in ('images','objects','width','height')), f"Missing keys in record {i}"
   ```

4. Use bfloat16 instead of float16:
   ```yaml
   model:
     torch_dtype: bfloat16
   training:
     bf16: true
   ```

## Additional Resources

- **Inference workflows**: See [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md)
- **Data preparation**: See [DATA_FORMATS.md](DATA_FORMATS.md)
- **Advanced topics**: See [REFERENCE.md](REFERENCE.md)
- **Code architecture**: See `../src/README.md`

---

**Last Updated**: October 25, 2025


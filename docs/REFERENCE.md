# Training & Inference Reference

Comprehensive guide for training, inference, deployment, and advanced topics.

**Source**: `src/sft.py`, `src/stage_a/`, `src/stage_b/`, `scripts/`, `configs/`

---

## Table of Contents

- [Training](#training)
  - [Essentials](#training-essentials)
  - [Modes](#training-modes)
  - [Two-Stage Recipe](#two-stage-recipe)
  - [Troubleshooting](#training-troubleshooting)
- [Inference](#inference)
  - [Checkpoints](#checkpoints)
  - [Dense Captioning](#dense-captioning)
  - [Stage-A & Stage-B](#stage-ab)
  - [Deployment Tips](#deployment-tips)
- [Advanced Topics & FAQ](#advanced-topics--faq)
- [Architecture & Implementation](#architecture--implementation)

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
- `src/datasets/geometry.py` - Core geometry transforms (bbox, quad, line)
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
# - Applies affine matrix M to bbox/quad/line
# - Clips to image bounds (with polygon/line clipping algorithms)
# - Preserves degenerate geometries (fallback to clamping)
# - Handles rotation without clipping for fully-inside quads
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

### Training Essentials

**YAML-Only Surface**:
- Avoid CLI flags beyond `--config` (and optional `--base_config`, `--debug`)
- All configuration in YAML files
- Single length knob: `global_max_length`

**Critical Setup**:
```yaml
# Always set these
model:
  model: /path/to/Qwen3-VL-4B-Instruct
template:
  template: qwen3_vl
  max_length: 4096           # Or use global_max_length

# Set global_max_length as the single length knob
global_max_length: 4096      # Proxies both model.max_model_len and template.max_length

# REQUIRED: ms-swift validation placeholder
# Even with custom dataset loading, ms-swift's TrainArguments.__post_init__
# validates that dataset is non-empty. Keep this dummy value.
data:
  dataset: ["dummy"]         # Never remove - required for ms-swift initialization
```

**Adapter Preparation** (Critical!):
```python
# Always call sft.prepare_model() before creating trainer
# This configures adapters, freezes, modules_to_save
sft.prepare_model(...)
```

❌ **Common mistake**: Forgetting `prepare_model()` → full model saved instead of adapter

### Training Modes

| Mode | Memory | Speed | Use Case |
|------|--------|-------|----------|
| **Full Fine-Tuning** | Highest | Slower | Maximum flexibility, production deployment |
| **LoRA** | ~240MB | Faster | Iteration, experimentation, adapter deployment |
| **Selective Freezing** | Variable | Fast | Targeted component training |

**LoRA Configuration**:
```yaml
tuner:
  train_type: lora
  lora_rank: 32
  lora_alpha: 64
  target_modules: [all-linear]      # Or specific modules
  freeze_llm: false                 # Control what to freeze
  freeze_vit: true
  freeze_aligner: false
```

**Selective Freezing** (Mix with LoRA or Full):
- `freeze_llm: true` - Freeze language model
- `freeze_vit: true` - Freeze vision encoder
- `freeze_aligner: false` - Train aligner (projector)

### Two-Stage Recipe (Recommended)

**Stage 1: Aligner-Only LoRA**
```yaml
# Learn vision-language alignment
tuner:
  train_type: lora
  target_modules: [all-linear]
  freeze_llm: true                  # Freeze LLM
  freeze_vit: true                  # Freeze ViT
  freeze_aligner: false             # Train aligner only

training:
  num_train_epochs: 3
  learning_rate: 1.0e-4
```

**Stage 2: LLM + Aligner LoRA**
```yaml
# Refine language while preserving alignment
model:
  model: /path/to/base/Qwen3-VL-4B-Instruct

tuner:
  train_type: lora
  target_modules: [all-linear]
  freeze_llm: false                 # Train LLM
  freeze_vit: true                  # Keep ViT frozen
  freeze_aligner: false             # Train aligner
  resume_from_checkpoint: /path/to/stage1/checkpoint-XXX

training:
  num_train_epochs: 2
  learning_rate: 5.0e-5             # Lower LR for fine-tuning
```

**Benefits**:
- Stage 1 learns alignment without language drift
- Stage 2 refines language without breaking alignment
- Faster convergence than single-stage
- Better generalization

### KL Anchoring with GKD

Use Generalized Knowledge Distillation (GKD) when dense-caption SFT starts hallucinating away from the base checkpoint.

- **Activation**: switch to the GKD overlays (`configs/stage_2_llm_lora_gkd.yaml`, `configs/stage_3_gkd.yaml`). They inherit the vanilla stage configs and only add:
  ```yaml
  rlhf:
    rlhf_type: gkd
    teacher_model: /abs/path/to/base/Qwen3-VL-4B-Instruct
    beta: 0.5        # KL weight
    sft_alpha: 0.3   # CE mix-in weight
    seq_kd: true
    lmbda: 0.5       # On-policy mixing ratio
    max_completion_length: 256
    temperature: 0.9
  custom:
    trainer_variant: gkd_monitor  # enable KL+CE logging wrapper
  ```
- **Launch**: run the usual entrypoint (`python -m src.sft --config <gkd-config.yaml>`). The loader instantiates `SwiftRLHF` behind the scenes, loads the frozen teacher, and routes training through ms-swift’s `GKDTrainer`.
- **Telemetry**: the wrapper keeps the huggingface `loss` scalar and emits `train/loss`, `train/sft_loss`, `train/llm_kd_loss`, `train/vision_kd_loss`, `train/token_acc`, plus the same `eval/*` counterparts. Metrics are prefixed exactly once (`train/*`, `eval/*`) to avoid TensorBoard duplication. `train/llm_kd_loss` reflects the **weighted** JSD term (`rlhf.llm_kd_weight * jsd`); when the weight is `0`, the metric is omitted entirely. Watch for `train/llm_kd_loss` spikes to catch drift early; compare `train/sft_loss` against your vanilla SFT runs to ensure language quality is intact.

#### LM-head KD Weight

- `rlhf.llm_kd_weight` defaults to `1.0` and scales the LM-head JSD term without modifying other losses.
- Set it below `1.0` to lighten logits anchoring, or `0.0` to disable LM KD while keeping visual KD active.
- The loader attaches the knob to both `TrainArguments` and nested `training_args`, so downstream tooling can introspect the runtime value.
- Example overlay to disable LM KD while preserving visual KD hooks:

```yaml
rlhf:
  llm_kd_weight: 0.0

custom:
  visual_kd:
    enabled: true
    weight: 0.1
    targets: [merger, deepstack]
```

#### Vision/Aligner Feature KD (optional)

When the vision encoder or aligner drifts while the language tower needs freedom to adapt (e.g., new coordinate formats), enable the feature distillation block:

```yaml
custom:
  visual_kd:
    enabled: true
    weight: 0.1              # scalar multiplied into the loss term
    targets: [merger, deepstack]  # final merged tokens and deepstack adapters
    distance: mse            # or `cosine`
```

- **Effect**: anchors student vision/aligner activations to the frozen teacher while leaving KL + CE to supervise the language tower.
- **Metrics**: trainer logs `train/vision_kd_loss` / `eval/vision_kd_loss` (post-weight) so you can monitor the regularizer alongside `llm_kd_loss` and `sft_loss` contributions.
- **Images only**: batches without `pixel_values` automatically skip the term; no special handling is required for summary-only validation shards.
- **Preset overlay**: `configs/stage_3_gkd_visual.yaml` extends the standard Stage-3 recipe with the block above—use it as the starting point for experiments.

#### Forward-only KD (recommended for domain migration)

Use this when you want CE to drive adaptation while KL lightly anchors logits to the base model, without any on-policy sampling.

```yaml
rlhf:
  rlhf_type: gkd
  teacher_model: /abs/path/to/base/Qwen3-VL-4B-Instruct
  sft_alpha: 1.0   # CE dominates (domain learning)
  beta: 0.1        # light KL anchoring
  seq_kd: false    # no teacher sampling
  lmbda: 0.0       # no student sampling
  # temperature/max_completion_length are ignored in forward-only mode
```

Notes:
- Teacher == Student base at init → KL≈0 initially; increases only with drift.
- If overfitting/drift persists: raise `beta` to 0.2–0.3.
- If under-adapting: lower `beta` to 0.05 or reduce LR/epochs.
- **Tuning**:
  - Increase `beta` (→ stronger anchoring) if hallucinations persist.
  - Increase `sft_alpha` if CE should dominate (e.g., when the dataset is clean but narrow).
  - Decrease `lmbda` to rely less on on-policy generations when the student is unstable.
- **Compute Overhead**: expect ~1.6–2.0× wall-clock vs. vanilla SFT (extra teacher forward pass; add teacher sampling cost only if `seq_kd=true`). Evaluation runs skip the teacher and only compute CE, so validation cost stays close to baseline.
- **Monitoring Checklist**:
- `train/llm_kd_loss` steady or slowly decreasing → healthy anchoring.
  - `train/sft_loss` aligns with prior SFT runs → no regression.
- `eval/llm_kd_loss` jump → teacher/template mismatch (fix tokenizer/template).
- **Smoke Test**: set `custom.sample_limit: 32` and `training.save_steps: 5` in a temporary overlay, then run `python -m src.sft --config configs/stage_3_gkd.yaml`. Verify `logging.jsonl` includes `train/llm_kd_loss`, `train/vision_kd_loss`, `train/sft_loss`, and the output directory writes checkpoints.

### Packing (Padding-Free Training)

**Enable Packing**:
```yaml
training:
  packing: true
```

**Benefits**:
- Eliminates padding waste
- 20-30% faster training
- Better GPU utilization

**Requirements**:
- Qwen3-VL (Flash Attention 2+)
- Incompatible with `lazy_tokenize`
- Compatible with LoRA and full fine-tuning

### Training Health Checks

Before training, verify:

- [ ] Vision tokens present (`pixel_values`, `image_grid_thw`)
- [ ] Image placeholders match image count in user messages
- [ ] `modules_to_save` lists any full-tuned modules (if used)
- [ ] Adapter config correct (for LoRA mode)
- [ ] Correct base model path
- [ ] `global_max_length` or `template.max_length` set

**Debug Mode**:
```bash
python -m src.sft --config config.yaml --debug
```

### Training Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Full model saved instead of adapter | Missing `sft.prepare_model()` | Always call before trainer creation |
| Zero gradients for vision/aligner | Wrong content format | Use `{"type": "image", "image": path}` |
| OOM | Batch size / length too large | Lower batch size, enable gradient checkpointing, use ZeRO |
| Slow convergence | Learning rate mismatch | Try 1e-4 for LoRA, 5e-5 for full |
| NaN loss | LR too high or bad data | Lower LR, check data validation |

---

## Inference

### Checkpoints

**Adapter-Based** (Recommended for Development):
```bash
# Load base + LoRA adapter
CUDA_VISIBLE_DEVICES=0 swift infer \
  --model /path/to/Qwen3-VL-4B-Instruct \
  --adapters /path/to/checkpoint-XXX \
  --stream true --max_new_tokens 2048
```

**Benefits**:
- Small adapter file (~240MB)
- Flexible (swap adapters easily)
- Easy to version control

**Merged** (Recommended for Production):
```bash
# Merge adapter into base model
CUDA_VISIBLE_DEVICES=0 swift export \
  --model /path/to/Qwen3-VL-4B-Instruct \
  --adapters /path/to/checkpoint-XXX \
  --merge_lora true \
  --output_dir /path/to/merged \
  --save_safetensors true
```

**Benefits**:
- Single self-contained checkpoint
- Faster inference (no adapter overhead)
- Easier deployment

### Dense Captioning (Standard)

**Input Format**:
```python
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": "img1.jpg"},
        {"type": "text", "text": "请描述图片中的所有物体"}
    ]
}]
```

**Output Format**:
```json
{
  "object_1": {"bbox_2d": [100, 200, 300, 400], "desc": "BBU设备/品牌:华为/型号:5900"},
  "object_2": {"line_points": 4, "line": [50, 60, 80, 120, 130, 180, 180, 220], "desc": "光纤/颜色:黄色/保护:有保护"}
}
```

**Best Practices**:
- Use deterministic generation (low temperature) for deployment
- Set `max_new_tokens` based on expected output length
- Stream responses for better UX

### Stage-A & Stage-B

**Stage-A: Per-Image Summaries**

Purpose: Generate one-line Chinese summaries per image

Runtime guardrails: `src.stage_a.cli.StageAConfig` validates checkpoint/input paths, mission name, and generation knobs before inference runs.

```bash
python -m src.stage_a.run \
  --model /path/to/merged \
  --input images/ \
  --output summaries.jsonl
```

Output format:
```json
{"image_id": "img1", "summary": "BBU设备×1，光模块×3，线缆×2"}
```

**Stage-B: Group-Level Verdict (Training-Free Loop)**

Purpose: Multi-image scene-level pass/fail judgment using the frozen Stage-B checkpoint plus deterministic judging.

1. Prepare a Stage-B config (see `configs/stage_b_training_free.yaml`).
2. Run the end-to-end loop:

```bash
bash scripts/stage_b_run.sh $(pwd)/configs/stage_b_training_free.yaml --log-level debug
```

This performs ingest → rollout → judge → selection in one pass and stores artifacts under `output/stage_b/`:

- `guidance.json`: mission guidance (focus plus optional rules/preferences, seeded from `STAGE_B_MISSION_FOCUS` if absent) with rolling snapshots under `output/stage_b/snapshots/`.
- `trajectories.jsonl`: hierarchical records per candidate rollout. Each entry now looks like:
  ```json
  {
    "group_id": "QC-0001",
    "mission": "挡风板安装检查",
    "epoch": 1,
    "result": {
      "candidate_index": 3,
      "temperature": 0.6,
      "top_p": 0.95,
      "max_new_tokens": 512,
      "seed": 43,
      "text": "不通过\n理由: 挡风板缺失",
      "verdict": "不通过",
      "reason": "挡风板缺失",
      "format_ok": true,
      "created_at": "2025-11-03T08:55:12.483920Z",
      "scores": {
        "label_match": false,
        "focus_consistency": 1.0,
        "heuristic_score": 0.6,
        "summary_confidence": "needs_review",
        "semantic_advantage": 0.822,
        "label_contradiction": true,
        "needs_recheck": true,
        "rationale_quality": 1.0
      }
    }
  }
  ```
- `selections.jsonl`: final verdicts with the same `{"group_id", "mission", "result": {...}}` shape (result includes `verdict`, `reason`, `summary_confidence`, `selected_candidate`, etc.).

> **Tip:** Set `sampler.samples_per_decode` > 1 when you need multiple attempts per ticket under the same temperature; keep it at `1` for deterministic sweeps.
> Configure multi-epoch runs via the `runner` block in `configs/stage_b_training_free.yaml` (e.g., `epochs: 5`) and set the top-level `seed` to lock randomness. Stage-B shuffles Stage-A tickets deterministically each epoch using this seed.

### Deployment Tips

**Latency Optimization**:
1. Use merged checkpoints (faster than adapter loading)
2. Enable Flash Attention 2 (default for Qwen3-VL)
3. Use bfloat16 precision (balance speed/quality)
4. Batch multiple images when possible

**Memory Optimization**:
1. Use quantization (int8/int4) for large-scale deployment
2. Lower `max_model_len` if not needed
3. Use gradient checkpointing during training (no impact on inference)

**Quality Optimization**:
1. Set temperature=0 for deterministic output
2. Use appropriate `top_p` (0.9 default)
3. Validate output format with regex/JSON parsing
4. Keep processor/template aligned with base model

---

## Advanced Topics & FAQ

## Learning Rate Scheduler (FAQ)

### Cosine with Min LR

Recommended scheduler with minimum learning rate floor:

```yaml
training:
  lr_scheduler_type: cosine_warmup_with_min_lr
  learning_rate: 1.0e-4
  warmup_ratio: 0.1
  lr_scheduler_kwargs:
    min_lr: 1.0e-6  # Prevents LR from going to zero
```

**Why min_lr matters:** Standard cosine decay goes to zero at end of training, which can cause instability. Setting `min_lr` provides a floor for continued gradual improvement.

## DeepSpeed Configuration (FAQ)

### ZeRO Stage 2 (Recommended)

Shard optimizer states + gradients across GPUs:

```yaml
deepspeed:
  enabled: true
  config: zero2
```

Memory savings: ~40% vs single GPU

### ZeRO Stage 3 (Maximum Savings)

Shard optimizer + gradients + parameters:

```yaml
deepspeed:
  enabled: true
  config: zero3
```

Memory savings: ~70% vs single GPU (but slower due to communication overhead)

## Augmentation Pipeline (FAQ)

Geometry-aware augmentation that updates both images and coordinates atomically:

```yaml
custom:
  augmentation:
    enabled: true
    ops:
      # Geometric (affects coordinates)
      - name: hflip
        params: { prob: 0.3 }
      - name: rotate
        params: { max_deg: 15.0, prob: 0.3 }
      # Color (doesn't affect coordinates)
      - name: color_jitter
        params: { brightness: [0.85, 1.15], prob: 0.3 }
      # Size enforcement
      - name: pad_to_multiple
        params: { multiple: 32 }
```

All geometric transforms automatically update bbox/quad/line coordinates to maintain spatial accuracy.

## Architecture Notes (FAQ)

### Qwen3-VL Components

- **Vision Encoder**: ViT-based, processes images into patch embeddings
- **Aligner** (MLP projector): Maps vision features → LLM embedding space
  - `model.visual.merger`
  - `model.visual.deepstack_merger_list.{0,1,2}`
- **Language Model**: Qwen3 transformer (36 layers for 4B model)

### Token Flow

```
Image → Vision Encoder → Aligner → <|image_pad|> tokens → LLM
```

Each image expands to variable number of vision tokens based on resolution and `image_grid_thw`.

## Chat Template Mechanics (FAQ)

### Image Placeholder Insertion

The model's native `chat_template` automatically inserts `<|image_pad|>` tokens:

```python
# You provide:
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": "img.jpg"},  # ← CRITICAL: key must be "image"
        {"type": "text", "text": "Describe this"}
    ]
}]

# Template expands to:
# <|im_start|>user
# <|image_pad|><|image_pad|>...<|image_pad|>Describe this<|im_end|>
```

**Never** hand-craft `<|image_pad|>` yourself - the template handles it based on image resolution.

### Common Mistake

```python
# ❌ WRONG - will silently fail
{"type": "image", "url": "img.jpg"}  # Wrong key name

# ✅ CORRECT
{"type": "image", "image": "img.jpg"}  # Key name matches type
```

ms-swift extracts media via `item.get(item['type'])`, so the value key must match the type key.

### Upstream internals (ms-swift, HF Qwen3‑VL)

- ms‑swift SFT/LoRA integration
  - `prepare_adapter(...)` builds LoRA config from `TunerArguments` and applies adapters via `Swift.prepare_model(...)`:
```148:171:/data/ms-swift/swift/llm/train/tuner.py
def prepare_adapter(args: TrainArguments, model, *, template=None, train_dataset=None, task_type=None):
    from swift.tuners import (AdaLoraConfig, AdapterConfig, BOFTConfig, LLaMAProConfig, LongLoRAModelType, LoraConfig,
                              LoRAConfig, ReftConfig, Swift, VeraConfig)
    task_type = (task_type or args.task_type).upper()
    target_modules = get_target_modules(args, model)
    modules_to_save = get_modules_to_save(args, model, task_type)
    lora_kwargs = {
        'r': args.lora_rank,
        'target_modules': target_modules,
        'lora_alpha': args.lora_alpha,
        'lora_dropout': args.lora_dropout,
        'bias': args.lora_bias,
        'modules_to_save': modules_to_save,
        'use_rslora': args.use_rslora,
        'use_dora': args.use_dora,
        'lorap_lr_ratio': args.lorap_lr_ratio,
        'init_lora_weights': args.init_weights,
    }
    if args.train_type in ('lora', 'longlora'):
        if args.use_swift_lora:
            lora_config = LoRAConfig(lora_dtype=args.lora_dtype, **lora_kwargs)
            model = Swift.prepare_model(model, lora_config)
            logger.info(f'lora_config: {lora_config}')
```
  - `get_target_modules` resolves `'all-linear'` to an exact regex for multimodal modules honoring freeze flags:
```92:106:/data/ms-swift/swift/llm/train/tuner.py
def get_target_modules(args, model) -> Union[str, List[str]]:
    """Replace all-linear to actual modules"""
    model_meta = model.model_meta
    if isinstance(args.target_modules, str):
        return args.target_modules
    target_modules = args.target_modules.copy()
    if 'all-linear' in target_modules:
        if model_meta.is_multimodal:
            return get_multimodal_target_regex(
                model,
                freeze_llm=args.freeze_llm,
                freeze_vit=args.freeze_vit,
                freeze_aligner=args.freeze_aligner,
                include_embedding='all-embedding' in target_modules)
```
  - Freeze knobs live in `TunerArguments` (defaults shown):
```105:114:/data/ms-swift/swift/llm/argument/tuner_args.py
# lora or full
freeze_llm: bool = False
freeze_vit: bool = True
freeze_aligner: bool = True
# tuners
target_modules: List[str] = field(default_factory=lambda: ['all-linear'])
target_regex: Optional[str] = None
modules_to_save: List[str] = field(default_factory=list)
```

- ms‑swift media extraction (strict key contract)
  - Content items must use matching keys: `{"type":"image","image":...}`; `_url` suffix is normalized, and the value is taken via `item.get(item['type'])`.
```240:254:/data/ms-swift/swift/llm/template/template_inputs.py
for item in content:
    key: str = item['type']
    value = item.get(key)
    if key == 'text':
        new_content += value
        continue
    # image/audio/video
    # image_url/audio_url/video_url
    if key.endswith('_url'):
        key = key[:-len('_url')]
    new_content += f'<{key}>'
    if isinstance(value, dict):
        value = value['url']
    if value:
        res[f'{key}s'].append(value)
```

- HF transformers Qwen3‑VL internals
  - Placeholder expansion in the processor scales `<|image_pad|>` by the image grid and merge size:
```185:195:/root/miniconda3/envs/ms/lib/python3.12/site-packages/transformers/models/qwen3_vl/processing_qwen3_vl.py
text = text.copy()  # below lines change text in-place
if image_grid_thw is not None:
    merge_length = self.image_processor.merge_size**2
    index = 0
    for i in range(len(text)):
        while self.image_token in text[i]:
            num_image_tokens = image_grid_thw[index].prod() // merge_length
            text[i] = text[i].replace(self.image_token, "<|placeholder|>" * num_image_tokens, 1)
            index += 1
        text[i] = text[i].replace("<|placeholder|>", self.image_token)
```
  - Model replaces special tokens with visual embeddings via `masked_scatter`:
```1137:1144:/root/miniconda3/envs/ms/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py
if pixel_values is not None:
    image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
    image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
    image_mask, _ = self.get_placeholder_mask(
        input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
    )
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
```

See also: Training Guide → LoRA Adapter Preparation, and Data Formats → Coordinate Normalization.


### Image content payload contract (verified)

- Content shape for images in user turns:
  - Preferred: `{ "type": "image", "image": <value> }`
  - Also accepted: `{ "type": "image_url", "image_url": {"url": <value>} }` or `{ "type": "image_url", "image_url": <value> }` — ms‑swift normalizes the `_url` suffix and extracts the `url` field when a dict is provided.
- Supported `<value>` types (per ms‑swift and HF Qwen3‑VL):
  - String path (absolute or relative). Relative paths are resolved via `ROOT_IMAGE_DIR` (we set this to the JSONL directory in `src/sft.py`).
  - Fully‑qualified URL (`http(s)://...`).
  - Data URL (`data:image/<fmt>;base64,...`). Our builder emits this automatically when the record contains bytes.
  - `PIL.Image.Image` objects are accepted by ms‑swift internals (images list is typed as `List[Union[str, Image.Image]]`).
- What the builder emits:
  - It always uses the `image` key and returns strings. If the input image is `{"bytes": ...}`, it converts to a PNG data‑URL.

References (repo code):
- Builder uses `{"type": "image", "image": ...}`:

  ````python
  # src/datasets/builders/jsonlines.py
  user_contents.append({"type": "image", "image": self._to_url(image)})
  ````
- ms‑swift extracts media and normalizes `_url` suffix:

  ````python
  # swift/llm/template/template_inputs.py
  key: str = item['type']
  value = item.get(key)
  if key.endswith('_url'):
      key = key[:-len('_url')]
  if isinstance(value, dict):
      value = value['url']
  if value:
      res[f'{key}s'].append(value)
  ````

### Dataset config: data.dataset placeholder vs custom.train_jsonl (CRITICAL)

**IMPORTANT: `data.dataset` is NOT optional in ms-swift's standard flow, but our pipeline bypasses it entirely.**

#### ms-swift's standard dataset loading (NOT used by us)

When you call `SwiftSft(train_args).main()`, ms-swift's `_prepare_dataset()` method is invoked:

````python
# swift/llm/train/sft.py, lines 123-143
@RayHelper.function(group='default')
def _prepare_dataset(self):
    args = self.args
    if args.cached_dataset:
        train_datasets, val_datasets = self._get_cached_dataset()
    else:
        train_datasets, val_datasets = [], []
    if args.dataset:  # ← CRITICAL: only loads if args.dataset is non-empty
        train_dataset, val_dataset = self._get_dataset()
        train_dataset, val_dataset = self._encode_dataset(train_dataset, val_dataset, pre_process=pre_process)
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)
    train_dataset = DatasetLoader._concat_datasets(train_datasets)
    val_dataset = DatasetLoader._concat_datasets(val_datasets)
    return [train_dataset, val_dataset]
````

If `args.dataset` is empty (or `[placeholder]`), the conditional `if args.dataset:` evaluates to `False`, and `_get_dataset()` is never called. The result is `train_datasets = []`, which gets concatenated to `None`:

````python
# swift/llm/dataset/loader.py, lines 181-186
@staticmethod
def _concat_datasets(datasets: List[HfDataset]) -> Optional[HfDataset]:
    if len(datasets) == 0:
        return  # Returns None
    if len(datasets) == 1:
        return datasets[0]
    return concatenate_datasets(datasets)
````

**Result**: If `data.dataset` is empty or a placeholder, ms-swift's `_prepare_dataset()` returns `[None, None]`, and the trainer receives `train_dataset=None`.

#### Our pipeline: complete bypass of ms-swift's dataset loading

Our `src/sft.py` **never calls** `sft.main()` or `sft.run()`. Instead, we:

1. Initialize `SwiftSft(train_args)` — this only prepares model, template, and callbacks (lines 28-34 of sft.py)
2. **Skip** `_prepare_dataset()` entirely
3. Construct our own dataset in-process:
   ````python
   # src/sft.py, lines 302-315
   dataset = DenseCaptionDataset.from_jsonl(
       train_jsonl,
       template=sft.template,
       user_prompt=custom_config.user_prompt,
       emit_norm=custom_config.emit_norm,
       ...
   )
   ````
4. Pass it directly to the trainer:
   ````python
   # src/sft.py, lines 524-533
   trainer = trainer_cls(
       model=sft.model,
       args=train_args.training_args,
       data_collator=data_collator,
       train_dataset=dataset,  # ← Our dataset, not ms-swift's
       eval_dataset=eval_dataset,
       ...
   )
   ````

#### Conclusion

- **`data.dataset` is NOT required** for our dense captioning pipeline because we bypass ms-swift's dataset loading entirely.
- **`data.dataset: [placeholder]` has NO effect** on our training; it is never read or used.
- **`custom.train_jsonl` is the authoritative dataset source** for our pipeline.
- **Recommendation**: You may safely omit `data.dataset` from configs, or keep it as `[placeholder]` for compatibility with ms-swift's CLI tools (if used separately). Either way, it does not affect our training.

## Performance Tips (FAQ)

### Memory Optimization

1. **Gradient checkpointing**: Trades compute for memory (~30% memory savings)
   ```yaml
   training:
     gradient_checkpointing: true
   ```

2. **Lower batch size + gradient accumulation**: Maintain effective batch size
   ```yaml
   training:
     per_device_train_batch_size: 1
     gradient_accumulation_steps: 16  # Effective batch = 16
   ```

3. **Reduce sequence length**: Lower if samples don't need full context
   ```yaml
   global_max_length: 8192  # Down from 20000
   ```

### Training Speed

1. **Flash Attention 2**: ~2x faster attention
   ```yaml
   model:
     attn_impl: flash_attention_2
   ```

2. **Packing**: Eliminate padding waste (~30% faster)
   ```yaml
   training:
     packing: true
   ```

3. **bf16**: Faster than fp32, more stable than fp16
   ```yaml
   model:
     torch_dtype: bfloat16
   training:
     bf16: true
   ```

## Common Issues (FAQ)

### Issue: "ValueError: self.dataset: [], self.cached_dataset: []. Please input the training dataset."

**Cause:** Missing required `dataset: ["dummy"]` in config's `data:` section

**Solution:**
```yaml
data:
  dataset: ["dummy"]  # Required by ms-swift TrainArguments validation
```

**Why needed:** ms-swift validates non-empty dataset during `TrainArguments.__post_init__()` before our custom dataset loading. The placeholder satisfies validation but is never used. See `DATA_AND_DATASETS.md` for details.

### Issue: "Expected all tensors to be on the same device"

**Cause:** Mixed CPU/GPU tensors, often from custom preprocessing

**Solution:** Ensure all tensor operations happen on same device as model

### Issue: Training stuck at 0% GPU utilization

**Cause:** Data loading bottleneck

**Solution:**
```yaml
data:
  dataloader_num_workers: 16  # Increase workers
  dataloader_pin_memory: true
```

### Issue: Loss spikes periodically

**Cause:** Learning rate too high or batch too small

**Solutions:**
1. Lower LR: `learning_rate: 5.0e-5`
2. Increase effective batch size via gradient accumulation
3. Add warmup: `warmup_ratio: 0.1`

## Aligner tuning playbook (from archive)

Recommended minimal settings to train the aligner effectively while keeping the rest stable.

```yaml
tuner:
  train_type: lora
  freeze_llm: true
  freeze_vit: true
  freeze_aligner: true      # aligner trained via modules_to_save
  target_regex: '^$'        # no LoRA targets
  modules_to_save:
    - model.visual.merger
    - model.visual.deepstack_merger_list.0
    - model.visual.deepstack_merger_list.1
    - model.visual.deepstack_merger_list.2
training:
  optimizer: multimodal
  aligner_lr: 1.0e-4
  weight_decay: 0.1
  warmup_ratio: 0.3
```

Alternatives (when full-param aligner overfits or needs better dynamics):
- DoRA: `tuner.use_dora: true` (weight‑decomposed LoRA)
- AdaLoRA: adaptive rank to reveal bottlenecks in aligner
- BOFT: orthogonal fine‑tuning; preserves geometry space
- FourierFT: frequency‑domain adaptation for spatial patterns

Augmentation guidance for grounding tasks:
- Conservative geometric (small rotate/scale); aggressive appearance (color/gamma/CLAHE)
- Keep `pad_to_multiple` to stabilize image grid and token counts

Monitoring: track bbox/quad/line metrics separately; reduce geometric ops if quad/line drifts.

## Stage‑A implementation notes (from archive)

- Hybrid batching gives ~4–5× throughput vs sequential
- Strict validation: non‑empty summaries; contiguous `object_{n}` indices; deterministic ordering
- Native chat_template via HF; no custom wrapper required
- Flat JSONL per mission enables easy downstream GRPO loading

## Dense/Summary mixed mode design (from archive)

- Mode is chosen per sample via epoch-seeded RNG (dense vs summary)
- Summary mode requires valid `summary` on all records
- Selection is deterministic per epoch (seeded RNG)
- Dataset temporarily injects the appropriate system prompt per group during encoding

## Additional Resources

- **Training**: [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
- **Data**: [DATA_FORMATS.md](DATA_FORMATS.md)
- **Inference**: [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md)
- **Archived docs**: `archive/` (historical references, detailed technical guides)

---

**Last Updated**: October 25, 2025

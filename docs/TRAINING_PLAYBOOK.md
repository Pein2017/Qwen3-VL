# Qwen3‑VL Training Playbook

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
    vit:
      enabled: false
    aligner:
      enabled: true
      weight: 0.1
      distance: mse
    deepstack:
      enabled: true
      weight: 0.1
      distance: mse
```

#### Vision/Aligner Feature KD (optional)

When the vision encoder or aligner drifts while the language tower needs freedom to adapt (e.g., new coordinate formats), enable the feature distillation block with per-component control:

```yaml
custom:
  visual_kd:
    enabled: true
    vit:
      enabled: true           # ViT output before aligner (pre-merger)
      weight: 0.1
      distance: mse           # or `cosine`
    aligner:
      enabled: true           # Aligner/merger output (post-merger)
      weight: 0.1
      distance: mse
    deepstack:
      enabled: true           # Intermediate vision layer outputs
      weight: 0.1
      distance: mse
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
| NCCL monitoredBarrier after best checkpoint | Token accuracy aggregated per rank → only some processes call `_save_checkpoint`, others block in collectives | Upgrade to the 2025‑11‑16 trainer (`src/trainers/gkd_monitor.py`) which reduces `{token_acc_correct, token_acc_total}` across ranks before logging so every process makes the same best-model decision. Older builds should stick to `metric_for_best_model=eval_loss`. |

### Training Tooling & Scripts

- `scripts/train.sh` — Preferred launcher that wraps `python -m src.sft`/`torchrun`, resolves relative configs, sets CUDA/NCCL env defaults, and toggles debug mode via `DEBUG=true`.
- `scripts/fuse_datasets.py` — Offline fusion builder for `src/datasets/fusion.py`. Provide a YAML/JSON fusion config to pre-create deterministic JSONL mixtures for reproducibility or inspection.
- `scripts/inspect_lora_ckpts.py` — CPU-only inspector that enumerates LoRA/module-to-save parameters in a checkpoint directory, ensuring adapter exports look correct before deployment.

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

All geometric transforms automatically update bbox/poly/line coordinates to maintain spatial accuracy.

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

Monitoring: track bbox/poly/line metrics separately; reduce geometric ops if poly/line drifts.

# Reference

Status: Active — Internal Engineering

Quick reference for advanced topics and troubleshooting.

## Learning Rate Scheduler

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

## DeepSpeed Configuration

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

## Augmentation Pipeline

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

## Architecture Notes

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

## Chat Template Mechanics

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

## Performance Tips

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

## Common Issues

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

## Additional Resources

- **Training**: [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
- **Data**: [DATA_FORMATS.md](DATA_FORMATS.md)
- **Inference**: [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md)
- **Archived docs**: `archive/` (historical references, detailed technical guides)

---

**Last Updated**: October 25, 2025


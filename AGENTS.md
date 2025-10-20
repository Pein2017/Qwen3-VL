<!-- OPENSPEC:START -->
# OpenSpec Instructions

These instructions are for AI assistants working in this project.

Always open `@/openspec/AGENTS.md` when the request:
- Mentions planning or proposals (words like proposal, spec, change, plan)
- Introduces new capabilities, breaking changes, architecture shifts, or big performance/security work
- Sounds ambiguous and you need the authoritative spec before coding

Use `@/openspec/AGENTS.md` to learn:
- How to create and apply change proposals
- Spec format and conventions
- Project structure and guidelines

Keep this managed block so 'openspec update' can refresh the instructions.

<!-- OPENSPEC:END -->

## Qwen3‑VL — Project Guide (ms‑swift first, single‑repo)

This guide is the living index for the active codebase at `/data/Qwen3-VL`. It documents how to train, extend, and operate Qwen3‑VL with a minimal, config‑only surface that relies on two upstream libraries:

- `transformers` (HF): `/root/miniconda3/envs/ms/lib/python3.12/site-packages/transformers`
- `ms‑swift` (training/runtime): `/data/ms-swift`

The legacy `qwen2.5‑VL` stack is considered historical context only. We migrate its capabilities into this repo with tighter ms‑swift integration and far less custom infrastructure.

### Scope & intent
- **Single home**: All project work happens in `/data/Qwen3-VL`.
- **Upstream first**: Prefer features provided by `ms‑swift` and HF `transformers` over custom code.
- **Configuration‑driven**: YAML controls behavior; avoid CLI flags beyond paths/debug.
- **Tight template coupling**: Use the model’s native chat_template; do not hand‑craft `<|image_pad|>`.

### Directory map (this repo)
- `src/` — minimal training stack
  - `config/` (YAML load/merge → `TrainArguments`)
  - `datasets/` (preprocessors, builders, geometry, dynamic pairing, collators)
  - `sft.py` (entry; integrates `swift.llm.train.sft.SwiftSft`)
- `configs/` — experiment YAMLs (e.g., `stage_1_full_aligner_only.yaml`, `stage_2_llm_lora.yaml`)
- `scripts/` — helpers (`train.sh` auto‑DDP if multiple GPUs)
- `output/`, `tb/`, `vis_out/` — run artifacts

### Minimal entrypoints
```bash
# Script launcher (recommended; from anywhere)
conda run -n ms bash /data/Qwen3-VL/scripts/train.sh config=/abs/path/to/config.yaml gpus=0

# Direct module (from repo root /data/Qwen3-VL)
conda run -n ms python -m src.sft --config /abs/path/to/config.yaml [--base_config /abs/base.yaml] [--debug]
```

### Configuration model (YAML)
- **Model/template**: `template.template: qwen3_vl` (uses native chat_template)
- **Length**: `global_max_length` sets both `model.max_model_len` and `template.max_length`
- **Tuning**: `tuner.train_type: {full|lora}`; freezes via `freeze_llm/freeze_vit/freeze_aligner`; LoRA `target_modules`
- **Data**: `custom.train_jsonl`, `custom.val_jsonl`; grouping via `images_per_user_turn`; augmentation via `augment_prob`
- **Coords**: `custom.emit_norm: {none|norm100|norm1000}` (text only); template normalizes to norm1000 during encoding
- **Performance**: `training.packing: true` (padding‑free), DeepSpeed under `deepspeed.enabled/config`

Example (abridged):
```yaml
model:
  model: /abs/path/to/Qwen3-VL-4B-Instruct
template:
  template: qwen3_vl
  max_length: 4096
training:
  num_train_epochs: 3
  per_device_train_batch_size: 2
custom:
  train_jsonl: /abs/data/train.jsonl
  val_jsonl: /abs/data/val.jsonl
  emit_norm: norm1000
```

### Data contract (JSONL)
```json
{
  "images": ["path/to/img1.jpg", "path/to/img2.jpg"],
  "objects": [ {"bbox_2d": [x1,y1,x2,y2], "desc": "..."}, {"quad": [x1,y1,...,x4,y4], "desc": "..."} ],
  "width": 1920,
  "height": 1080,
  "summary": "optional"
}
```
- Paths resolved relative to the JSONL file’s directory
- Top‑level `objects` preserve pixel coordinates; template normalizes to norm1000 during encoding

### Training recipes
- **Stage 1 (Aligner‑only LoRA)** — start from base; freeze LLM+ViT
```yaml
tuner:
  train_type: lora
  target_modules: [all-linear]
  freeze_llm: true
  freeze_vit: true
  freeze_aligner: false
```

- **Stage 2 (LLM + Aligner LoRA)** — load stage‑1 adapter; freeze ViT
```yaml
model:
  model: /abs/path/to/base/Qwen3-VL
tuner:
  train_type: lora
  target_modules: [all-linear]
  freeze_llm: false
  freeze_vit: true
  freeze_aligner: false
  resume_from_checkpoint: /abs/path/to/output/stage_1/checkpoint-XXX
```

Inference with adapter (lightweight):
```bash
CUDA_VISIBLE_DEVICES=0 conda run -n ms swift infer \
  --model /abs/path/to/base/Qwen3-VL \
  --adapters /abs/path/to/output/stage_2/checkpoint-XXX \
  --stream true --max_new_tokens 2048
```

Merge LoRA into base (deployment):
```bash
CUDA_VISIBLE_DEVICES=0 conda run -n ms swift export \
  --model /abs/path/to/base/Qwen3-VL \
  --adapters /abs/path/to/output/stage_2/checkpoint-XXX \
  --merge_lora true --output_dir /abs/path/to/output/merged/checkpoint-XXX \
  --save_safetensors true
```

### Migration from qwen2.5‑VL (conceptual map)
- **Templates**: Hand‑crafted tokens → model’s native `chat_template` (HF `transformers`)
- **Training stack**: Custom runners → `ms‑swift` `SwiftSft` pipeline with typed `TrainArguments`
- **Grouping & packing**: Legacy packers → `training.packing: true` (padding‑free, Qwen3‑VL compatible)
- **Geometry**: Same JSONL contract; normalization handled by template at encode time
- **Configs**: Strict YAML; no in‑code defaults; inheritance via `extends`

### Where to add new capabilities
- **Preprocessors**: `src/datasets/preprocessors/` — implement against `BasePreprocessor`
- **Builders**: `src/datasets/builders/` — implement against `BaseBuilder`
- **Geometry utils**: `src/datasets/geometry.py` — pure transforms; I/O at edges
- **Prompts**: `src/config/prompts.py`; configuration under `prompts.scheme/system/user`
- **Runner**: Prefer composition via YAML; keep `sft.py` thin

### Performance & scaling tips
- `torch_dtype=bfloat16`, `attn_impl=flash_attn`, `gradient_checkpointing=true`
- Tune `global_max_length` and enable `training.packing` for utilization
- Use DeepSpeed ZeRO when scaling batch size across GPUs

### Health checks (fail‑fast)
- Image placeholders (`<image>`) count equals images in the user turn
- `image_grid_thw` matches expanded vision tokens; `pixel_values` dimensions consistent
- Assistant spans include end‑of‑turn; non‑target tokens labeled −100; image tokens masked
- Geometry preserved at top level; template normalizes to norm1000; alignment verified post‑encode

### Troubleshooting
- **Full model saved, not adapter**: Ensure `sft.prepare_model()` is called before creating the trainer
- **`modules_to_save` empty**: Verify LoRA targets; check wrapped model type (SwiftModel/PeftModel)
- **Path errors**: Use absolute paths; dataset image paths are relative to JSONL directory
- **OOM/length issues**: Lower `global_max_length`; enable packing; reduce per‑device batch size

### Notes
- This repo is intentionally config‑driven and minimal; add features here rather than reviving legacy stacks
- Use absolute paths when launching from outside `/data/Qwen3-VL`; scripts print resolved settings
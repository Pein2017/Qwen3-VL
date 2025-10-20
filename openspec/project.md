# Project Context

## Purpose
Single‑repo home for Qwen3‑VL training and extension under `/data/Qwen3-VL`, migrating from historical `qwen2.5‑VL` infra to a simpler, ms‑swift–first stack. Goals:
- Dense captioning with structured geometry (bbox/quad/line) at scale
- Minimal, configuration‑only surface; upstream first (ms‑swift + HF transformers)
- Clear extension points (preprocessors, builders, prompts) without custom runners
- Reproducible training, packing‑friendly, and easy deployment (adapter or merged)

## Tech Stack
- Language: Python 3.12 (conda env: `ms`)
- Core libs:
  - PyTorch (torch) 2.8.0+cu128
  - Transformers 4.57.1 — `/root/miniconda3/envs/ms/lib/python3.12/site-packages/transformers`
  - ms‑swift 3.10.0.dev0 — `/data/ms-swift`
  - TRL 0.23.1 (optional)
  - DeepSpeed (optional; ZeRO‑2 config)
- Hardware: 8 × NVIDIA A100 80GB (padding‑free training supported)

## Source Layout (high‑level)
- `src/` — minimal training stack
  - `config/` — YAML load/merge → `TrainArguments`; prompt resolution
  - `datasets/` — preprocessors, builders, geometry, augmentation, dynamic pairing, collators
  - `sft.py` — entry; integrates `swift.llm.train.sft.SwiftSft` and `TrainerFactory`
- `configs/` — experiment YAMLs (`stage_1_full_aligner_only.yaml`, `stage_2_llm_lora.yaml`, etc.)
- `scripts/` — helpers (e.g., `train.sh`, adapter merge/inspect)
- `output/`, `tb/`, `vis_out/` — run artifacts

## Data Contract (JSONL)
```json
{
  "images": ["path/to/img1.jpg", "path/to/img2.jpg"],
  "objects": [ {"bbox_2d": [x1,y1,x2,y2], "desc": "..."}, {"quad": [x1,y1,...,x4,y4], "desc": "..."} ],
  "width": 1920,
  "height": 1080,
  "summary": "optional"
}
```
- Image paths are resolved relative to the JSONL file directory
- Top‑level `objects` keep pixel coordinates; the template normalizes to norm1000 during encoding

## Architecture & Conventions
- Configuration‑driven: All hyperparameters must come from YAML; CLI only for paths/debug
- Template‑first: Use the model’s native `chat_template`; never hand‑craft `<|image_pad|>`
- Separation of concerns:
  - Data (JSONL) → Preprocessing (row) → Building (pair) → Encoding (template) → Training
  - `datasets/geometry.py` is pure transforms; I/O only at edges
- Extension points (minimal, explicit):
  - `BasePreprocessor.preprocess(row) -> row | None`
  - `BaseBuilder.build(a, b) -> {"messages": ..., "images": ..., ["objects": ...]}`
  - Prompt registration in a single place (`src/config/prompts.py`)
- Reproducibility: epoch‑seeded pairing; no hidden globals; explicit RNG/config injection
- Health checks (fail‑fast): placeholder count, `image_grid_thw`, masked image tokens, assistant span includes end‑of‑turn, geometry normalization to norm1000

## Workflows
### Entrypoints
```bash
# Script launcher (recommended)
conda run -n ms bash /data/Qwen3-VL/scripts/train.sh config=/abs/config.yaml gpus=0

# Direct module (from repo root)
conda run -n ms python -m src.sft --config /abs/config.yaml [--base_config /abs/base.yaml] [--debug]
```

### Training Recipes
- Stage 1 (Aligner‑only, freeze LLM+ViT)
  - `tuner.train_type: full` with `freeze_llm: true`, `freeze_vit: true`, `freeze_aligner: false`
- Stage 2 (LLM LoRA + frozen ViT; aligner via `modules_to_save`)
  - `tuner.train_type: lora`, `target_modules: [all-linear]`, freeze ViT
  - `modules_to_save: [model.visual.merger, model.visual.deepstack_merger_list.{0,1,2}]`

### Adapter Preparation (critical)
Always call `sft.prepare_model(train_args, model, template, train_dataset)` before creating the Trainer. This wraps the model (SwiftModel/PEFT), applies LoRA, sets `modules_to_save`, and ensures adapter‑only checkpoints.

### Inference & Export
- Lightweight: load base + adapter via `swift infer --adapters ...`
- Deployment: `swift export --merge_lora true` to bake LoRA into base; use merged model for production or further full tuning

## Configuration Conventions (YAML)
- Sections: `model`, `template`, `tuner`, `training`, `data`, `prompts`, `custom`, `deepspeed`
- Inheritance: `extends` or `inherit` (string or list); earlier bases are lower precedence; cycles error
- Global length: `global_max_length` sets both `model.max_model_len` and `template.max_length`
- Prompts: `prompts.scheme: A|B`; resolves to `template.system`; `custom.user_prompt` consumed by dataset
- Custom keys (consumed by `src/sft.py` and datasets):
  - `train_jsonl`, `val_jsonl`
  - `images_per_user_turn` (default 2)
  - `augment_prob` (geometry‑aware augmentation)
  - `emit_norm: none|norm100|norm1000` (text only; template normalizes geometry to norm1000)
  - `dump_conversation_text` and optional `dump_conversation_path`
- DeepSpeed: `deepspeed.enabled: true` + `config: zero2` (see `configs/deepspeed_zero2_lora.json`)

## Prompts
- Scheme A: minimal/prior‑free format enforcement
- Scheme B: informative, with ordering/taxonomy hints and domain priors (e.g., BBU挡风板规则)
- User prompt: requests grouped JSON output with norm1000 integer coordinates

## Testing & QA
- Smoke check: enable `--debug` or `custom.dump_conversation_text: true` to dump one decoded conversation
- Validation: provide `custom.val_jsonl` and set eval cadence in `training.eval_strategy/steps`
- Logging: `training.logging_dir` (per‑run subdir is auto‑appended via `run_name`)

## Git Workflow & Conventions
- Branch: `main` (tracking `origin/main`)
- Commits: Conventional Commits style (`feat:`, `fix:`, `docs:`, `chore:`, etc.)
- PRs: small, focused; reference configs/files touched; include rationale and expected effects on training
- Releases: tag milestones (e.g., `v0.1.0-migration`) and update CHANGELOG

## External Dependencies
- HF Transformers (template/tokenizer, processors)
- ms‑swift (TrainArguments, SwiftSft, TrainerFactory, LoRA/adapters)
- Optional: TRL (RL experiments), DeepSpeed (ZeRO for multi‑GPU)

## Important Constraints
- Template enforces image placeholders; count must equal attached images
- `image_grid_thw` must match expanded vision tokens; `pixel_values` dimensions consistent
- Assistant spans include end‑of‑turn; non‑target tokens labeled −100; image tokens masked
- Geometry preserved in top‑level `objects` (pixels) and normalized to norm1000 during encoding

## Version Matrix (current)
- torch: 2.8.0+cu128
- transformers: 4.57.1 (`/root/miniconda3/envs/ms/.../transformers`)
- ms‑swift: 3.10.0.dev0 (`/data/ms-swift`)
- trl: 0.23.1 (optional)

## Roadmap & Migration Notes
- Ongoing migration from `qwen2.5‑VL` custom infra → `Qwen3‑VL` + ms‑swift
- Gradually add legacy‑parity features as clean builders/preprocessors/templates
- Keep surface minimal; prefer config composition over new flags or ad‑hoc utilities

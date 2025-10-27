# Training

Source of truth: `src/sft.py`, `src/README.md`, `docs/TRAINING_GUIDE.md`

## Essentials
- YAML-only surface; avoid CLI flags beyond `--config` (and `--base_config`, `--debug`)
- Set `global_max_length` as the single length knob
- Always call `sft.prepare_model(...)` before creating the trainer (adapters, freezes, modules_to_save)

## Modes
- Full fine-tuning: maximum flexibility, highest memory
- LoRA: adapter-only (~240MB); preferred for iteration and deployment
- Selective freezing: control `freeze_llm|freeze_vit|freeze_aligner`

## Two-stage recipe (recommended)
1) Stage 1: Aligner-only LoRA (freeze LLM+ViT) → learn alignment
2) Stage 2: LLM+Aligner LoRA (freeze ViT) → refine language while preserving alignment

## Packing (padding-free)
- `training.packing: true` for higher utilization; incompatible with `lazy_tokenize`

## Health checks
- Vision tokens present (`pixel_values`, `image_grid_thw`)
- Image placeholders match image count
- `modules_to_save` lists any full-tuned modules (if used)

## Troubleshooting (top)
- Full model saved instead of adapter → missing `sft.prepare_model()`
- Zero grad for vision/aligner → content items must use `{ "type": "image", "image": path }`
- OOM → lower batch size/length, enable gradient checkpointing, use ZeRO

## See also
- Data: `docs/DATA.md`  ·  Datasets: `docs/DATASETS.md`  ·  Architecture: `docs/ARCHITECTURE.md`

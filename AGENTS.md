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

## Snapshot
- Single repository for the Qwen3‑VL training stack under `/data/Qwen3-VL`; ms‑swift is the orchestration layer and Hugging Face transformers supply the model/template implementations.
- Configuration drives behavior. Prefer editing YAML under `configs/` or config helpers in `src/config/` over introducing ad-hoc CLI flags.
- Training artifacts live under `output/`, `tb/`, and `vis_out/`; keep them out of commits.

## Core Surface
- `src/` — code:
  - `sft.py` launches supervised fine tuning via `swift.llm.train.sft.SwiftSft`.
  - `config/` resolves YAML into `TrainArguments`, prompt schemes, and template wiring.
  - `datasets/` covers geometry helpers, augmentation, preprocessors, builders, and dynamic pairing.
  - `utils/`, `callbacks/`, `stage_a/`, `stage_b/` hold supporting logic for training variants.
- `docs/` — authoritative background (augmentation, data prep, reference workflows).
- `configs/` — experiment presets (stages 1‑4, summary variants, debug/base overlays).
- `scripts/` — runnable helpers: `train.sh`, adapter merge/inspection, GRPO runner, LoRA tools.
- `vis_tools/` — visualization utilities for augmentation, raw samples, and crop debugging.

## How to Run
```bash
# Recommended launcher (handles env + DDP sizing)
conda run -n ms bash scripts/train.sh config=/abs/path/to/config.yaml gpus=0

# Direct module entry point from repo root
conda run -n ms python -m src.sft --config /abs/path/to/config.yaml [--base_config /abs/base.yaml] [--debug]
```

## Development Expectations
- Favor upstream capabilities in ms‑swift or transformers before adding custom modules.
- Keep augmentations and data handling geometry-aware; run `vis_tools/vis_augment_compare.py` for spot checks when touching `src/datasets/augmentation/`.
- Update documentation in `docs/` when behavior or workflows change; stage config examples in `configs/` should remain runnable.
- For feature or spec work, consult `openspec/AGENTS.md` and follow the change process there.

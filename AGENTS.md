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
  - `config/` resolves YAML into `TrainArguments`, prompts, and template wiring.
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

## Stage-A Inference
- Entry point `python -m src.stage_a.cli` wraps `run_stage_a_inference` and enforces typed config + validation.
- Inputs: mission-scoped image tree `<root>/<mission>/{审核通过|审核不通过}/<group_id>/*.jpg`; checkpoints must be on disk.
- Outputs: one JSONL per mission at `<output_dir>/<mission>_stage_a.jsonl` with `group_id`, `mission`, `label`, ordered `per_image` summaries.
- Generation stack: `AutoProcessor` + `Qwen3VLForConditionalGeneration`, batched with optional verification logs; `max_pixels` defaults to `786432` for throughput.
- Prompt alignment: prompts import from `src.config.prompts`; mission focus text comes from `src.config.missions.STAGE_A_MISSION_FOCUS`.
- CLI flags expose device, batch size, generation params, prompt focus, and verification mode; all numeric inputs validated before model load.

## Stage-B Reflection Pipeline
- Run via `python -m src.stage_b.runner --config /abs/path/to/config.yaml [--log-level {debug|logging|warning}]`; the pipeline currently executes the full loop (`--step all`).
- Config schema lives in `src.stage_b.config` (frozen dataclasses); values come from YAML under `configs/stage_b/`.
- High-level flow per mission:
  1. `ingest_stage_a` normalizes Stage-A JSONL into `GroupTicket` objects and ensures mission guidance exists.
  2. `RolloutSampler` builds prompts with mission guidance, decodes multiple samples per ticket using the configured grid.
  3. `attach_signals` annotates candidates with deterministic metrics (label match, consistency, confidence heuristics).
  4. `select_for_group` chooses the winner using semantic advantage + tie-break policy, exporting trajectories/selections incrementally.
  5. `ReflectionEngine` batches experience records, calls the in-process model with `reflection.prompt_path`, and writes guidance updates through `GuidanceRepository` snapshots.
  6. Optional holdout split + `evaluate_holdout` report label-match uplift around each reflection cycle.
- Outputs are written under `{output.root}/{output.run_name}/{mission}/` with `trajectories.jsonl`, `selections.jsonl`, per-mission parquet, `guidance.json`, and `reflection.jsonl` logs. Guidance snapshots rotate in `snapshots/` respecting retention.
- Guidance lifecycle: `GuidanceRepository` lazily initializes empty guidance, enforces non-empty experience dicts, and snapshots previous files before each write. Reflection proposals require structured `[Gx].` entries; parsing failures fall back to noop without mutating guidance.
- Sampling prompts live in `src.stage_b.prompts` (pipeline) and `src.stage_b.sampling.prompts` (LLM bundle rendering); both expect ordered `per_image` maps from Stage-A.
- Deterministic judge + metrics helpers are in `src.stage_b.signals`, `src.stage_b.selection`, and `src.stage_b.scoring` for reuse in GRPO-style evaluation.

### Stage-B Config Highlights
- `stage_a_paths`: absolute or relative JSONL files from Stage-A; multiple paths allowed.
- `model`: `model_name_or_path`, `torch_dtype`, `device_map` forwarded to `Qwen3VLForConditionalGeneration` and tokenizer.
- `sampler`: `grid` of decode configs (`temperature`, `top_p`, `max_new_tokens`, optional `seed`, `stop`), `samples_per_decode`, and `format_filter` toggle.
- `signals`: enable/disable confidence/self-consistency and override semantic weights.
- `reflection`: prompt template path, batch size, delta threshold, change cap per epoch, diversity parameters, and `max_reflection_length` guard.
- `selection`: policy (`top_label` or `top_semantic`) and tie-break (`confidence` or `temperature`).
- `output`: root directory + run name; mission folders created automatically; parquet path optional override.
- `runner`: epoch count; `evaluation`: holdout size + metrics list.
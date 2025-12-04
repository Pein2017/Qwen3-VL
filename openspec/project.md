# Project Context

## Purpose & Scope
- Single home for Qwen3-VL supervised training, evaluation, and feature work under `/data/Qwen3-VL`.
- Uses ms-swift (`/data/ms-swift`) for training orchestration and Hugging Face transformers for model/template implementations.
- Focus: dense captioning with structured geometry (bbox/quad/line) plus augmentation and visualization tooling.

## Spec Status
- Active, production-ready specs: `specs/sft-training`, `specs/fusion-dataset`, `specs/multi-dataset-fusion`, `specs/stage-b-training-free`, `specs/data-augmentation`, `specs/hard-sample-mining`.
- Padding-only runtime: `training.packing` is not supported; legacy packing changes live in `openspec/archive/changes/` (`2025-12-02-add-grouped-packing-wrapper`, `2025-12-02-add-packing-optimizations`, `add-unpacked-group-telemetry`).
- Experimental/in-progress changes (open tasks): `changes/add-language-fusion-wrapper`, `changes/refactor-grouped-metrics-sync`, `changes/refactor-stage-b-simple-guidance`.
- Completed/production-ready changes: `changes/remove-packing-path`, `changes/update-fusion-target-only-policies`, Stage-B reflection/guardrail changes (`add-stageb-3step-reflection`, `add-stageb-guardrails`, `align-stageb-chat-convo`, `refactor-stageb-line-protocols`, `refactor-stageb-two-line-protocol`).
- Legacy (not supported): `specs/packing-optimizations` kept for traceability only; padding-only runtime supersedes it.
- Use `openspec list` for current status before editing specs or configs.

## Quick Orientation
1. **Run surface**
   - `scripts/train.sh` (preferred launcher) and `src/sft.py` (Python entry point).
   - Conda env: `ms`; call binaries via `conda run -n ms ...`.
2. **Configs**
   - `configs/base.yaml` sets defaults; stage configs (`stage_1`–`stage_4`, `summary*.yaml`) layer behavior for specific training phases.
   - Summary of new knobs should live alongside the config file.
3. **Docs**
   - `docs/README.md` — onboarding & repo goals.
   - `docs/data/DATA_AUGMENTATION.md` — authoritative augmentation/geometry spec.
   - `docs/data/DATA_AND_DATASETS.md` — JSONL schema, preprocessing expectations.
   - `docs/experiments/` — archived studies and recommended hyperparameters.
4. **Visual Tooling**
   - `vis_tools/vis_augment_compare.py` (side-by-side augmentation check).
   - `vis_tools/vis_qwen3.py`, `vis_tools/vis_raw.py` for data sanity.

## Source Layout Highlights
- `src/sft.py` — loads YAML (`src/config/loader.py`), prepares the model via `swift.llm.train.sft.SwiftSft`, and wires datasets/augmenters.
- `src/config/` — prompts, TrainArguments adapters, config merging helpers.
- `src/datasets/`
  - `geometry.py` — canonical geometry math (affines, clipping, coverage, normalization).
  - `augmentation/` — operator registry, Compose pipeline, YAML builder.
  - `preprocessors/` — record-level transforms (augmentation, dense caption prep).
  - `builders/` — chat/user turn assembly (JSONLinesBuilder, etc.).
  - `dense_caption.py` — dataset orchestrating single-image records, augmentation, and mode selection.
- `src/utils/`, `callbacks/`, `stage_a/`, `stage_b/` — supplemental glue for training variants and callbacks.
- `tests/augmentation/` — current unit tests; extend when touching geometry/augment code paths.

## Data Contract (JSONL)
```json
{
  "images": ["rel/path/to.jpg"],
  "objects": [
    {"bbox_2d": [x1, y1, x2, y2], "desc": "..."},
    {"quad": [x1, y1, x2, y2, x3, y3, x4, y4], "desc": "..."},
    {"line": [x1, y1, x2, y2, ...], "desc": "..."}
  ],
  "width": 768,
  "height": 1024,
  "summary": "optional"
}
```
- Paths resolve relative to the JSONL directory; `sft.py` sets `ROOT_IMAGE_DIR` accordingly.
- Geometry stays in pixel space until the template normalizes to `norm1000` during encoding.

## Configuration Patterns
- All knobs flow through YAML. CLI flags are limited to `--config`, `--base_config`, and `--debug`.
- Common sections: `model`, `template`, `tuner`, `training`, `data`, `prompts`, `custom`, `deepspeed`.
- `custom` keys commonly used:
  - `train_jsonl`, `val_jsonl`
  - `augmentation` → `ops` list consumed by `src/datasets/augmentation/builder.py`
  - `emit_norm` (`none|norm100|norm1000`), `user_prompt`, `use_summary`
  - `bypass_prob` (augmentation skip rate)
- DeepSpeed config JSON lives under `configs/` when needed.

## Workflow Cheatsheet
1. Select or clone a base YAML in `configs/`.
2. Update/extend preprocessors or augmentation as needed in `src/datasets/`.
3. Modify prompts or config helpers in `src/config/` if customization is needed.
4. Keep documentation in sync (relevant `docs/*.md`, config file headers).
5. Validate with unit tests + visualization + targeted training run (or documented plan).
6. For feature work, follow the OpenSpec process (`openspec/AGENTS.md`) and archive changes once shipped.

## Validation & Tooling
- Unit tests: `pytest tests/augmentation -k <name>`.
- Visualization: `python vis_tools/vis_augment_compare.py --config /abs/config.yaml --sample 4`.
- Quick dataset dump: enable `custom.dump_conversation_text` in YAML and run with `--debug` to print encoded conversations.
- Training smoke: subset JSONL, reduce epochs/steps in config, and run via `scripts/train.sh`.

## Coding Conventions
- Pure functions for geometry/math; avoid side effects in `src/datasets/geometry.py`.
- Registry-based augmentation: register new ops in `src/datasets/augmentation/ops.py` and expose through YAML builder.
- Prefer small, composable modules; watch for duplicated logic between docs and code (keep docs authoritative).
- Tests should assert both geometry integrity and metadata (kept indices, completeness labels).

## Key External Versions
- Python 3.12 (conda env `ms`)
- torch 2.8.0+cu128
- transformers 4.57.1
- ms-swift 3.10.0.dev0
- Optional: trl 0.23.1, DeepSpeed (ZeRO-2)

## Useful References
- `compare/data-augmentation-potentails/final_analysis.md` — latest augmentation audit and risk list.
- `docs/CHANGELOG.md` — timeline of major updates.
- `scripts/inspect_lora_ckpts.py`, `scripts/merge_stage2_lora.sh` — adapter workflows.

Keep this document concise and up to date so new contributors can ramp quickly and understand where to make and validate changes.

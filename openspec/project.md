# Project Context

## Purpose & Scope
- Single home for Qwen3-VL supervised training, evaluation, and feature work under `/data/Qwen3-VL`.
- Uses ms-swift (`/data/ms-swift`) for training orchestration and Hugging Face transformers for model/template implementations.
- Focus: dense captioning with structured geometry (bbox/poly/line) plus augmentation and visualization tooling.

## Spec Status
- Active, production-ready specs: `specs/sft-training`, `specs/fusion-dataset`, `specs/multi-dataset-fusion`, `specs/grpo-integration`, `specs/summary-grpo-post-training`, `specs/llm-kd-config`, `specs/data-augmentation`, `specs/detection-preprocessor`, `specs/hard-sample-mining`, `specs/stage-b-training-free`, `specs/chatml-stageb`, `specs/llm-free-reflection`.
- Summary-mode GRPO post-training is fully specified by `specs/summary-grpo-post-training` (supersedes summary guidance in `specs/grpo-integration`).
- Padding-only runtime: `training.packing` is not supported; legacy packing changes are archived in `openspec/archive/changes/` (`2025-12-02-add-grouped-packing-wrapper`, `2025-12-02-add-packing-optimizations`, `add-unpacked-group-telemetry`) and `openspec/changes/archive/2025-12-30-remove-packing-path`.
- Legacy (not supported): `specs/packing-optimizations` kept for traceability only; padding-only runtime supersedes it.
- Active change proposals: *(none; bulk archived on 2025-12-30 - see `openspec/changes/archive/2025-12-30-*`)*.
- Archived/deprecated changes: legacy bundles in `openspec/changes/archive/2025-01-27-deprecated-specs` and `openspec/changes/archive/2025-12-16-deprecated-specs`, plus older history under `openspec/archive/changes/`.
- Use `openspec list` and `openspec list --specs` for current status before editing specs or configs.

## Quick Orientation
1. **Run surface**
   - `scripts/train.sh` (preferred launcher) and `src/sft.py` (Python entry point).
   - Conda env: `ms`; call binaries via `conda run -n ms ...`.
2. **Configs**
   - `configs/base.yaml` sets defaults; stage configs (`stage_1`-`stage_4`, `summary*.yaml`) layer behavior for specific training phases.
   - Summary of new knobs should live alongside the config file.
3. **Docs**
   - `docs/README.md` - onboarding, directory map, and recent updates.
   - `docs/data/DATA_AUGMENTATION.md` - authoritative augmentation/geometry spec.
   - `docs/data/DATA_AND_DATASETS.md` - JSONL schema, preprocessing expectations.
   - `docs/training/TRAINING_PLAYBOOK.md`, `docs/training/REFERENCE.md`, `docs/training/GRPO_MS_SWIFT_PIPELINE.md` - training runbooks.
   - `docs/runtime/STAGE_A_RUNTIME.md`, `docs/runtime/STAGE_B_RUNTIME.md`, `docs/runtime/STAGE_A_STAGE_B.md` - inference runbooks + business guidance.
4. **Visual Tooling**
   - `vis_tools/vis_augment_compare.py` (side-by-side augmentation check).
   - `vis_tools/vis_qwen3.py`, `vis_tools/vis_raw.py` for data sanity.

## Source Layout Highlights
- `src/sft.py` - loads YAML (`src/config/loader.py`), prepares the model via `swift.llm.train.sft.SwiftSft`, and wires datasets/augmenters.
- `src/config/` - prompts, TrainArguments adapters, config merging helpers.
- `src/rlhf/` - summary GRPO reward registry and hooks.
- `src/datasets/`
  - `geometry.py` - canonical geometry math (affines, clipping, coverage, normalization).
  - `augmentation/` - operator registry, Compose pipeline, YAML builder.
  - `preprocessors/` - record-level transforms (augmentation, dense caption prep).
  - `builders/` - chat/user turn assembly (JSONLinesBuilder, etc.).
  - `dense_caption.py` - dataset orchestrating single-image records, augmentation, and mode selection.
- `src/utils/`, `callbacks/`, `stage_a/`, `stage_b/` - supplemental glue for training variants and callbacks.
- `tests/augmentation/` - current unit tests; extend when touching geometry/augment code paths.

## Data Contract (JSONL)
```json
{
  "images": ["rel/path/to.jpg"],
  "objects": [
    {"bbox_2d": [x1, y1, x2, y2], "desc": "..."},
    {"poly": [x1, y1, x2, y2, x3, y3, ...], "desc": "..."},
    {"line": [x1, y1, x2, y2, ...], "desc": "..."}
  ],
  "width": 768,
  "height": 1024,
  "summary": "optional"
}
```
- Paths resolve relative to the JSONL directory; `sft.py` sets `ROOT_IMAGE_DIR` accordingly.
- `poly` covers any polygon (including quads); conversion canonicalizes ordering for consistent visualization.
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
- `docs/README.md` - directory map and update timeline.
- `docs/training/GRPO_MS_SWIFT_PIPELINE.md` - GRPO rollout/trainer notes.
- `src/datasets/changelog.md` - dataset contract changes.
- `scripts/merge_stage2_lora.sh` - adapter workflow.

Keep this document concise and up to date so new contributors can ramp quickly and understand where to make and validate changes.

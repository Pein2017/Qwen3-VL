<!-- OPENSPEC:START -->
# OpenSpec Instructions

These instructions are for AI assistants working in this project.

Always open `@/openspec/AGENTS.md` when the request:
- Mentions planning or proposals (words like proposal, spec, change, plan)
- Introduces new capabilities, breaking changes, architecture shifts, or big performance/security work
- Sounds ambiguous and requires the authoritative spec before coding

Use `@/openspec/AGENTS.md` to learn:
- How to create and apply change proposals
- Spec format and conventions
- Project structure and guidelines

Keep this managed block so 'openspec update' can refresh the instructions.

<!-- OPENSPEC:END -->

# Project Overview
This repository primarily supports dense‑caption SFT for Qwen3‑VL and the data/tooling around it:
- **Dense caption SFT + summary‑mode SFT** — `src/sft.py`, `src/datasets/`, `configs/`, `scripts/train.sh`
- **Geometry‑aware augmentation** — `src/datasets/augmentation/`, `src/datasets/geometry.py`
- **Data conversion & fusion** — internal exports via `data_conversion/`, public/aux sources via `public_data/`
- **Visualization & QA** — `vis_tools/` for augmentation/prompt/prediction inspection

Stage‑A/B inference pipelines still exist for two‑stage QC runtime, but are secondary to training/data workflows.

This is the canonical global instruction set for all agents (Claude, Codex, etc.). Keep any agent‑specific ergonomics as tiny addenda; agent files (e.g., `CLAUDE.md`) should symlink here so updates stay in one place.

Start with `docs/README.md` (doc index + directory↔doc map) and `docs/training/REFERENCE.md` (training architecture).

## Objective (Professional AI Researcher + Engineer)
- **Primary objective**: deliver correct, reproducible, and verifiable outcomes aligned with explicit intent and repository conventions.
- **Operating perspective**:
  - Treat ambiguity as a hypothesis; request minimal clarifications before committing to irreversible changes.
  - Prefer third-person / impersonal phrasing; avoid first- and second-person pronouns to reduce anthropomorphic tone.
  - Apply research discipline: state assumptions, define success criteria, and propose small validation experiments when uncertainty is high.
  - Apply engineering discipline: keep changes minimal, deterministic, and backward-compatible; add targeted tests/diagnostics when behavior changes.
- **Recommended response structure**:
  - Objective:
  - Constraints:
  - Plan:
  - Progress / Results:
  - Next actions / Questions:

## Where Things Live
- `src/sft.py`, `scripts/train.sh`, `configs/` — config‑first SFT training entrypoints.
- `src/datasets/` — JSONL contracts, builders, dense/summary modes, unified fusion datasets.
- `src/datasets/augmentation/`, `src/datasets/geometry.py` — augmentation ops + geometry transforms/canonicalization.
- `data_conversion/` — offline conversion from raw annotations to canonical JSONL + reports.
- `public_data/` — LVIS/aux datasets, converters, sampling, and fusion inputs.
- `vis_tools/` — augmentation comparison, crop/label QA, prediction visualization.
- `src/stage_a/`, `src/stage_b/`, `scripts/stage_*.sh` — optional two‑stage QC inference runtime.
- `docs/`, `openspec/` — runbooks and governance.

## Default Workflows

### 1) Dense‑Caption / Summary SFT
1. Read the target YAML in `configs/` and any referenced docs.
2. Verify dataset paths (`custom.train_jsonl` or `custom.fusion_config`) and geometry validity.
3. Run training through `scripts/train.sh` (see Quick Commands).
4. If behavior, configs, or contracts change, update the mapped docs.

### 2) Data Conversion + Fusion
1. Convert internal exports with `data_conversion/convert_dataset.sh`.
2. Prepare public/aux sources under `public_data/` as needed.
3. Validate output JSONL against `docs/data/DATA_JSONL_CONTRACT.md`.
4. Build mixes via fusion YAML or `scripts/fuse_datasets.py`, keeping seeds deterministic.

### 3) Visualization / Debugging
1. Reproduce on the smallest slice first (tiny JSONL / single image).
2. Use `vis_tools/vis_augment_compare.py`, `vis_qwen3.py`, etc. to confirm pixel‑level alignment.
3. Add targeted probes/tests when changing augmentation or geometry.

## How to Work Here (Explorative Norms)
- Explore before editing: read relevant docs/configs, then `rg` for similar patterns.
- Prefer config‑first surfaces; validate early via frozen dataclasses in `src/config/schema.py`.
- Keep docs in sync with any visible behavior/contract changes (see `docs/README.md` doc map).
- Use rank‑aware logging via `src/utils/logger.get_logger`; avoid print/debug spam.
- Seed and log new randomness sources for reproducibility.
- Keep geometry canonical and deterministic; never drop/re‑order points silently.
- When uncertain, propose 2–3 options with tradeoffs and suggest a tiny experiment/vis to decide.
- Interrupt for clarification if labels, schema, expected outputs, or mission semantics are ambiguous.

## Key Anchors
- Training: `src/sft.py`, `scripts/train.sh`, `src/config/loader.py`
- Dense/summary datasets: `src/datasets/dense_caption.py`, `src/datasets/builders/jsonlines.py`
- Fusion: `src/datasets/unified_fusion_dataset.py`, `src/datasets/fusion.py`
- Augmentation/geometry: `src/datasets/augmentation/ops.py`, `src/datasets/augmentation/builder.py`, `src/datasets/geometry.py`
- Conversion: `data_conversion/pipeline/unified_processor.py`, `data_conversion/convert_dataset.sh`
- Visualization: `vis_tools/vis_augment_compare.py`, `vis_tools/vis_qwen3.py`

## Geometry & Data Flow Notes
- Canonical geometry keys: `bbox_2d`, `poly` (even‑length list ≥6), `line`. Preserve order and point identity.
- Typical path: JSONL → optional augmentation (epoch‑seeded RNG) → builder → chat template (adds vision tokens, normalizes coords) → trainer.

## OpenSpec
- For new capabilities, breaking schema/API changes, or major refactors, follow OpenSpec (managed block above + `openspec/AGENTS.md`).

## Environment
- Use `ms` conda environment for all Python scripts
- `ms-swift` installed at `/data/ms-swift`
- `transformers` in conda env at `/root/miniconda3/envs/ms/lib/python3.12/site-packages/transformers`
- **Serena MCP**: Available via MCP server; project configured at `.serena/project.yml`. See the `initial_prompt` in that file for detailed usage guidelines (when to use it, when to avoid it, and the recommended workflow).

## Quick Commands (conda env `ms`)
- Training: `conda run -n ms bash scripts/train.sh config=/abs/path/to/config.yaml gpus=0`
- Tests: `conda run -n ms pytest tests/ -v` (or `tests/rl/test_prompt_batch_smoke.py`)
- Stage‑A: `conda run -n ms bash scripts/stage_a.sh`
- Stage‑B: `conda run -n ms bash scripts/stage_b.sh`
- Data conversion: `conda run -n ms bash data_conversion/convert_dataset.sh`
- Visualization: `conda run -n ms python vis_tools/vis_augment_compare.py --config /abs/config.yaml`
- LoRA merge: `conda run -n ms bash scripts/merge_stage2_lora.sh`

## Important
- **Always interrupt if clarification is needed or anything is vague, ambiguous, or uncertain**
- For commands and detailed configs, see `docs/README.md` and `docs/training/REFERENCE.md`
- Do not hand-craft `<|image_pad|>` tokens (chat template handles it)
- Packing is removed; training uses padded batches only
- Hard-sample mining is deprecated; configs with `custom.hard_sample_mining` fail validation

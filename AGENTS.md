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

## Project Overview
AI quality‑inspection system with two stages:
- **Stage‑1 (Stage‑A) Basic Object Recognition** — single‑image evidence capture with rare/long‑tail object coverage.
- **Stage‑2 (Stage‑B) Group‑Ticket Verification** — consumes Stage‑1 evidence + labels to issue binary `pass|fail` verdicts with auditable rationale.

Training, inference, and guidance workflows share a single repository. Start with `docs/README.md` (index) and `docs/REFERENCE.md` (architecture map).

## Component Roles
- **Stage‑1 (Stage‑A) Inference** — `src/stage_a/`, `scripts/stage_a_infer.sh`; owns per‑image prompts, mission validation, and summary JSONL emission.
- **Stage‑2 (Stage‑B) Verdict & Guidance** — `src/stage_b/`, `scripts/stage_b_run.sh`; ingests Stage‑A JSONL + labels, runs rollout/critic/selection/reflection, returns `pass|fail` per ticket.
- **Data Preprocessing & Intake** — `data_conversion/`; optional offline step that normalizes human‑annotated exports (taxonomy, geometry canonicalization, smart resize) into train/val JSONL plus QA artifacts.
- **Training & Fusion** — `src/sft.py`, `src/datasets/`, `configs/`, `scripts/train.sh`; config‑first fine‑tuning and multi‑dataset fusion.
- **Docs & Runbooks** — `docs/` (public), `openspec/` (governance); keep doc ↔ code pointers current.

## Standard Workflow Outline
1. **Data intake** → optional `data_conversion/convert_dataset.sh` to produce train/val/tiny JSONL and validation reports.
2. **Fusion/curation** → prepare `custom.train_jsonl` or `custom.fusion_config` (see `docs/DATA_AND_DATASETS.md`, `docs/UNIFIED_FUSION_DATASET.md`).
3. **Train/finetune** → run `scripts/train.sh --config <yaml>`; update `docs/TRAINING_PLAYBOOK.md` & `docs/REFERENCE.md` if behaviors change.
4. **Stage‑1 inference** → `scripts/stage_a_infer.sh` writes per‑image summaries; verify outputs before Stage‑2.
5. **Stage‑2 verdicts** → `scripts/stage_b_run.sh` emits selections/reflection logs; promote guidance snapshots as needed.
6. **Documentation & governance** → sync `docs/` with changes; open/modify OpenSpec changes only when behavior or contracts shift.

## Codebase Layout
- `src/` — Training/inference code (datasets, config, stage_a, stage_b, utils)
- `configs/` — YAML for training, Stage‑B, fusion, prompts
- `scripts/` — Canonical entrypoints (train, infer, stage_a, stage_b, fusion helpers); see `scripts/README.md`
- `data_conversion/` — Offline preprocessing from annotation platform exports
- `docs/` — Authoritative documentation, pipeline guides, data contracts
- `openspec/` — Change management specs and proposals
- `vis_tools/` — Visualization & QA helpers

## Environment
- Use `ms` conda environment for all Python scripts
- `ms-swift` installed at `/data/ms-swift`
- `transformers` in conda env at `/root/miniconda3/envs/ms/lib/python3.12/site-packages/transformers`
- **Serena MCP**: Available via MCP server; project configured at `.serena/project.yml`. Activate with "activate the project Qwen3-VL" or by path. Project-specific memories stored in `.serena/memories/`.

## Development Approach
- **Configuration-first**: Edit YAML in `configs/` rather than adding ad‑hoc flags
- **Reuse over custom**: Prefer ms‑swift/transformers primitives before adding custom modules
- **Documentation**: Update `docs/` when visible behavior, configs, or workflows change
- **Spec-driven**: For features or major changes, consult `openspec/AGENTS.md` and follow the change process
- **Geometry-aware**: Keep augmentation and data handling geometry‑aware; add tests/visualization when touching `src/datasets/`

## Design Principles (High-Level)
- **Explicit over implicit**: No silent defaults; all config via YAML/CLI/constructor with early validation
- **Type safety**: Strong typing, frozen configs (`@dataclass(frozen=True)`), predictable APIs
- **Clean architecture**: Small public interfaces, dependency injection, compose over inherit, clean import graph (never import upward)
- **Fail fast**: Validate early, clear error messages with remediation hints, no silent failures
- **Extensibility**: Extend via new `Builder`/`Preprocessor`/`Template`, not by editing core logic

## Common Rules & Preferences
- **Docs stay in lockstep with code**: touch the mapped doc when you touch its directory (see `docs/README.md` doc map). Stage‑A/B changes require updates to `docs/REFERENCE.md` and the Stage‑A/B runbooks.
- **Config-first surface**: prefer adding YAML knobs in `configs/` over new CLI flags; validate early via dataclasses in `src/config/schema.py`.
- **Determinism**: seed everything (`seed`, `fusion seed`, `sampler` grids) and log seeds in new entrypoints; avoid implicit randomness.
- **Geometry & grounding**: never drop or re-order geometry silently; use helpers in `src/datasets/geometry.py` and keep `poly` canonicalization rules consistent with `data_conversion/`.
- **Logging**: use `src/utils/logger.get_logger` (rank-aware) and emit remediation hints in errors; avoid print/debug spam.
- **Pass/Fail canonicals**: treat verdicts as `pass|fail` (lowercase) using `GroupLabel`; normalize variants via `normalize_verdict` helpers.
- **Third-party additions**: prefer ms‑swift/transformers primitives; justify new deps and update `UPSTREAM_DEPENDENCIES.md` when behavior changes.
- **Validation before merge**: run existing tests or targeted probes for dataset/geometry changes; add probes when altering preprocessing.

## Important
- **Always interrupt if clarification is needed or anything is vague, ambiguous, or uncertain**
- Run all Python scripts with `ms` conda environment
- For commands and detailed configs, see `docs/README.md` and `docs/REFERENCE.md`

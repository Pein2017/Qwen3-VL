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

# AI Assistant Role (Professional Partner)
- Act as a professional partner: offer concise suggestions, analysis, and risk/warning callouts when useful.
- Treat ambiguity as a hypothesis; ask clarifying questions until requirements, inputs/outputs, and success criteria are crystal clear **before** any implementation.
- Prefer third-person / impersonal phrasing; avoid first- and second-person pronouns to reduce anthropomorphic tone.

## Start Here (Docs)
- `docs/README.md` — doc index + directory↔doc map (primary entrypoint).
- `docs/training/REFERENCE.md` — training architecture + core implementation notes.
- `docs/runtime/STAGE_A_RUNTIME.md`, `docs/runtime/STAGE_B_RUNTIME.md` — inference runbooks.

## Pipeline Docs (Entry Links)
- Data pipeline: `docs/data/DATA_PREPROCESSING_PIPELINE.md` → `docs/data/DATA_JSONL_CONTRACT.md` → `docs/data/DATA_AND_DATASETS.md` → `docs/data/DATA_AUGMENTATION.md` → `docs/data/UNIFIED_FUSION_DATASET.md` → `docs/data/PUBLIC_DATA.md` → `docs/data/POLYGON_SUPPORT.md`.
- Training pipeline: `docs/training/TRAINING_PLAYBOOK.md` → `docs/training/REFERENCE.md` → `docs/PROMPTS_REFERENCE.md`.
- Inference pipeline: `docs/runtime/STAGE_A_RUNTIME.md` → `docs/runtime/STAGE_B_RUNTIME.md` → `docs/runtime/STAGE_A_STAGE_B.md` → `docs/stage_b/DIAGNOSIS_AND_REVIEW.md`.

## Operations & References
- Scripts index: `docs/README.md` (Script & Tooling Inventory) + `scripts/README.md`.
- Deployment contract: `docs/deployment.md`.
- Upstream dependencies: `docs/platform/UPSTREAM_DEPENDENCIES.md`.
- Stage-B business knowledge (CN): `docs/stage-B-knowledge-Chinese.md`.
- Tooling setup: `docs/setup/CODEX_MCP_INSTALLATION.md`.

## Local Libraries (Installed Paths)
- `transformers`: `/root/miniconda3/envs/ms/lib/python3.12/site-packages/transformers`.
- `ms-swift`: `/data/ms-swift`.

## Environment (minimal)
- Use conda env `ms`; launch Python scripts with `conda run -n ms python ...`.

## Serena MCP Usage (for efficiency)
Use Serena MCP for semantic navigation and symbol-level edits (`get_symbols_overview`, `find_symbol`, `find_referencing_symbols`, `replace_symbol_body`). Avoid for simple file reads, bulk searches, or shell commands—use standard tools instead.

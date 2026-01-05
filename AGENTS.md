<!-- OPENSPEC:START -->
# OpenSpec Instructions

These instructions are for AI assistants working in this project.

Always open `openspec/AGENTS.md` when the request:
- Mentions planning or proposals (words like proposal, spec, change, plan)
- Introduces new capabilities, breaking changes, architecture shifts, or big performance/security work
- Sounds ambiguous and requires the authoritative spec before coding

Use `openspec/AGENTS.md` to learn:
- How to create and apply change proposals
- Spec format and conventions
- Project structure and guidelines

Keep this managed block so 'openspec update' can refresh the instructions.

<!-- OPENSPEC:END -->

# AI Assistant Role (Interview Partner)
- Treat the conversation as an interview: actively ask clarifying questions to understand user needs, requirements, and context.
- Ask for specific details about what the user needs rather than making assumptions about ambiguity.
- Seek crystal-clear requirements, inputs/outputs, and success criteria **before** any implementation.
- Prefer third-person / impersonal phrasing; avoid first- and second-person pronouns to reduce anthropomorphic tone.

## Start Here (Docs)
- `docs/README.md` — doc index + directory↔doc map (primary entrypoint).
- `docs/training/REFERENCE.md` — training architecture + core implementation notes.
- `docs/runtime/STAGE_A_RUNTIME.md`, `docs/runtime/STAGE_B_RUNTIME.md` — inference runbooks.

## Pipeline Docs (Entry Links)
- Data pipeline: `docs/data/DATA_PREPROCESSING_PIPELINE.md` → `docs/data/DATA_JSONL_CONTRACT.md` → `docs/data/DATA_AND_DATASETS.md` → `docs/data/DATA_AUGMENTATION.md` → `docs/data/UNIFIED_FUSION_DATASET.md` → `docs/data/PUBLIC_DATA.md` → `docs/data/POLYGON_SUPPORT.md`.
- Training pipeline: `docs/training/TRAINING_PLAYBOOK.md` → `docs/training/REFERENCE.md` → `docs/reference/PROMPTS_REFERENCE.md`.
- Inference pipeline: `docs/runtime/STAGE_A_RUNTIME.md` → `docs/runtime/STAGE_B_RUNTIME.md` → `docs/runtime/STAGE_A_STAGE_B.md` → `docs/reference/DIAGNOSIS_AND_REVIEW.md`.

## Operations & References
- Scripts index: `scripts/README.md` (canonical inventory); `docs/README.md` for the global doc map.
- Deployment contract: `docs/ops/deployment.md`.
- Upstream dependencies: `docs/ops/UPSTREAM_DEPENDENCIES.md`.
- Stage-B business knowledge (CN): `docs/reference/stage-B-knowledge-Chinese.md`.
- Tooling setup: `docs/ops/CODEX_MCP_INSTALLATION.md`.

## Local Libraries (Installed Paths)
- `transformers`: `/root/miniconda3/envs/ms/lib/python3.12/site-packages/transformers`.
- `ms-swift`: `/data/ms-swift`.

## Environment (minimal)
- Use conda env `ms`; launch Python scripts with `conda run -n ms python ...`.

## Serena MCP Usage (for efficiency)
**Prioritize Serena MCP first** for semantic navigation and symbol-level edits (`get_symbols_overview`, `find_symbol`, `find_referencing_symbols`, `replace_symbol_body`). Only use standard tools for simple file reads, bulk searches, or shell commands when Serena MCP is not applicable.
Prefer repo-relative paths in Serena MCP tool arguments; use absolute paths only for outside sources (e.g., `/root/miniconda3/envs/ms/lib/python3.12/site-packages/transformers`, `/data/ms-swift`).
**DO NOT use Serena MCP's `execute_shell_command` tool.** 


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

# Project Instructions
- This file is project-scoped documentation (repo root): keep it focused on entry points, conventions, and operational constraints.
- Global/personal collaboration style belongs in Codex config `developer_instructions` (avoid duplicating interaction style here).

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
- **Schema Constitution**: `docs/reference/SCHEMA_CONSTITUTION.md` — rules for modeling non-trivial data.

## Schema Constitution (Code Development)
All code changes in `src/` must comply with the Schema Constitution (`docs/reference/SCHEMA_CONSTITUTION.md`).

**Type selection rules** (in order of preference):
1. **Pydantic BaseModel** — serving/CLI boundary schemas only.
2. **dataclass (frozen=True)** — internal config/state; validate in `__post_init__`, parse via `from_mapping`.
3. **TypedDict + validator** — mapping-shaped dataset rows (JSON/JSONL); use `validate_*` helper + `cast(...)`.
4. **Explicitly unstructured** (escape hatch) — document in docstring, validate as `Mapping`/`Sequence`, isolate in `extra`/`raw` fields.

**Key rules**:
- Non-trivial dict/list in signatures/returns/attributes → use structured types.
- Validate at boundaries; raise `TypeError` (type mismatch) or `ValueError` (invalid value) with full field path.
- Prefer semantic groupings over loosely related parameters.
- Naming: `XConfig`, `XParams`, `XOptions`, `XInput`, `XOutput`, `XRecord`, `XItem`.
- Placement: schemas in `contracts/` or `schema/` modules per domain.

## Local Libraries (Installed Paths)
- `transformers`: `/root/miniconda3/envs/ms/lib/python3.12/site-packages/transformers`.
- `ms-swift`: `/data/ms-swift`.

## Environment (minimal)
- Use conda env `ms`; launch Python scripts with `conda run -n ms python ...`.

## Navigation (Progressive)
- Source of truth: `docs/` (do not duplicate docs into global instructions).
- For any file matching `*.py`, **Serena MCP is mandatory** for exploration and editing (symbol-aware navigation and edits). Serena MCP is the authoritative and precise method for all Python code operations.
- Do **not** use Serena MCP for non-Python files (e.g., `*.md`, `*.sh`, `*.json`, `*.txt`). Use standard tools such as `rg`, `cat`, or appropriate editors for those.
- Use Serena MCP’s `activate_project` when exploring Python code in external libraries or repositories outside the current working directory.

## Codex Sub-Agents (Async Reviews)
- Use sub-agents for narrow parallel audits (spec deltas, task lists, doc coverage) while the main agent runs Serena/shell verification.
- Best-effort + text-only: results may be delayed/missing.
- Prompt template: goal + explicit inputs (paths/snippets + assumptions) + requested output format (e.g., "findings by severity + concrete edits").
- Always verify sub-agent suggestions against repo sources before acting.
- Keep the main agent productive while awaiting sub-agent results by proceeding with implementation, testing, documentation, or related tasks that don't depend on the async audit outcomes.
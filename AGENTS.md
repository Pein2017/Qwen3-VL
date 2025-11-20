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
Single repository for the Qwen3‑VL training stack with two main flows: model training and inference (Stage‑A per‑image summaries, Stage‑B group‑level reflection). For detailed architecture, workflows, and examples, see `docs/README.md` and `docs/REFERENCE.md`.

## Key Directories
- `src/` — Python source for training and inference
- `configs/` — YAML configs for experiments and inference
- `docs/` — authoritative documentation
- `scripts/` — shell entrypoints wrapping common workflows
- `vis_tools/` — visualization and debugging helpers

## Environment
- Use `ms` conda environment for all Python scripts
- `ms-swift` installed at `/data/ms-swift`
- `transformers` in conda env at `/root/miniconda3/envs/ms/lib/python3.12/site-packages/transformers`

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

## Important
- **Always interrupt if clarification is needed or anything is vague, ambiguous, or uncertain**
- Run all Python scripts with `ms` conda environment
- For commands and detailed configs, see `docs/README.md` and `docs/REFERENCE.md`
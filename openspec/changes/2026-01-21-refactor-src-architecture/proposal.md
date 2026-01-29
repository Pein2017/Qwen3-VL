# Proposal: Refactor `src/` architecture for modularity, decoupling, and scalable maintenance

## Why

The `src/` codebase has grown into a set of highly capable pipelines (SFT, Stage-A, Stage-B, augmentation, generation) but several structural traits make it increasingly expensive to maintain:

- **Monolithic orchestrators**: `src/sft.py`, `src/stage_a/inference.py`, and `src/stage_b/runner.py` bundle config parsing, IO, business rules, orchestration, and runtime concerns into single modules. This amplifies merge conflicts and makes changes risky.
- **Import-time side effects**: Top-level package imports pull in heavyweight runtime dependencies (torch/swift/transformers) and perform registration at import time, which slows tooling and encourages circular-import workarounds.
- **Duplicated parsing and schema glue**: Multiple copies of bool/int/float coercion and summary-JSON formatting/parsing exist across Stage-A and Stage-B, creating drift risk and inconsistent error semantics.
- **Layering leaks**: Training config modules contain runtime-domain knowledge and prompt registries; datasets import config prompts; private helper functions are imported across modules.

The refactor goal is to make the `src/` codebase easier to extend (new pipelines, new backends, new datasets), safer to change, and cheaper to test — without changing user-facing behavior.

## What Changes

Introduce an explicit, testable internal architecture for `src/` with the following outcomes:

1) **Cheap imports + no hidden side effects**
- `import src` and key subpackages SHALL be lightweight (no heavyweight model/training imports unless explicitly requested).
- Import-time global registration MUST be eliminated or deferred behind explicit entrypoints.

2) **Centralized shared utilities (single source of truth)**
- Consolidate duplicated coercion/validation helpers (bool/int/float parsing) into a single module.
- Consolidate Stage-A/Stage-B summary JSON extraction/formatting/header-stripping into a shared utility.

3) **Schema Constitution compliance at boundaries**
- Add explicit validators for mapping-shaped boundary payloads (TypedDict + validator).
- Replace “many-parameter” call surfaces with semantic grouping `XOptions` / `XParams` dataclasses where the inputs are interdependent.

4) **Decompose monoliths into composable subsystems**
- Keep public entrypoints stable (`python -m src.sft`, `python -m src.stage_a.cli`, `python -m src.stage_b.runner`, and Stage-B `run_all`).
- Move orchestration logic into dedicated modules/classes (e.g., `training/app.py`, `stage_a/pipeline.py`, `stage_b/pipeline.py`) so CLI modules become thin wrappers.

5) **Reduce cross-module coupling**
- Remove cross-module imports of underscore-prefixed helpers by promoting them into explicit public utility modules.
- Clarify dependency directions to reduce circular-import pressure.

## Scope

- **In scope**: `src/` module layout, imports, internal APIs, shared utility extraction, boundary validation additions, and internal refactors required to satisfy the new architecture requirements.
- **Out of scope**:
  - Any intentional change to training/inference semantics.
  - Data contract changes (JSONL schemas) beyond validation improvements.
  - New model capabilities or new reward definitions.

## Non-goals

- Rewriting augmentation/geometry math (only structural moves and API stabilization).
- Changing CLI flags, config keys, or output artifact formats (compatibility preserved).
- Introducing new external dependencies.

## Risks

- Refactors can introduce subtle behavioral changes (ordering, seeding, default prompts, IO paths) even when not intended.
- Import changes can reveal hidden dependencies (code that relied on side effects).
- Large file moves can increase short-term churn; careful staging is required.

## Validation plan

- `openspec validate 2026-01-21-refactor-src-architecture --strict`
- Static checks: `ruff` + pyright/basedpyright (via conda env `ms`).
- Smoke checks (no training required):
  - `python -c "import src"` must be fast and must not import heavy deps by default.
  - `python -m src.sft --help` and Stage-A/Stage-B CLIs must still work.
- Targeted unit tests for:
  - shared parsing utilities
  - shared summary-json utilities
  - Stage-A record validator

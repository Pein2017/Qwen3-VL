# Design: Refactor `src/` architecture

## Design goals

- **Decoupling**: separate IO/config/orchestration from domain logic.
- **Scalability**: enable adding new pipeline components (datasets, prompts, engines) without editing monoliths.
- **Testability**: make core logic importable and unit-testable without requiring GPU/runtime setup.
- **Schema Constitution alignment**: structured types at boundaries and in non-trivial signatures.
- **Backward compatibility where practical**: preserve core entrypoints and stable imports.

## Target module boundaries (conceptual)

This refactor does not require a single large package rename; instead it establishes clear layering:

1) **Foundation**
- `src/utils/` (logger, distributed, unstructured validators)

2) **Shared contracts + parsing**
- `src/utils/parsing.py`: coercion helpers (`coerce_bool/int/float`, etc.) with consistent error types/paths.
- `src/utils/summary_json.py`: summary JSON parsing/formatting/header-stripping used by both Stage-A and Stage-B.

3) **Domain pipelines**
- `src/training/` (new): training orchestration extracted from `src/sft.py`.
- `src/stage_a/` (refactor): split `inference.py` into discovery + image IO + batching/inference + output writing + sanitization.
- `src/stage_b/` (refactor): split `runner.py` into CLI + pipeline orchestrator + services (rollouts/proposer/gating) + IO codecs.

4) **Existing stable subsystems**
- `src/generation/` remains the centralized engine; call sites should use contracts (`GenerationOptions`, etc.) rather than bespoke parsing.
- `src/datasets/` remains the training dataset subsystem; reduce coupling to config/prompt modules and remove cross-module private helper imports.

## Compatibility strategy

- Preserve CLI entrypoints:
  - `python -m src.sft --config ...`
  - `python -m src.stage_a.cli ...`
  - `python -m src.stage_b.runner ...`
- Preserve Stage-B public API: `src.stage_b.run_all` remains available (via re-export or lazy `__getattr__`).
- Use *forwarder modules* where needed (thin wrappers that import and delegate to the new implementation).

## Dependency rules (informal)

- `src/utils/*` MUST NOT import training/runtime pipelines.
- `src/generation/*` MUST NOT import Stage-A/B or training code.
- Stage-A/B MAY depend on generation + prompts + utils.
- Training MAY depend on datasets + generation + utils.
- Avoid importing from modules with underscore-prefixed names across package boundaries. Promote shared helpers into explicit public modules.

## Key refactor moves

### 1) Lazy imports and reduced side effects
- Convert heavyweight `__init__.py` aggregators to lazy import patterns (`__getattr__`) so `import src` is cheap.
- Move reward registration and other global setup from import-time to runtime entrypoints.

### 2) Shared parsing and summary-json utilities
- Replace duplicated bool parsing in:
  - `src/config/loader.py`
  - `src/config/schema.py`
  - `src/stage_b/config.py`
  - `src/datasets/wrappers/__init__.py`
  with a single canonical helper.

- Replace duplicated summary JSON formatting/parsing in:
  - `src/stage_a/inference.py`
  - `src/stage_a/postprocess.py`
  - `src/stage_b/ingest/stage_a.py`
  - `src/stage_b/sampling/prompts.py`
  with `src/utils/summary_json.py`.

### 3) Boundary validation for Stage-A JSONL
- Add a `validate_stage_a_group_record(...)` adjacent to `StageAGroupRecord` and apply it before writing or postprocessing.

### 4) Semantic grouping for configuration-heavy constructors
- Introduce `XOptions` dataclasses for dataset construction and pipeline entrypoints.
- Update factories/orchestrators to pass these objects, reducing long parameter lists.

## Rollout strategy

This refactor should be staged:

- Stage 1: add shared utilities + validators; switch call sites.
- Stage 2: lazy-import cleanup and import-time side-effect removal.
- Stage 3: split monoliths into internal modules while keeping public entrypoints stable.
- Stage 4: remove remaining private cross-module imports and align docs/tests.


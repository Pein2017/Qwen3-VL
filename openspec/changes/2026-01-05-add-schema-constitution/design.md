# Design: Schema Constitution Refactor (src/)

## Intent
Refactor `src/` to comply with the Schema Constitution by replacing non-trivial dict/list usage with structured types that align with existing patterns.

## Decisions
- **Default patterns**:
  - Boundary dataset rows: `TypedDict` + validator (mapping-shaped JSON/JSONL).
  - Internal config/state: frozen `dataclass` with `__post_init__` validation and `from_mapping` parsing.
- **Pydantic scope**: permitted only for serving/CLI boundary schemas; not used for internal `src/` refactors unless the module already depends on Pydantic.
- **Backward compatibility**: none. Refactor may break internal call sites; compliance is prioritized over compatibility.

## Refactor Rules (src/)
- Replace non-trivial dict/list usage in function signatures, return types, and class attributes with structured types.
- Keep trivial local dict/list usage where no hard trigger applies (local expressions, 1–2 semantic fields, no boundary crossing).
- Unstructured payloads must be isolated as `extra`/`raw` fields and documented.

## Risk Controls
- Incremental inventory -> refactor -> lint loop.
- Maintain explicit validators to preserve error-path clarity (TypeError vs ValueError).

## Acceptance
- All `src/` modules follow the constitution rubric.
- Ruff + pyright pass under `scripts/run_lint_loop.sh`.

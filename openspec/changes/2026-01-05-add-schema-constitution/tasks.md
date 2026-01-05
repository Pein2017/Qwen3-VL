# Tasks: Schema Constitution and `src/` Refactor

- [x] Confirm the canonical location and filename for the Schema Constitution (`docs/reference/SCHEMA_CONSTITUTION.md`).
- [x] Draft the Schema Constitution with: minimal non-trivial rubric (rubric C), type selection rules, validation/error guidance, and escape-hatch rules for unstructured mappings.
- [x] Add generic before/after examples that illustrate TypedDict + validator and frozen dataclass + `from_mapping`.
- [x] Inventory non-trivial dict/list usage under `src/` (function signatures, returns, attributes) and map to target structured types or explicit unstructured exceptions.
- [x] Refactor `src/` to replace non-trivial dict/list usage with dataclasses, TypedDict + validators, or documented+validated unstructured payloads per the constitution.
- [x] Apply required updates outside `src/` if refactors change test or script expectations.
- [x] Run the python-lint-loop (ruff + pyright) until clean after Python edits.

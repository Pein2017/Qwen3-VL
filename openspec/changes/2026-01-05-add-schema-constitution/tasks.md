# Tasks: Schema Constitution and Checklist Refactor

- [ ] Choose the canonical location and filename for the Schema Constitution under `docs/reference/`.
- [ ] Draft the Schema Constitution with: non-trivial rubric (hard/soft triggers), type selection rules, validation/error guidance, and migration examples.
- [ ] Add before/after examples that align with `src/datasets/contracts.py` and `src/config/schema.py`.
- [ ] Define a single schema review checklist (either in the constitution or a linked reference file).
- [ ] Refactor existing checklists to reference the canonical schema checklist instead of duplicating items.
- [ ] Update `docs/README.md` and section indices to include the new constitution and checklist references.
- [ ] Verify internal links remain valid after the update.
- [ ] Inventory non-trivial dict/list usage under `src/` (function signatures, returns, attributes) and map to target structured types.
- [ ] Refactor `src/` to replace non-trivial dict/list usage with dataclasses, TypedDict + validators, or Pydantic models per the constitution.
- [ ] Run the python-lint-loop (ruff + pyright) until clean after Python edits.

# Proposal: Add Schema Constitution and Refactor Checklists

## Summary
Establish a Schema Constitution document that standardizes how non-trivial data is modeled (dataclass, TypedDict, Pydantic), refactor schema-related checklists to a single canonical source, and update `src/` to comply with the constitution.

## Motivation
- Non-trivial data structures appear across datasets, configs, and runtime boundaries, but current guidance is distributed and implicit.
- A single, explicit constitution improves review consistency, reduces drift in data contracts, and makes schema choices auditable.
- Checklist refactoring prevents duplication and keeps schema compliance actionable in code reviews.

## Scope
- Add a dedicated Schema Constitution document under `docs/reference/` with clear rules, decision trees, and examples.
- Define a concise, trackable rubric for what counts as non-trivial data.
- Provide before/after examples that align with existing patterns in `src/datasets/contracts.py` and `src/config/schema.py`.
- Centralize schema review checklist items in one canonical location and update references.
- Update documentation indices to include the new constitution and checklist references.
- Scan all Python modules under `src/` and refactor non-trivial dict/list usage to structured types that comply with the constitution.

## Non-Goals
- No refactoring outside `src/`.
- No enforcement tooling changes (lint/CI) beyond documentation guidance; compliance is enforced via code review and the canonical checklist.
- No new runtime behavior or configuration changes.

## Risks and Mitigations
- **Risk:** Overly strict rules could slow development.
  - **Mitigation:** Use a minimal non-trivial rubric with hard/soft triggers and explicit escape hatches.
- **Risk:** Checklist duplication persists across docs.
  - **Mitigation:** Require a single canonical checklist and replace duplicates with references.
- **Risk:** Refactor changes may break existing schemas or data paths.
  - **Mitigation:** No backward-compatibility guarantees are provided; refactor is allowed to be breaking within `src/` and will be validated by tests and linting.

## Success Criteria
- A Schema Constitution document exists, is indexed, and provides decision rules, examples, and validation guidance.
- A single schema review checklist exists and is referenced by other checklists where schema compliance is required.
- Documentation indices are updated and all links remain valid.
- `src/` modules comply with the constitution (non-trivial data is modeled using structured types; raw dict/list usage is isolated to trivial or explicitly unstructured cases).
- Pydantic usage is limited to serving/CLI boundary schemas; dataclass + `from_mapping` and TypedDict + validators remain the default internal/boundary patterns elsewhere.

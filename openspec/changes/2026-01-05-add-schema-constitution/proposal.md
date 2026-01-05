# Proposal: Add Schema Constitution and Align `src/` to the Constitution

## Summary
Establish a Schema Constitution document that standardizes how non-trivial data is modeled (dataclass, TypedDict, limited Pydantic) and refactor `src/` to comply with the constitution.

## Motivation
- Non-trivial data structures appear across datasets, configs, and runtime boundaries, but current guidance is distributed and implicit.
- A single, explicit constitution improves review consistency, reduces drift in data contracts, and makes schema choices auditable.
- Clear rules for structured vs unstructured data reduce ad-hoc `dict`/`list` usage.

## Scope
- Add a dedicated Schema Constitution document under `docs/reference/SCHEMA_CONSTITUTION.md` with clear rules, decision guidance, and examples.
- Define a minimal non-trivial rubric (rubric C) and an explicit escape-hatch rule for unstructured mappings.
- Provide generic before/after examples (no codebase references).
- Scan all Python modules under `src/` and refactor non-trivial dict/list usage in signatures/returns/class attributes to structured types, or document and validate explicitly unstructured payloads.
- Apply targeted updates outside `src/` only when required by refactors (tests/scripts that consume refactored types).

## Non-Goals
- No enforcement tooling changes (lint/CI) beyond documentation guidance.
- No new runtime behavior or configuration changes beyond refactors needed for compliance.

## Risks and Mitigations
- **Risk:** Overly strict rules could slow development.
  - **Mitigation:** Use a minimal rubric and explicit escape-hatch for intentionally unstructured payloads.
- **Risk:** Refactor changes may break existing schemas or data paths.
  - **Mitigation:** No backward-compatibility guarantees; validate via tests and linting.

## Success Criteria
- A Schema Constitution document exists and provides the minimal rubric, type-selection rules, validation/error handling guidance, and escape-hatch rules.
- `src/` modules comply with the constitution; non-trivial structures use structured types, and explicitly unstructured payloads are documented and validated.
- Pydantic usage is limited to serving/CLI boundary schemas unless a module already depends on it.

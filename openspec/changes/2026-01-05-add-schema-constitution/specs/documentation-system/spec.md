## ADDED Requirements
### Requirement: Schema Constitution Document
A Schema Constitution SHALL define the project-wide rules for modeling non-trivial data, including the boundary/internal decision tree and type-selection rules aligned with current patterns.

#### Scenario: Constitution is discoverable and complete
- **WHEN** schema guidance is published or updated
- **THEN** a Schema Constitution document exists under `docs/reference/SCHEMA_CONSTITUTION.md`
- **AND** it is indexed in `docs/README.md` and `docs/reference/README.md`
- **AND** it includes a concise non-trivial rubric (hard/soft triggers), a boundary/internal decision tree, validation/error handling rules, and before/after examples aligned to `src/datasets/contracts.py` and `src/config/schema.py`
- **AND** it explicitly states Pydantic is permitted only for serving/CLI boundary schemas unless a module already depends on it
- **AND** it explicitly states there are no backward-compatibility guarantees for the `src/` refactor

### Requirement: Canonical Schema Review Checklist
A single schema review checklist SHALL be the canonical source of schema compliance checks, and other checklists MUST reference it rather than duplicate items.

#### Scenario: Checklist references are centralized
- **WHEN** a review checklist or runbook includes schema compliance checks
- **THEN** it links to the canonical schema checklist instead of restating the items
- **AND** the canonical checklist is linked from the Schema Constitution document
- **AND** the canonical checklist covers boundary vs internal schema selection, structured type usage, validation expectations, and allowed escape hatches (for example, `extra` mappings)

### Requirement: `src/` Refactor Compliance
All Python modules under `src/` SHALL comply with the Schema Constitution by replacing non-trivial dict/list usage with structured types.

#### Scenario: `src/` is brought into compliance
- **WHEN** the refactor is completed
- **THEN** function signatures, return types, and class attributes use structured types for non-trivial data
- **AND** trivial local dict/list usage is retained only when no hard trigger applies
- **AND** unstructured payloads are isolated in `extra`/`raw` fields when required

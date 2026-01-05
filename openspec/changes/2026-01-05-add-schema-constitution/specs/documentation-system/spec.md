## ADDED Requirements
### Requirement: Schema Constitution Document
A Schema Constitution SHALL define the project-wide rules for modeling non-trivial data, including the boundary/internal decision rules, type-selection rules, validation/error handling, and escape-hatch guidance for explicitly unstructured mappings.

#### Scenario: Constitution is discoverable and complete
- **WHEN** schema guidance is published or updated
- **THEN** a Schema Constitution document exists under `docs/reference/SCHEMA_CONSTITUTION.md`
- **AND** it is written as pure guidance (no execution, monitoring, or checklist coupling)
- **AND** it includes a minimal non-trivial rubric (rubric C) that applies to function signatures, return types, and class attributes
- **AND** it includes type-selection rules aligned to dataclass, TypedDict + validator, and limited Pydantic usage
- **AND** it includes validation/error handling guidance (TypeError vs ValueError with full field paths)
- **AND** it includes escape-hatch rules permitting explicitly unstructured mappings only when documented and validated as mappings
- **AND** it explicitly states Pydantic is permitted only for serving/CLI boundary schemas unless a module already depends on it
- **AND** it explicitly states there are no backward-compatibility guarantees for the `src/` refactor

### Requirement: `src/` Refactor Compliance
All Python modules under `src/` SHALL comply with the Schema Constitution by replacing non-trivial dict/list usage with structured types or documented+validated explicitly unstructured payloads.

#### Scenario: `src/` is brought into compliance
- **WHEN** the refactor is completed
- **THEN** function signatures, return types, and class attributes use structured types for non-trivial data
- **AND** trivial dict/list usage is retained only for simple lookups, flat lists of primitives, or local expressions
- **AND** explicitly unstructured payloads are documented and validated as mappings or sequences

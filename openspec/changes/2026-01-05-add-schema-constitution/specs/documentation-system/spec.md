## ADDED Requirements
### Requirement: Schema Constitution Document
A Schema Constitution SHALL define the project-wide rules for modeling non-trivial data and MUST be maintained as a first-class reference document.

#### Scenario: Constitution is discoverable and complete
- **WHEN** schema guidance is published or updated
- **THEN** a Schema Constitution document exists under `docs/reference/SCHEMA_CONSTITUTION.md`
- **AND** it is indexed in `docs/README.md` and `docs/reference/README.md`
- **AND** it includes a concise non-trivial rubric (hard/soft triggers), a type selection decision tree, validation/error handling rules, and before/after examples aligned to `src/datasets/contracts.py` and `src/config/schema.py`

### Requirement: Canonical Schema Review Checklist
A single schema review checklist SHALL be the canonical source of schema compliance checks, and other checklists MUST reference it rather than duplicate items.

#### Scenario: Checklist references are centralized
- **WHEN** a review checklist or runbook includes schema compliance checks
- **THEN** it links to the canonical schema checklist instead of restating the items
- **AND** the canonical checklist is linked from the Schema Constitution document
- **AND** the canonical checklist covers boundary vs internal schema selection, structured type usage, validation expectations, and allowed escape hatches (for example, `extra` mappings)

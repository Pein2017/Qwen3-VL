# auto-context-router Specification

## Purpose
Define the auto-context router behavior for loading relevant docs/procedures based on request cues, explicit paths, and safe manual overrides.
## Requirements
### Requirement: Complex-feature detection and immediate loading
The auto-context router SHALL detect complex feature work and immediately load relevant docs and procedures.

#### Scenario: Architecture or refactor request
- WHEN a request includes complex-feature cues (refactor, architecture, algorithm, performance, security, proposal/spec/design, or cross-module work)
- THEN the router SHALL immediately load the mapped docs and procedures using `context_map.yaml` and `docs/README.md`.

### Requirement: Path-based routing
The router SHALL use explicit path mentions to select docs.

#### Scenario: Directory mention
- WHEN a request references a project directory (e.g., `src/`, `src/stage_b/`, `configs/`, `scripts/`)
- THEN the router SHALL load the docs mapped to that directory in `docs/README.md`.

### Requirement: Business and math knowledge loading
The router SHALL load business and mathematical knowledge sources when tasks require them.

#### Scenario: Stage-B guidance or BBU/RRU business logic
- WHEN a request references Stage-B guidance, rule_search, or BBU/RRU business logic
- THEN the router SHALL load `docs/reference/stage-B-knowledge-Chinese.md` and `docs/data/BBU_RRU_BUSINESS_KNOWLEDGE.md` along with relevant runtime docs.

### Requirement: Manual override and fail-safe
The router SHALL preserve manual control and default to manual selection on uncertainty.

#### Scenario: Manual override requested
- WHEN a user requests manual loading or says "do not auto-load"
- THEN the router SHALL skip auto-loading and ask for the desired doc list.

#### Scenario: No match found
- WHEN no routing match is found
- THEN the router SHALL load only `docs/README.md` and ask a single clarifying question.

### Requirement: Maintainable routing map
The router SHALL keep routing logic maintainable via a single explicit map.

#### Scenario: New pipeline added
- WHEN a new pipeline or directory is added to the codebase
- THEN maintainers SHALL update `docs/README.md` and `context_map.yaml` with the new directory->doc mapping.

### Requirement: Language and compatibility
The router SHALL keep Stage-B audit guidance in Chinese and avoid runtime changes.

#### Scenario: Stage-B audit guidance
- WHEN Stage-B guidance audit is required
- THEN the router SHALL keep `stageb-guidance-audit` content in Chinese and load Chinese business knowledge docs.


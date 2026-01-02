## Capability
Stage-B reflection operates without LLM parsing

## ADDED Requirements
### Requirement: Configurable reflection engine
Reflection MUST allow an `engine` setting with values `llm` or `deterministic`; configs MAY choose deterministic without changing other stages.
#### Scenario: Engine flag respected
- Given Stage-B reflection config sets `engine: deterministic`
- When reflection runs
- Then no LLM prompt/generation/parsing SHALL be invoked for reflection, and the deterministic path SHALL execute.

### Requirement: Deterministic reflection updates guidance
When reflection is deterministic, contradictions/all-wrong/conflict batches MUST produce guidance updates without LLM output parsing.
#### Scenario: Contradiction bundle
- Given a bundle with mixed label_match or verdict conflicts
- When deterministic reflection runs
- Then it MUST generate at least one `upsert` operation and apply it via GuidanceRepository, incrementing guidance step if applied.

### Requirement: LLM path remains available
The existing LLM reflection path MUST remain usable when `engine: llm` so prior behavior is preserved if needed.
#### Scenario: Engine set to llm
- Given reflection.engine is `llm`
- When reflection runs
- Then the previous LLM-based reflection MUST execute unchanged.

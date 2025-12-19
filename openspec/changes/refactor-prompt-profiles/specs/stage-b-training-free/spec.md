# stage-b-training-free Specification

## ADDED Requirements

### Requirement: Stage‑B inference SHALL inject domain knowledge for BBU vs RRU as a read‑only block.
Stage‑B MUST include a domain knowledge block in the system prompt for inference. The block SHALL be derived from the domain pack dataclasses (BBU/RRU), MUST be treated as read‑only scaffolding, and MUST NOT be modified by reflection or written back into guidance files.

#### Scenario: Domain block is appended to system prompt
- **GIVEN** Stage‑B runs with domain resolved to `bbu`
- **WHEN** the system prompt is built for rollout
- **THEN** the prompt contains the BBU domain knowledge block
- **AND THEN** the block is not part of `guidance.experiences` and is excluded from reflection updates

---

### Requirement: Stage‑B SHALL resolve domain deterministically with validation.
Stage‑B MUST resolve the domain using config only, with deterministic precedence:
1) `config.domain_map[mission]` (if configured)
2) `config.default_domain` fallback
Unknown or missing domains MUST raise a validation error before prompt construction.

#### Scenario: Domain resolved from config mapping
- **GIVEN** a config mapping for the ticket’s mission
- **WHEN** Stage‑B resolves the domain
- **THEN** it selects the mapped domain and proceeds with the corresponding pack

#### Scenario: Missing domain fails fast
- **GIVEN** no matching mapping and no default domain configured
- **WHEN** Stage‑B resolves the domain
- **THEN** it fails with a clear error stating the required config sources

# sft-training Specification

## ADDED Requirements

### Requirement: Summary prompt profiles SHALL separate training and inference roles.
When `custom.use_summary` is enabled, the system SHALL resolve a summary prompt profile. The default profile for training SHALL be `summary_train_min`, and it MUST include **only** format rules and task criterion (including evidence‑only / no‑hallucination constraints); it MUST NOT include domain knowledge, mission rules, or dataset-specific priors.

#### Scenario: Default summary training uses minimal profile
- **GIVEN** a training config with `custom.use_summary: true` and no prompt profile override
- **WHEN** prompts are resolved for summary training
- **THEN** the system selects `summary_train_min`
- **AND THEN** the resulting system prompt contains format + task criterion only (including evidence‑only / no‑hallucination constraints), with no BBU/RRU domain rules or mission-specific priors

#### Scenario: Runtime profile is explicit and opt-in
- **GIVEN** a config that sets `prompts.profile: summary_runtime`
- **WHEN** prompts are resolved
- **THEN** the system uses the runtime profile and allows domain knowledge injection (if a domain is provided)

---

### Requirement: Domain knowledge packs SHALL be defined as Python dataclasses and excluded from training prompts.
Domain knowledge (BBU/RRU schema hints, priors, and restrictions) SHALL be defined in Python dataclasses and composed into prompts only when the runtime profile is selected. Training profiles MUST ignore domain packs entirely.

#### Scenario: Training profile ignores domain packs
- **GIVEN** domain packs defined for BBU and RRU
- **WHEN** a training run resolves `summary_train_min`
- **THEN** the system prompt excludes domain pack content even if a domain is configured

#### Scenario: Runtime profile includes domain pack
- **GIVEN** `prompts.profile: summary_runtime` and `prompts.domain: rru`
- **WHEN** the summary system prompt is built
- **THEN** the prompt includes the RRU domain pack content and excludes BBU-only rules

---

### Requirement: Prompt profile selection SHALL be configurable and validated.
Prompt profile and domain selection SHALL be configured via the `prompts` section. `prompts.system` or `prompts.user` overrides MUST remain authoritative. If a runtime profile is selected and the domain is missing or unknown, the system MUST fail fast with an actionable error.

#### Scenario: Unknown domain is rejected
- **GIVEN** `prompts.profile: summary_runtime` and `prompts.domain: unknown`
- **WHEN** the loader resolves prompts
- **THEN** it raises a validation error describing the allowed domains

#### Scenario: Explicit prompt override bypasses profiles
- **GIVEN** `prompts.system` or `prompts.user` is set in the config
- **WHEN** prompts are resolved
- **THEN** profile composition is bypassed and the provided override is used verbatim

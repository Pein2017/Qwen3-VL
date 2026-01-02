## MODIFIED Requirements

### Requirement: Stage-B SHALL generate group verdict candidates via prompt-only rollouts under a strict two-line binary contract.
Stage-B rollout prompts SHALL treat guidance rules as a conjunctive (AND) checklist by default. OR semantics are only allowed when a rule text explicitly states an OR/exception condition. Reasons MUST cover the required checks or state which check is missing and therefore fails.

#### Scenario: AND semantics for guidance rules
- **WHEN** guidance contains multiple scaffold (S*) and learnable (G*) rules,
- **THEN** the rollout decision logic SHALL treat them as an AND checklist by default,
- **AND** OR semantics SHALL only be used when a rule text explicitly states an OR/exception condition.

#### Scenario: Evidence coverage in Reason
- **WHEN** a verdict is produced,
- **THEN** the Reason line SHALL cover evidence for each required check or state which check is missing,
- **AND** missing evidence SHALL result in a fail verdict.

## ADDED Requirements

### Requirement: Stage-B SHALL drop overlength prompts instead of truncating.
Stage-B SHALL enforce prompt budgets for rollout, rule-search proposer, and reflection by dropping overlength prompts rather than truncating, to preserve sample completeness.

#### Scenario: Rollout drop on overlength prompts
- **WHEN** a rollout prompt exceeds its configured max prompt tokens,
- **THEN** the ticket SHALL be dropped for that rollout attempt,
- **AND** the system SHALL NOT truncate or partially generate the prompt.

#### Scenario: Proposer drop on overlength prompts
- **WHEN** the rule-search proposer prompt exceeds its configured max prompt tokens after trimming examples,
- **THEN** the proposer SHALL skip generation for that iteration,
- **AND** the system SHALL NOT truncate the proposer prompt.

#### Scenario: Reflection drop on overlength prompts
- **WHEN** a reflection prompt exceeds its configured max prompt tokens after packing,
- **THEN** the reflection pass SHALL skip generation for that cycle,
- **AND** the system SHALL NOT truncate the reflection prompt.

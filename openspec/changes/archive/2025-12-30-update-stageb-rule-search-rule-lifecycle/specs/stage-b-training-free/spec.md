# stage-b-training-free Specification (Delta)

## ADDED Requirements

### Requirement: Stage‑B rule_search SHALL support lifecycle operations for guidance rules (upsert/update/merge/remove).
The rule_search proposer MUST be able to request rule lifecycle operations, including:
- **upsert**: add a new rule.
- **update**: replace an existing rule by signature match.
- **merge**: merge multiple existing rules into a single rule.
- **remove**: delete an existing rule.

The proposer response MUST support a structured `operations` array. Each item MUST include:
- `op`: one of `upsert|update|merge|remove`.
- `text`: required for `upsert|update|merge`; omitted for `remove`.
- `target_signature`: required for `update|remove`.
- `target_signatures`: required for `merge`, array length >= 2.

Legacy `rules` list (upsert-only) is still supported when `operations` is absent. `rules` and `operations` MUST NOT both be present in the same response.

The system MUST validate operations and MUST NOT modify scaffold rules (`S*`). `G0` MUST NOT be deleted.

#### Scenario: Rule_search proposes an update operation
- **WHEN** rule_search proposer submits an operation `{op:"update", target_signature:"...", text:"..."}`.
- **THEN** Stage‑B MUST resolve the target signature to an existing mutable rule.
- **THEN** Stage‑B MUST preview the updated guidance before gating.

#### Scenario: Rule_search proposes a merge operation
- **WHEN** rule_search proposer submits an operation `{op:"merge", target_signatures:["...","..."], text:"..."}`.
- **THEN** Stage‑B MUST validate that all target signatures exist and are mutable.
- **THEN** Stage‑B MUST preview the merged guidance before gating.

#### Scenario: Rule_search proposes a remove operation
- **WHEN** rule_search proposer submits an operation `{op:"remove", target_signature:"..."}`.
- **THEN** Stage‑B MUST validate the target signature exists and is mutable.
- **THEN** Stage‑B MUST preview the guidance with the target rule removed before gating.

### Requirement: Stage‑B rule_search SHALL detect harmful rules via ablation and apply corrective operations subject to fp‑prioritized gates.
Stage‑B MUST be able to evaluate negative impact by comparing baseline guidance to an ablated guidance set (removing a target rule). An update/merge/remove operation MUST pass fp‑prioritized gates:
- **fp_rate improvement**: fp_rate MUST improve (delta < 0).
- **acc improvement**: acc MUST improve (delta > 0).
- **max_fp_rate_increase**: fp_rate MUST NOT increase above `max_fp_rate_increase` (default 0.01).
- **max_changed_fraction**: prediction churn MUST be <= `gate.max_changed_fraction` (default 0.05).

#### Scenario: Ablation removes a harmful rule
- **WHEN** ablation improves both fp_rate and acc without violating fp/acc/churn gates.
- **THEN** Stage‑B MUST accept a remove/update operation and record it in rule_search artifacts.

## MODIFIED Requirements

### Requirement: Stage‑B SHALL select one final verdict per ticket and export deterministic artifacts.
Rule_search artifacts MUST include operation metadata (op type and target signatures) for accepted update/merge/remove actions.

#### Scenario: Accepted rule_search operation is recorded
- **WHEN** a lifecycle op passes gate and is applied.
- **THEN** `benchmarks.jsonl` MUST record operation type and target signature(s).

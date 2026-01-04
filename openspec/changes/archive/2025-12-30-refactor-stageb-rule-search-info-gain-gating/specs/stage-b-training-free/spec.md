# stage-b-training-free (delta)

## MODIFIED Requirements

### Requirement: Stage‑B SHALL perform two-pass reflection to update mission guidance from gradient candidates.
Stage‑B MUST support two execution modes selected by config:
- `legacy_reflection`: the existing two-pass decision/ops reflection flow (including stop-gradient queue routing).
- `rule_search`: a rule-search (tree-growth) flow where reflection is only a lightweight **rule proposer**, and **all guidance changes are gated by large-scale rollout metrics**.

In `legacy_reflection` mode, Stage‑B MUST preserve the existing semantics:
1) **Decision pass**: classify stop-gradient tickets after seeing `gt_label`.
2) **Ops pass**: propose strict JSON guidance operations using only learnable cases.

In `rule_search` mode, Stage‑B MUST:
- Select `reflect_size` high-value mismatch tickets from a baseline validation rollout.
- Run a reflection proposer that outputs `K` candidate binary rules under a strict JSON schema.
- Validate each candidate rule via A/B rollouts on a large validation set.
- Admit at most one rule per iteration, **only if it passes the gate** (see below).
- Enforce “only gate writes guidance”: reflection outputs MUST NOT be applied directly.

#### Scenario: rule_search proposes rules from a small mismatch set and validates on a large set
- **WHEN** Stage‑B runs with `mode=rule_search`.
- **THEN** reflection MUST only consume `reflect_size` mismatch tickets (default 16) and output up to `K` candidate rules (default 3).
- **AND THEN** Stage‑B MUST evaluate each candidate rule on a large validation set using rollout-only inference (no extra LLM passes beyond proposing rules).
- **AND THEN** Stage‑B MUST accept a rule only if it passes the gate (relative error reduction + bootstrap + changed_fraction).

### Requirement: Stage‑B SHALL route stop-gradient tickets only after decision pass.
In `legacy_reflection` mode, Stage‑B MUST treat the stop-gradient quarantine queue (`*_queue.jsonl` / aggregated JSON) as the only stop-gradient sink.

In `rule_search` mode, Stage‑B MUST NOT run a stop-gradient decision pass and MUST NOT emit quarantine artifacts as part of the learning loop.

#### Scenario: rule_search mode does not write quarantine artifacts
- **WHEN** Stage‑B runs with `mode=rule_search`.
- **THEN** Stage‑B MUST NOT create or update stop-gradient queue artifacts as part of rule learning.

## ADDED Requirements

### Requirement: Stage‑B SHALL gate new guidance rules by relative error reduction with bootstrap verification.
In `rule_search` mode, Stage‑B MUST gate candidate rule admission using rollout-measured truth-consistency:
- Let `acc` be majority-vote accuracy on a validation set; let `err = 1 - acc`.
- Let `RER = (err_base - err_new) / max(err_base, eps)` be relative error reduction.
- The system MUST accept a candidate rule only if:
  - `RER >= 0.1` on validation, AND
  - `changed_fraction >= 0.01` on validation, AND
  - bootstrap verification passes (ticket-level resampling with configurable probability threshold).

#### Scenario: Candidate rule improves accuracy and passes bootstrap → accepted
- **WHEN** a candidate rule’s validation rollouts reduce error rate by ≥10% relative (RER ≥ 0.1).
- **AND WHEN** `changed_fraction >= 0.01`.
- **AND WHEN** bootstrap verification confirms the improvement probability exceeds the configured threshold.
- **THEN** Stage‑B MUST add the rule to mission guidance and record a benchmark snapshot.

#### Scenario: Candidate rule fails RER or bootstrap → rejected
- **WHEN** a candidate rule does not meet the gate thresholds (RER, changed_fraction, bootstrap).
- **THEN** Stage‑B MUST NOT add the rule to mission guidance and MUST record the rejection in `rule_candidates.jsonl`.

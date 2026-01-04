# stage-b-training-free — Delta Spec (Scaffold + Hypothesis Gating)

## ADDED Requirements

### Requirement: Stage‑B MUST support mission-scoped immutable scaffold experiences (`S*`) seeded from `initial_guidance`.
Stage‑B MUST allow missions to define a set of **immutable scaffold rules** in `initial_guidance`, expressed as experience keys `S1..Sn` (mission‑wise).

Scaffold rules are **structural invariants** (e.g., evidence coverage requirements, global/local evidence discipline, multi-subject primary evidence selection, third-state prohibition policy). They MUST:
- be visible to rollout prompts every time,
- be treated as read-only (MUST NOT be modified by reflection), and
- NOT depend on mutable guidance content (e.g., MUST NOT require “G1 contains keywords” to activate).
- have the highest priority over `G0+`.

#### Scenario: Mission seeds scaffold keys and they remain immutable during a run
- **WHEN** a mission is seeded from `initial_guidance` that contains `S1..Sn` entries.
- **THEN** Stage‑B rollout prompts MUST include those `S*` rules for all tickets.
- **AND THEN** reflection MUST NOT update/delete/merge any `S*` rule.

### Requirement: Stage‑B MUST support a hypothesis pool with deterministic delayed promotion into guidance.
Stage‑B MUST support a mission-scoped hypothesis workflow for reflection:
- ops pass SHOULD output hypotheses as candidates for learnable rule discovery;
- the system MUST persist hypotheses and accumulate support evidence across reflection cycles;
- hypotheses MUST be promoted into `guidance.json` only after passing deterministic thresholds.

Hypotheses MUST be:
- binary (only “通过/不通过” conclusions),
- generalizable (no sample identifiers / object-chain copy),
- falsifiable (include a short falsifier condition),
- free of third-state wording (including common variants of “复核/佐证/不应直接/证据不足/待定”等).
- `dimension` is optional; if present, it MUST NOT be `brand` (or brand-equivalent).

#### Scenario: Hypothesis is promoted only after repeated support across cycles
- **WHEN** ops pass proposes a hypothesis `H` with evidence ticket_keys.
- **AND WHEN** `H` is proposed again in a later reflection cycle with additional evidence.
- **THEN** the hypothesis pool MUST accumulate support counts and evidence union.
- **AND THEN** `H` MUST be promoted into a new `G*` experience only after it reaches the configured thresholds (e.g., ≥2 cycles and ≥K unique ticket_keys).

## MODIFIED Requirements

### Requirement: Stage‑B SHALL perform two-pass reflection to update mission guidance from gradient candidates.
Stage‑B MUST run reflection only on **gradient candidates** (tickets with learnable signal), and MUST split reflection into two passes:
1) **Decision pass**: classify stop-gradient cases after seeing `gt_label`.
2) **Ops pass**: propose strict JSON updates using only learnable cases.

Gradient-candidate eligibility (group-level) MUST include at least:
- `label_match=false` (final selected verdict mismatches `gt_label` after deterministic overrides),
- rollout contradiction (format_ok candidates contain both `pass` and `fail` verdicts), OR
- `low_agreement=true` where `vote_strength < manual_review.min_verdict_agreement`.

The pipeline MUST treat `conflict_flag` and `needs_manual_review` as additional eligibility signals when present (they MUST NOT be interpreted as stop-gradient).

Signal definitions:
- `vote_strength`: majority-vote ratio on format_ok candidates, range `[0.0, 1.0]`.
- `low_agreement`: `vote_strength < manual_review.min_verdict_agreement` (mission configs default to 0.67).
- `label_match`: whether the final selected verdict (after deterministic overrides) equals `gt_label`.
- `conflict_flag`: `label_match=false`.
- `needs_manual_review`: a group-level observability flag for “high uncertainty even if label_match=true”; it MUST NOT change the semantics of `label_match/conflict_flag`.

Decision pass output schema MUST be a single strict JSON object:
```json
{
  "no_evidence_group_ids": ["QC-xxx::fail", "QC-yyy::pass"],
  "decision_analysis": "..."
}
```
Notes:
- `no_evidence_group_ids` values MUST be ticket_keys (`{group_id}::{gt_label}`) from the decision pass input.

Ops pass output schema MUST be a single strict JSON object and MUST support hypotheses:
```json
{
  "has_evidence": true,
  "evidence_analysis": "...",
  "operations": [
    {"op":"add","text":"...","rationale":"...","evidence":["QC-1::fail","QC-2::pass"]},
    {"op":"update","key":"G1","text":"...","rationale":"...","evidence":["QC-3::fail"]},
    {"op":"delete","key":"G2","rationale":"...","evidence":["QC-4::pass"]},
    {"op":"merge","key":"G3","merged_from":["G4","G5"],"text":"...","rationale":"...","evidence":["QC-6::fail"]}
  ],
  "hypotheses": [
    {"text":"...","dimension":"global_local","falsifier":"...","evidence":["QC-7::fail","QC-8::pass"]}
  ],
  "coverage": {"learnable_ticket_keys":[], "covered_ticket_keys":[], "uncovered_ticket_keys":[]}
}
```
Notes:
- `hypotheses` is optional but, when present, MUST be validated with the same evidence rules as operations.
- `coverage` is optional and MUST be treated as advisory (system-computed sets are source of truth).
- `S*` scaffold keys MUST be treated as read-only and MUST NOT be targeted by ops.
- `G0+` MUST be mutable (add/update/delete/merge allowed), with the exception that `G0` MUST always exist and MUST NOT be removed.
- A single stop-gradient queue file is produced; no manual-review queue artifacts are produced.

Strict evidence requirements:
- Every operation (including `delete`) MUST include non-empty `evidence`.
- Every hypothesis MUST include non-empty `evidence`.
- `operations[*].evidence` and `hypotheses[*].evidence` MUST be subsets of learnable ticket_keys (ops pass input).
- The system MUST NOT apply any “missing evidence ⇒ default whole bundle” fallback.

Learnability closure and bounded retries:
- Let `L` be learnable ticket_keys for ops pass input.
- Let `E` be the union of validated `operations[*].evidence`.
- Let `H` be the union of validated `hypotheses[*].evidence`.
- The system MUST enforce closure `L == (E ∪ H)` by retrying uncovered groups `L \\ (E ∪ H)` via reflection-only (no re-rollout).
- Retries MUST be bounded by `reflection.retry_budget_per_group_per_epoch` (default 2) and an optional mission-level cap `reflection.max_calls_per_epoch` (decision+ops calls).

#### Scenario: Final verdict correct but rollouts contradict → still eligible for decision pass
- **WHEN** a ticket’s final verdict matches `gt_label` (`label_match=true`).
- **AND WHEN** format_ok rollouts contain both `pass` and `fail` verdicts.
- **THEN** the ticket MUST be included in reflection decision pass input.

#### Scenario: Ops pass enforces strict evidence and retries uncovered groups
- **WHEN** ops pass receives learnable groups `L` as input CASES.
- **AND WHEN** the proposal’s validated operations and hypotheses only cover `E ∪ H` where `E ∪ H != L`.
- **THEN** the system MUST enqueue `L \\ (E ∪ H)` for bounded retry (reflection-only) and MUST NOT silently drop them.

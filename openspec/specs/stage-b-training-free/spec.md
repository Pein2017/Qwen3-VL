# stage-b-training-free Specification

## Purpose
Define the training-free Stage‑B pipeline for group/ticket verdicts: prompt-only rollouts, deterministic selection, **two-pass reflection** that updates mission guidance via strict JSON operations, and **stop-gradient `need_review` routing** for samples that remain unlearnable even after seeing `gt_label`.

## Requirements

### Requirement: Stage‑B SHALL generate group verdict candidates via prompt-only rollouts under a strict two-line binary contract.
Stage‑B MUST generate one or more candidates per ticket using mission guidance plus Stage‑A summaries (without exposing GT labels to the rollout prompt). The decoded output MUST be strictly two lines:
- `Verdict: 通过|不通过`
- `Reason: <single-line Chinese rationale>`
The final output MUST NOT contain any third-state wording (e.g., “需复核/待定/证据不足/need-review”).

#### Scenario: Rollout produces parseable candidates per ticket
- **WHEN** Stage‑B runs rollout for a mission with a configured decode grid and `samples_per_decode`.
- **THEN** each candidate response MUST be parseable into `(verdict, reason)` while preserving the raw text for auditing.
- **THEN** candidate-level metadata MUST include decode parameters, `candidate_index`, and `ticket_key="{group_id}::{label}"`.

### Requirement: Stage‑B SHALL select one final verdict per ticket and export deterministic artifacts.
Stage‑B MUST select a single final verdict per ticket (e.g., majority vote with deterministic tie-break) and export per-run artifacts under `{output.root}/{run_name}/{mission}/` with stable JSONL schemas. Hard failures (e.g., malformed two-line outputs, no usable candidates, selection errors) MUST be logged to `failure_malformed.jsonl` and MUST NOT be routed to human review.

#### Scenario: Selection completes and exports stable outputs
- **WHEN** selection completes for a ticket after rollout.
- **THEN** `selections.jsonl` MUST contain one record for the ticket with final verdict, reason, `vote_strength` (when available), and determinism/provenance fields (epoch, steps, guidance step, reflection cycle).
- **THEN** `metrics.jsonl` MUST include step-wise windows and per-epoch summaries consistent with `selections.jsonl`.

### Requirement: Stage‑B SHALL perform two-pass reflection to update mission guidance from gradient candidates.
Stage‑B MUST run reflection only on **gradient candidates** (tickets with learnable signal), and MUST split reflection into two passes:
1) **Decision pass**: classify stop-gradient cases after seeing `gt_label`.
2) **Ops pass**: propose strict JSON guidance operations using only learnable cases.

Gradient-candidate eligibility (group-level) MUST include at least:
- `label_match=false` (final selected verdict mismatches `gt_label` after deterministic overrides),
- rollout contradiction (format_ok candidates contain both `pass` and `fail` verdicts), OR
- `low_agreement=true` where `vote_strength < manual_review.min_verdict_agreement`.

The pipeline MUST treat `conflict_flag` and `needs_manual_review` as additional eligibility signals when present (they MUST NOT be interpreted as stop-gradient).

Signal definitions:
- `vote_strength`: majority-vote ratio on format_ok candidates, range `[0.0, 1.0]`.
- `low_agreement`: `vote_strength < manual_review.min_verdict_agreement`.
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

Ops pass output schema MUST be a single strict JSON object:
```json
{
  "has_evidence": true,
  "evidence_analysis": "...",
  "operations": [
    {"op":"add","text":"...","rationale":"...","evidence":["QC-1","QC-2"]},
    {"op":"update","key":"G1","text":"...","rationale":"...","evidence":["QC-3"]},
    {"op":"delete","key":"G2","rationale":"...","evidence":["QC-4"]},
    {"op":"merge","key":"G3","merged_from":["G4","G5"],"text":"...","rationale":"...","evidence":["QC-6"]}
  ],
  "coverage": {"learnable_group_ids":[], "covered_group_ids":[], "uncovered_group_ids":[]}
}
```
Notes:
- `coverage` is optional and MUST be treated as advisory (system-computed sets are source of truth).
- `S*` MUST be treated as read-only scaffold (mission-wise structural invariants, highest priority).
- `G0+` MUST be mutable (add/update/delete/merge allowed), except `G0` MUST NOT be deleted.

Strict evidence requirements:
- Every operation (including `delete`) MUST include non-empty `evidence`.
- `operations[*].evidence` MUST be a subset of learnable `ticket_key`s (ops pass input; i.e., `{group_id}::{gt_label}`).
- The system MUST NOT apply any “missing evidence ⇒ default whole bundle” fallback.

Learnability closure and bounded retries:
- Let `L` be learnable groups for ops pass input; let `E` be the union of validated `operations[*].evidence`.
- The system MUST enforce closure `L == E` by retrying uncovered groups `L \\ E` via reflection-only (no re-rollout).
- Retries MUST be bounded by `reflection.retry_budget_per_group_per_epoch` (default 2) and an optional mission-level cap `reflection.max_calls_per_epoch` (decision+ops calls).

#### Scenario: Final verdict correct but rollouts contradict → still eligible for decision pass
- **WHEN** a ticket’s final verdict matches `gt_label` (`label_match=true`).
- **AND WHEN** format_ok rollouts contain both `pass` and `fail` verdicts.
- **THEN** the ticket MUST be included in reflection decision pass input.

#### Scenario: Ops pass enforces strict evidence and retries uncovered groups
- **WHEN** ops pass receives learnable groups `L` as input CASES.
- **AND WHEN** the proposal’s validated operations only cover `E` where `E != L`.
- **THEN** the system MUST enqueue `L \\ E` for bounded retry (reflection-only) and MUST NOT silently drop them.

### Requirement: Stage‑B SHALL route `need_review` as stop-gradient-only after decision pass.
Stage‑B MUST treat `need_review_queue.jsonl` / `need_review.json` as the **stop-gradient queue**: tickets that remain unlearnable after seeing `gt_label` (root cause is intentionally not distinguished).

Need-review routing MUST be:
- **Strict ticket_key granularity** (i.e., `{group_id}::{gt_label}`),
- **Decision-pass single source of truth** (`no_evidence_group_ids`),
- **Non-sticky across epochs** (re-evaluated each epoch),
- **Stop-gradient isolation** (stop-gradient tickets MUST NOT appear in ops pass inputs nor in validated `operations[*].evidence`, and MUST NOT drive rule hit/miss feedback).

When bounded retry budgets are exhausted for uncovered learnable groups, the system MUST route them to need-review with a distinct `reason_code` (e.g., `budget_exhausted`) to preserve auditability.

#### Scenario: Decision pass declares no-evidence → ticket is routed to need-review and excluded from ops pass
- **WHEN** decision pass includes a ticket’s `ticket_key` (i.e., `{group_id}::{gt_label}`) in `no_evidence_group_ids`.
- **THEN** Stage‑B MUST append the ticket to `need_review_queue.jsonl` and aggregate into `need_review.json`.
- **AND THEN** the ticket MUST NOT appear in ops pass inputs and MUST NOT be counted in evidence coverage `E`.

### Requirement: Stage‑B multi-GPU execution SHALL be ticket-parallel for rollout and single-writer for artifacts.
When running in distributed mode, Stage‑B SHOULD parallelize rollout across ranks (ticket-parallel workers) while keeping selection and reflection centralized (rank 0). Only rank 0 MUST write artifacts under the run directory; non-zero ranks MUST NOT race on file writes.

#### Scenario: Multi-GPU rollout runs without artifact races
- **WHEN** `scripts/stage_b.sh` launches Stage‑B via `torchrun`.
- **THEN** rollout work SHOULD be distributed across ranks.
- **THEN** selection/reflection and artifact writes MUST be performed by rank 0 only.

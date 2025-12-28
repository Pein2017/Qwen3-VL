# stage-b-training-free Specification

## Purpose
Define the training-free Stage‑B pipeline for group/ticket verdicts in **rule-search mode only**: prompt-only rollouts, rule proposer + metric-gated updates, and deterministic artifact logging.

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

### Requirement: Stage‑B SHALL run rule-search as the only supported execution mode.
Stage‑B MUST execute the rule-search loop: baseline rollout on a train pool → proposer emits 1–N rule candidates → A/B gate on train pool → optional eval-pool auditing → apply only gated candidates. Legacy reflection/selection paths MUST NOT be available.

#### Scenario: Rule-search loop proposes and gates candidate rules
- **WHEN** Stage‑B runs with `rule_search.train_sampler` and `rule_search.eval_sampler`.
- **THEN** the proposer MUST emit candidate operations in strict JSON and the system MUST validate them before evaluation.
- **THEN** only candidates that pass metric gating (relative error reduction + bootstrap + churn caps) may be applied.
- **THEN** eval-pool metrics MUST be logged for audit but MUST NOT veto accepted candidates.

### Requirement: Stage‑B SHALL export deterministic rule-search artifacts.
Stage‑B MUST write rule-search artifacts under `{output.root}/{run_name}/{mission}/` including:
- `rule_candidates.jsonl` (proposed ops with gate metrics)
- `benchmarks.jsonl` (accepted ops with baseline/after metrics)
- `rule_search_hard_cases.jsonl` and `rule_search_candidate_regressions.jsonl` (audit trails)

#### Scenario: Rule-search artifacts are written by rank 0 only
- **WHEN** Stage‑B runs with multiple GPUs.
- **THEN** rollout MUST be ticket-parallel while rank 0 writes artifacts exclusively.

### Requirement: Stage‑B SHALL optionally export rule-search distillation chat logs after convergence.
When `stage_b_distillation.enabled=true` and rule-search reaches early-stop, Stage‑B MUST sample `distill_size` tickets from the input pool, run low-temperature rollouts, and export ChatML conversations to `distill_chatml.jsonl` (or `log_chatml_path` when provided).

#### Scenario: Distill export after rule-search early stop
- **WHEN** rule-search triggers early-stop and distillation is enabled.
- **THEN** the system MUST sample `distill_size` tickets (seeded for reproducibility), run low-temperature rollouts, and export ChatML with system/user prompts plus a two-line assistant verdict.

# stage-b-training-free Specification (Delta)

## MODIFIED Requirements

### Requirement: Stage‑B SHALL run rule-search as the only supported execution mode.
Stage‑B MUST execute the rule-search loop by default: baseline rollout on a train pool → proposer emits 1–N rule candidates → A/B gate on train pool → optional eval-pool auditing → apply only gated candidates.

Stage‑B MUST ALSO support a baseline-only audit execution that skips proposer/reflection and gating when explicitly requested (e.g., CLI `--jump-reflection` or YAML `jump_reflection: true`).

#### Scenario: Baseline-only audit skips proposer/reflection
- **WHEN** Stage‑B runs with `--jump-reflection` (or `jump_reflection: true` in config).
- **THEN** Stage‑B MUST run a baseline rollout and export baseline audit artifacts.
- **THEN** Stage‑B MUST NOT call proposer/reflection or apply any guidance updates.

#### Scenario: Default rule-search behavior is unchanged
- **WHEN** Stage‑B runs without `--jump-reflection`.
- **THEN** Stage‑B MUST execute the full rule-search loop with proposer + gate + optional eval audit.

## ADDED Requirements

### Requirement: Stage‑B SHALL export baseline audit artifacts in jump_reflection mode.
When jump_reflection is enabled (`--jump-reflection` or config `jump_reflection: true`), Stage‑B MUST write baseline audit artifacts under `{output.root}/{mission_name}/{output.run_name}/`:
- `baseline_metrics.json`
- `baseline_ticket_stats.jsonl`
- `baseline_wrong_cases.jsonl`

#### Scenario: Baseline audit artifacts are written by rank 0 only
- **WHEN** Stage‑B runs with multiple GPUs in jump_reflection mode.
- **THEN** rollout MUST be ticket-parallel while rank 0 writes baseline artifacts exclusively.

# Proposal: Add Stage‑B `jump_reflection` (baseline-only audit)

## Motivation
In some missions, the Stage‑B proposer/reflection model cannot reliably produce useful rule candidates. In these cases, iterating the proposer wastes time and can pollute artifacts without improving verdict accuracy.

This change adds a **baseline-only audit mode** that:
- runs Stage‑B rollouts over the available tickets,
- skips proposer/reflection and metric gating entirely,
- dumps compact artifacts to support manual diagnosis and manual edits to `output_post/stage_b/initial_guidance.json`.

## Scope
- Add a `--jump-reflection` CLI flag (and `jump_reflection=true` convenience in `scripts/stage_b.sh`).
- In `jump_reflection` mode, run a baseline rollout and write audit artifacts:
  - `baseline_metrics.json`
  - `baseline_ticket_stats.jsonl`
  - `rule_search_hard_cases.jsonl` (baseline sampler tag) for quick triage

## Non-goals
- No automatic modification of `initial_guidance.json`.
- No rule candidate generation, gating, or promotion in `jump_reflection` mode.

## Risks
- Baseline-only mode can be mistaken for full rule-search; artifacts must clearly indicate mode.
- Distillation export may be expensive; this mode does not attempt to converge or export distillation.

## Success criteria
- `--jump-reflection` produces the baseline artifacts and exits without running proposer/reflection.
- Existing rule-search behavior is unchanged when the flag is not provided.


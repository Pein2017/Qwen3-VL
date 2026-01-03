# Stage-B Diagnosis & Review (Rule-Search)

Status: Active
Scope: Rule-search diagnostics checklist and triage flow for Stage-B runs.
Owners: Runtime
Last updated: 2026-01-02
Related: [runtime/STAGE_B_RUNTIME.md](../runtime/STAGE_B_RUNTIME.md), [runtime/STAGE_A_STAGE_B.md](../runtime/STAGE_A_STAGE_B.md)

Stage‑B now runs **rule_search only**. Legacy selection/reflection diagnostics are removed; consult Git history if needed.

## Rule-search review checklist

- `rule_candidates.jsonl`: inspect promoted vs rejected candidates and gate metrics (RER, changed_fraction, bootstrap_prob).
- `benchmarks.jsonl`: track train/eval deltas per iteration; watch for eval accuracy drop.
- `rule_search_hard_cases.jsonl`: sample hard tickets for prompt or coverage gaps.
- `rule_search_candidate_regressions.jsonl`: inspect regression lists for recurring failure modes.
- `guidance.json` + `snapshots/`: verify accepted guidance edits and audit history.
- `distill_chatml.jsonl` (if enabled): confirm low‑temperature ChatML samples are emitted for SFT.

## Suggested triage flow

1. Start from `rule_search_candidate_regressions.jsonl` to identify harmful candidates.
2. Validate `benchmarks.jsonl` trends to ensure gating is improving metrics.
3. Review `rule_search_hard_cases.jsonl` to refine proposer prompts or add explicit guidance.
4. Promote verified guidance changes from the run-local `guidance.json` to the shared seed only after approval.

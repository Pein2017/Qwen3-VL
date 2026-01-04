# Design: Rule‑Search + Info‑Gain Gating for Stage‑B

## Overview
The new `mode=rule_search` converts Stage‑B guidance learning into a **greedy tree-growth loop**:

1) **Baseline rollout** on a large validation sample (low temperature; multiple seeds/samples per ticket).
2) Compute per-ticket **difficulty** and pick `reflect_size` high-value mismatches.
3) **Reflection proposer** outputs `K` candidate rules (binary-only).
4) For each candidate rule, run **A/B validation rollouts** on the same validation set.
5) Accept the best rule if it passes the **gate** (relative error reduction + bootstrap + changed_fraction).
6) Record a benchmark snapshot; repeat until early stopping.

The key shift is that **truth-consistency is enforced by rollout-measured gain**, not by reflection heuristics.

## Data Model

### Per-ticket rollouts (Monte‑Carlo)
For ticket `i`, run `M` independent rollouts (typically temperature=0.1 with seed offsets) and get a verdict distribution:
- `p_pass(i) = #pass / M`
- `p_fail(i) = #fail / M`

Define:
- `majority_pred(i) = argmax(p_pass, p_fail)` with deterministic tie-break
- `majority_correct(i) = 1[majority_pred(i) == gt(i)]`
- `difficulty(i) = 1 - max(p_pass(i), p_fail(i))` (ambiguity; higher is harder)
- `hard_wrong(i) = max(p_pass, p_fail)` if `majority_pred != gt` else `0` (high-confidence mismatch)

### Validation metric: Relative Error Reduction (RER)
Compute accuracy on a set of tickets using majority vote:
- `acc = mean_i majority_correct(i)`
- `err = 1 - acc`
- `RER = (err_base - err_new) / max(err_base, eps)`

Acceptance requires `RER >= 0.1` on the validation set.

### Coverage sanity: changed_fraction
Let `changed(i) = 1[majority_pred_base(i) != majority_pred_new(i)]`.
`changed_fraction = mean_i changed(i)`.
Acceptance requires `changed_fraction >= 0.01`.

### Bootstrap verification
To reduce variance and avoid “lucky” acceptances:
- Perform `B` bootstrap resamples over tickets (replacement=true) on the validation set.
- For each resample, recompute `RER`.
Gate passes if `P(RER >= threshold) >= bootstrap_min_prob` (e.g., 0.8).

Bootstrap is applied to **ticket-level** outcomes (majority_correct and majority_pred change), avoiding model re-runs.

## Holdout Strategy
- Default: holdout fraction 0.2 stratified by label.
- Per-mission override: `挡风板安装检查` uses 0.1.
- Holdout is treated as a safety report by default (optional hard constraint via config).

## Reflection Proposer
Reflection becomes “low difficulty”:
- Input: `reflect_size=16` tickets selected from baseline mismatches (prefer high-confidence wrong).
- Output: `num_candidate_rules=3` candidate rules (binary-only, no third-state wording, no brand dimension).
- Output schema is strict JSON:
```json
{"rules":[{"text":"...","rationale":"..."}]}
```

No direct `operations` are permitted in rule-search mode.

## Determinism and Reproducibility
- Ticket sampling, holdout split, and bootstrap RNG MUST be seeded via config.
- Validation runs are paired A/B on the exact same ticket set and decode seeds.

## Artifacts
Under `{output.root}/{mission}/{run_name}/`:
- `rule_candidates.jsonl`: for each iteration and candidate rule:
  - rule text/signature, baseline metrics, candidate metrics, RER, changed_fraction, bootstrap stats, accept/reject decision
- `benchmarks.jsonl`: one record per accepted rule:
  - baseline metrics, after metrics, guidance step, timestamp, config snapshot hash/seed

## Compatibility
- `legacy_reflection` mode remains available for existing workflows.
- `rule_search` mode disables stop-gradient queue routing and does not run decision/ops reflection passes.

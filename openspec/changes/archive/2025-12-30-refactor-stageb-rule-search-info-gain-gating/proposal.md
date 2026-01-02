# Proposal: Refactor Stage‑B to Rule‑Search + Info‑Gain Gating

## Summary
Stage‑B currently learns mission guidance primarily via reflection outputs gated by schema validity and small-sample heuristics (e.g., support cycles / evidence membership). This can admit semantically wrong rules, and reflection is the throughput bottleneck.

This change introduces a **rule-search (tree-growth) mode** where reflection is reduced to a lightweight **rule proposer**, and all rule admission is controlled by a **metric gate** computed from **large-scale low-temperature rollouts** with **bootstrap verification**.

## Goals
- Make guidance updates **truth-consistent**: only admit a rule if it measurably improves correctness on new tickets.
- Reduce reflection cost: reflection proposes **K=3** candidate rules from **reflect_size=16** high-value mismatches.
- Use rollout throughput to validate rules: A/B evaluate candidate rules on a large validation sample.
- Use a simple deterministic acceptance gate:
  - Primary metric: **relative error reduction** (RER) with threshold **0.1**
  - Coverage sanity: `changed_fraction >= 0.01`
  - Robustness: **bootstrap** verification on the evaluation set (and optional holdout safety check)
- Support per-mission holdout fractions (e.g., `挡风板安装检查` uses **0.1**).

## Non‑Goals
- No additional LLM pass beyond the proposer reflection.
- No attempt to “fix labels” or create a manual review pipeline.
- No changes to Stage‑A outputs or summary formats.

## Proposed User‑Facing Changes
- Add a Stage‑B `mode`:
  - `legacy_reflection` (default, current behavior)
  - `rule_search` (new)
- Add `rule_search` config block for:
  - train/holdout split (per-mission overrides)
  - proposer parameters (`reflect_size`, `num_candidate_rules`)
  - validation sampling sizes/fractions
  - gate thresholds and bootstrap settings
  - early stopping
- New per-run artifacts under `{output.root}/{mission}/{run_name}/`:
  - `rule_candidates.jsonl`: proposed rules + per-rule A/B metrics + decision
  - `benchmarks.jsonl`: accepted-rule step history and baseline/after metrics

## Success Criteria / Acceptance
- In `mode=rule_search`, Stage‑B MUST NOT apply reflection operations directly; guidance changes MUST occur only via gate acceptance.
- Gate uses relative error reduction (RER) and bootstrap:
  - RER >= 0.1 on validation set to accept
  - changed_fraction >= 0.01 on validation set to accept
  - bootstrap check passes (configurable)
- `挡风板安装检查` supports `holdout_fraction=0.1` override.
- Unit tests cover:
  - RER computation and gate logic
  - bootstrap verification behavior
  - changed_fraction calculation
  - per-mission holdout override selection
- OpenSpec validation passes: `openspec validate refactor-stageb-rule-search-info-gain-gating --strict`


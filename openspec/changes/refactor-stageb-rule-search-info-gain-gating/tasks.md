# Tasks: Refactor Stage‑B to Rule‑Search + Info‑Gain Gating

- [x] Add `mode` to Stage‑B config (`legacy_reflection` | `rule_search`)
- [x] Add `rule_search` config block (holdout split, proposer, validation sampling, gate thresholds, bootstrap, early stop)
- [x] Implement rule-search loop (baseline → propose → validate K rules → gate accept/reject → repeat)
- [x] Implement per-ticket rollout stats (p_pass/p_fail, majority_pred, difficulty, hard_wrong)
- [x] Implement metrics and gate:
  - [x] Relative error reduction (RER) with threshold 0.1
  - [x] changed_fraction threshold 0.01
  - [x] bootstrap verification (ticket-level resampling)
- [x] Add new artifacts: `rule_candidates.jsonl`, `benchmarks.jsonl`
- [x] Ensure “only gate writes guidance” in rule-search mode
- [x] Update prompts: add rule proposer prompt template (binary-only rules; forbid third-state/brand)
- [x] Update docs for new mode and configs
- [x] Add/Update unit tests under `tests/stage_b` for:
  - [x] metric computations and bootstrap behavior
  - [x] sampling / holdout override selection
  - [x] gate accept/reject logic
- [x] Run `conda run -n ms pytest -q tests/stage_b`
- [x] Run `openspec validate refactor-stageb-rule-search-info-gain-gating --strict`

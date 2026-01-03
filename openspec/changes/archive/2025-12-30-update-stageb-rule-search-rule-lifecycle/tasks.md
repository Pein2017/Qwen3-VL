# Tasks: update-stageb-rule-search-rule-lifecycle

- [ ] Add rule lifecycle operation schema to rule_search proposer (update/merge/remove) with strict JSON validation.
- [ ] Implement ablation evaluation for existing rules and promote update/remove when gates pass.
- [ ] Add fp‑prioritized gates: fp_rate improvement requirement, acc improvement requirement, max_fp_rate_increase, max_changed_fraction.
- [ ] Update Stage‑B guidance repository to apply rule updates/removals in rule_search mode.
- [ ] Update configs to expose new thresholds (defaults: max_changed_fraction=0.05, max_fp_rate_increase=0.01).
- [ ] Add tests or smoke checks for rule_search lifecycle ops and gating.
- [ ] Update docs: docs/training/REFERENCE.md, docs/runtime/STAGE_A_STAGE_B.md, docs/reference/stage-B-knowledge-Chinese.md.

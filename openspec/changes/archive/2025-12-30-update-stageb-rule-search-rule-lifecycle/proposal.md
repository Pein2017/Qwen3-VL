# Proposal: update-stageb-rule-search-rule-lifecycle

## Summary
Extend Stage‑B rule_search to manage rule lifecycle operations (update/merge/remove) and to identify harmful rules via metric‑gated ablation. Add fp‑prioritized safety gates with configured thresholds (max_changed_fraction=0.05, max_fp_rate_increase=0.01).

## Motivation
Current rule_search only appends new rules, which can accumulate inaccurate or harmful rules. Without a rollback/refresh mechanism, fp risk can grow. The change enables targeted rule updates/removals while preserving deterministic, metric‑gated behavior.

## Scope
- Add lifecycle operations in rule_search proposals: update/merge/remove (in addition to upsert).
- Add negative‑impact evaluation for existing rules using ablation against the current guidance set.
- Add fp‑prioritized gates and thresholds to accept updates/removals.

## Non‑Goals
- No change to Stage‑A pipeline or SFT training.
- No change to legacy_reflection behavior unless explicitly specified.

## Assumptions
- Business priority is minimizing false positives (fp) over false negatives (fn).
- Default thresholds: max_changed_fraction=0.05, max_fp_rate_increase=0.01.
- Update/merge/remove acceptance requires both fp_rate improvement and acc improvement.

## Risks
- Over‑aggressive rule removal could degrade recall; mitigated by fp/acc improvement gates + max_changed_fraction.
- Additional rollouts for ablation increase runtime; mitigated by limiting per‑epoch candidates.

## Success Criteria
- Harmful rules can be detected and removed/updated without increasing fp.
- Accepted update/merge/remove operations improve both fp_rate and acc.
- Accepted operations do not require `max_acc_drop` gating; acc improvement is mandatory.
- Prediction churn remains below `max_changed_fraction` threshold.

## Decisions
- max_fp_rate_increase=0.01 (configurable).
- update/merge/remove require acc improvement as well as fp_rate improvement.

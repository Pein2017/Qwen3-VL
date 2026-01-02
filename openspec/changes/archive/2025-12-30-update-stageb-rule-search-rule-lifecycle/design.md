# Design: update-stageb-rule-search-rule-lifecycle

## Overview
Introduce rule lifecycle operations in rule_search: update, merge, and remove. Add a metric‑gated ablation path to detect harmful existing rules and apply corrective operations with fp‑prioritized gates.

## Lifecycle Operations
- **upsert**: add a new rule (current behavior).
- **update**: replace text of an existing rule (identified by signature key).
- **merge**: combine multiple existing rules into a single replacement rule.
- **remove**: delete an existing rule.

## Rule Identification
- Continue using normalized signature for de‑duplication and lookup.
- For update/merge/remove, the proposer MUST provide target signature(s).

## Negative‑Impact (Harm) Detection
- For each existing rule candidate, build an ablation guidance set by removing the rule.
- Compare metrics against baseline (current guidance) on the same train/eval pools.
- Mark a rule as harmful if ablation improves both fp_rate and acc without violating gates.

## Gates (fp‑prioritized)
- **fp_rate improvement**: update/merge/remove MUST improve fp_rate (delta < 0).
- **acc improvement**: update/merge/remove MUST improve acc (delta > 0).
- **max_fp_rate_increase**: fp_rate MUST NOT increase beyond `max_fp_rate_increase=0.01` (configurable).
- **max_changed_fraction**: prediction churn <= 0.05 (configurable).
- Existing bootstrap gate remains in effect.

## Outputs / Artifacts
- Log accepted lifecycle ops to rule_candidates/benchmarks with operation type and targets.
- Record ablation metrics for transparency.

## Compatibility
- Legacy reflection path remains unchanged.
- rule_search JSON schema is extended; backward‑compatible if `operations` is optional and defaults to rules list.

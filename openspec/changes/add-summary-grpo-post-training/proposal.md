# Proposal: add-summary-grpo-post-training

## Summary
- Introduce a dedicated specification for GRPO post-training of summary-mode outputs (Stage-A summaries), covering reward design, format contract, and rollout settings.
- Formalize single-line irrelevant output (`无关图片`) while keeping two-line format for non-irrelevant summaries, with `_fusion_domain + mission` header matching.
- Deprecate the legacy `grpo-integration` spec in favor of the new summary-focused GRPO spec.

## Motivation
- Summary-mode post-training now targets strict format stability and schema-accurate content; this differs materially from the historical Stage-B GRPO verdict spec.
- Consolidating summary GRPO requirements in a new spec reduces ambiguity and aligns with current pipelines (summary_runtime, per-epoch irrelevant prompt alternation).

## Scope
- New `summary-grpo-post-training` spec with requirements for reward wiring, format/JSON validation, header matching, and rollout settings.
- Deprecation of `grpo-integration` as the authoritative GRPO spec for summary-mode training.

## Non-goals
- No changes to Stage-B verdict GRPO pipelines or two-line verdict contract.
- No new model architectures or new data converters.
- No length penalty or length-based reward.

## Risks
- Dual-spec period may confuse readers if deprecation is not surfaced clearly in docs.
- Reward strictness could reduce exploration; must balance penalties with content accuracy goals.

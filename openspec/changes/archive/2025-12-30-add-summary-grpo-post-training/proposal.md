# Proposal: add-summary-grpo-post-training

## Summary
- Introduce a dedicated specification for GRPO post-training of summary-mode outputs (Stage-A summaries), covering reward design, format contract, rollout settings, and the launch path via `scripts/train.sh`.
- Formalize single-line irrelevant output (`无关图片`) while keeping two-line format for non-irrelevant summaries, with `<DOMAIN=BBU|RRU>, <TASK=SUMMARY>` header matching.
- Require `rlhf` config block usage, fusion-based dataset toggles, and summary prompt profile alignment (`summary_runtime` + `assistant_prefix_format`).
- Deprecate the legacy `grpo-integration` spec in favor of the new summary-focused GRPO spec.

## Motivation
- Summary-mode post-training now targets strict format stability and schema-accurate content; this differs materially from the historical Stage-B GRPO verdict spec.
- Consolidating summary GRPO requirements in a new spec reduces ambiguity and aligns with current pipelines (summary_runtime, per-epoch irrelevant prompt alternation).

## Scope
- New `summary-grpo-post-training` spec with requirements for reward wiring (including metadata summary references and domain tokens), format/JSON validation, header matching, rollout settings, and the GRPO launch path (shared modules via `scripts/train.sh` + `rlhf` block).
- Fusion-based dataset toggle requirements aligned to `configs/fused_data` + `configs/fusion`.
- Deprecation of `grpo-integration` as the authoritative GRPO spec for summary-mode training.

## Non-goals
- No changes to Stage-B verdict GRPO pipelines or two-line verdict contract.
- No new model architectures or new data converters.
- No length penalty or length-based reward.

## Risks
- Dual-spec period may confuse readers if deprecation is not surfaced clearly in docs.
- Reward strictness could reduce exploration; must balance penalties with content accuracy goals.

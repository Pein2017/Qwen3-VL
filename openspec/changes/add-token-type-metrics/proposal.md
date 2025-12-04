# Proposal: Token-type accuracy & entropy telemetry for dense caption SFT

## Summary
Add optional token-type monitoring (desc/coord/format) during SFT to pinpoint whether errors come from textual descriptions, geometric numbers, or JSON formatting. Metrics cover both train and eval, limited to target and lvis datasets, and remain off by default.

## Motivation
- Current `token_acc` is too coarse; grounding regressions can hide behind good text accuracy.
- Fusion runs mix text-only sources; we need per-dataset gating to avoid noise and wasted compute.
- Entropy by token type surfaces confidence gaps (e.g., high uncertainty on coordinates).

## Scope
- SFT training path (`src/sft.py`, ms-swift trainer mixin).
- Collator-based token typing derived from assistant payload + tokenizer.
- Config knobs to enable and scope datasets.
- Unit tests with real tokenizer to ensure label↔type alignment.
- No model architecture changes.

## Non-Goals
- Runtime generation metrics or decoding-time probes.
- Geometry-key metrics or number-span accuracy (explicitly out of scope).
- Stage-B or inference-time dashboards.

## Risks & Mitigations
- **Alignment mismatch** between labels and computed token types → fail-soft: skip metrics for that batch, log debug; covered by tokenizer-based tests.
- **Overhead** from extra logging → keep optional and dataset-filtered.
- **Metric spam** → stable naming and dataset gating; default off.

## Validation Plan
- Unit tests on synthetic payloads (bbox/poly/line_points) asserting type alignment and presence of desc/coord/format types.
- Training smoke: enable metrics on a tiny fusion split; verify logs contain per-type acc/entropy only for target/lvis.

## Rollout
- Ship behind `custom.token_type_metrics.enabled` with include/exclude lists (default include target+lvis, exclude coig_lang_chat).
- Document knobs in config/README or REFERENCE once implemented.

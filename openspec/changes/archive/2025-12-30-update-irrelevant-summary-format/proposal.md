# Proposal: update-irrelevant-summary-format

## Summary
- Clarify the summary prompt contract: irrelevant samples output a single line `无关图片` only; non-irrelevant samples keep the two-line `<DOMAIN=...>, <TASK=...>` + JSON format.
- Identify irrelevant summary samples via `metadata._fusion_source` (e.g., `irrelevant_summary`) while reusing the BBU summary prompt (no separate template).
- Adjust Stage-B prompt wording to remove line-count assumptions for irrelevant summaries.
- Keep the irrelevant pool listed under fusion **targets** for full coverage, while treating it as **source-like** via `_fusion_source` for analytics/handling.
- Alternate irrelevant prompts between `summary_bbu` and `summary_rru` (~50/50) with per-sample randomization during SFT training while preserving a single `_fusion_source` identity.

## Motivation
- Separate true task summaries from noise/irrelevant samples without extra post-processing.
- Keep irrelevant SFT samples aligned with existing BBU-style prompts while allowing one-line output.
- Avoid brittle Stage-B prompt assumptions about summary line formatting.

## Scope
- Prompt contract update for summary-mode templates (training + runtime prompt text only).
- Stage-B prompt wording update (input description only).
- No runtime sanitizers or decoding rules.
- No changes to fusion sampling logic; irrelevant may remain a target with ratio 1.0 to consume all samples.
- Per-sample prompt override for the irrelevant pool only; no template cloning or new dataset entries required.

## Non-goals
- No Stage-A output enforcement or stabilizer; runtime outputs may vary until later post-training stabilization.
- No changes to Stage-B verdict format (still strict two-line verdict output).

## Risks
- Models may still emit the legacy two-line irrelevant format during Stage-A rollout; this proposal does not add enforcement.
- Downstream tools that hard-parse two-line summaries may need to reference `_fusion_source` when deciding irrelevance.

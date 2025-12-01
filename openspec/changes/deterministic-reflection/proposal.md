## Title
Replace Stage-B reflection LLM with deterministic experience updater

## Problem
- Reflection LLM frequently fails formatting (missing ACTION/ops), causing generation_error and stalled guidance updates.
- Even with prompt/parse tweaks, maintenance cost is high and guidance often stays at step=1.

## Proposal
- Remove LLM generation from Stage-B reflection; use a deterministic rule-based updater that converts contradictions/all-wrong/conflict signals into conservative guidance ops.
- Keep rollout/critic/selection unchanged; reflection now updates guidance without parsing LLM text.

## Success Criteria
- reflection.engine configurable; setting `deterministic` applies updates without LLM calls or JSON/line parsing.
- Guidance step increments when contradictions/all-wrong bundles appear; no generation_error entries in reflection log.
- Configs (run/debug) default to deterministic reflection.

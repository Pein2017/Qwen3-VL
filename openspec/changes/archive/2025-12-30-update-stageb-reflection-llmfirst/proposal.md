# Update Stage-B to reflection-first (no critic) with guidance dedup and richer evidence

## Why
- Current Stage-B code already dropped CriticEngine and deterministic tie-break knobs but the spec still mandates them; config knobs (selection.policy/tie_break, signals) are inert.
- Malformed rollouts get salvaged and marked format_ok, bypassing failure/manual-review; reflection eligibility ignores low-agreement/manual-review flags.
- Reflection/guidance updates lack Stage-A evidence grounding and append duplicate rules; mission guidance is reset per run and caches/manual-review queues persist across reruns.
- Group reports only read the first Stage-A file; multi-mission runs miss labels.

## What
- Spec a reflection-first loop using the rollout model for reflection, with eligibility on label mismatch OR low-agreement/uncertainty/manual-review signals.
- Require strict format handling (no auto-salvage) and manual-review when reflection cannot justify GT.
- Enrich reflection/guidance context with Stage-A per-image summaries or parsed Reason segments.
- Define guidance lifecycle: reuse mission guidance per run, dedup/merge/summarize rules on update, and per-run hygiene for reflection cache/manual_review/failure logs.
- Make selection defaults explicit (majority vote; config knobs either honored or pruned) and route malformed/empty reasons to failure + manual-review.
- Ensure group reports merge all provided Stage-A paths for multi-mission runs.

## Impact
- Spec delta for `stage-b-training-free`.
- No critic dependency; simpler signals; clarified artifacts/hygiene and guidance management.***

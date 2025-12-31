# Design: add-stageb-3step-reflection

## Reference pattern
- Borrow the math pipeline: (1) per-trajectory summary, (2) per-problem critique w/ ops add|modify, (3) batch merge/merge+reindex, all strict JSON, cached to disk for reuse.
- Keep training-free: no weight updates; guidance is the optimizer step.

## Proposed Stage-B mapping
1) **Trajectory summary stage**: For each reflection bundle, compress candidate+critic info into a short JSON summary file (one per bundle) to cut token load and freeze upstream noise.
2) **Critique stage**: Consume summaries + current guidance to produce JSON ops (add/modify/merge) limited by `max_operations`; fail on non-JSON.
3) **Batch update stage**: Merge ops, dedup semantically similar rules, compact keys to `G0..Gn`, persist preview + applied guidance; snapshots kept.

## Guardrails
- Eligibility: only bundles with label_conflict / selected_mismatch / mixed label_match ("partial-correct") enter stage 2; others noop with reason.
- Strict parsing: ban fallbacks; stop tokens to prevent multiline; reject texts containing Stage-A-style counts/labels.
- Dedup: normalize whitespace, drop duplicates, stable sort then reindex.
- Caching: each stage writes under `{run}/{mission}/reflection_cache/` with deterministic filenames; rerun reuses if intact (hash/size check optional minimal version: presence check).

## Impacted surfaces
- Prompts: new minimal templates for summary/critique/batch_update.
- Reflection engine: orchestration + parsing + caching + reindexing.
- Guidance writer: whitelist + compaction hook.
- Docs/specs: stage-b-training-free, runtime guide, zh knowledge doc.

## Open questions
- How to score "partial-correct" for binary pass/fail? Tentative: mixed label_match within bundle OR conflict_flag OR needs_manual_review==True. Can refine during implementation.

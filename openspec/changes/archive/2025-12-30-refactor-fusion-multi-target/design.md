# Design: Multi-target fusion with ratio-balanced targets

## Config & parsing
- Accept `targets: [...]` (list). Keep legacy `target:` and normalize to a one-element `targets` list.
- Each target entry may carry optional `ratio`. If absent, default is "use full coverage" (current behavior).
- Enforce unique dataset names across targets and sources during parsing.

## Target scheduling (train)
- For each target i: let `len_i` be pool size after any sample_limit; `ratio_i` default 1.0 when absent.
- Compute `capacity_i = len_i / ratio_i`; `base = floor(min(capacity_i))`. This is the largest scalar that respects the smallest target when ratios are applied.
- Per-target quota: `quota_i = round(base * ratio_i)`, capped at `len_i` (should already hold by construction).
- Shuffle each target pool deterministically per epoch using mixed seed (global seed ⊕ epoch ⊕ target.seed ⊕ constant); take first `quota_i` indices.
- Total target quota = sum(quota_i); this value drives source quotas.

Example: lengths (100, 200, 300), ratios (0.33, 0.33, 0.34)
- capacities ≈ (303, 606, 882) → base = 303
- quotas ≈ round(303*0.33)=100, 100, round(303*0.34)=103
- total target quota = 303

## Source scheduling (train)
- Source quota = `round(source.ratio * total_target_quota)`; sample with replacement from source pool using mixed seed (global ⊕ epoch ⊕ source.seed ⊕ constant).
- Sources remain augmentation=off, curriculum=off, object caps enforced when configured.

## Eval scheduling
- Eval iterates all target val splits concatenated (deterministic order, no shuffle by default).
- Sources are excluded from eval. Optional future `eval_sample_limit` not in scope now.

## Telemetry
- epoch_plan: include per-dataset `count`, `ratio` (if provided), `base`, augmentation/curriculum flags, and cap info.
- last_sample_debug: include dataset name, domain, quota context, and whether ratio balancing applied.

## Backward compatibility
- Single-target configs continue to work; lack of `ratio` means full coverage of each target per epoch.
- Source semantics unchanged except that source quotas now key off combined target quota when multiple targets are present.

## Loss attribution under padded batching
- Problem: dataset identity can be lost when batches mix multiple targets/sources unless labels are preserved for metrics.
- Approach: attach per-sample dataset labels and lengths to padded batches and reuse the grouped-metrics reducer for per-dataset loss/accuracy.
- Outcome: Works with the padding-only runtime; no packing path is required.

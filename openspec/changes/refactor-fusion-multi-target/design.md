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

## Packing & loss attribution (future consideration)
- Problem: ms-swift packing merges multiple samples into one packed sequence, losing per-sample dataset identity for loss monitoring.
- Options to evaluate:
  1) **Group-by-dataset packing**: constrain packing bins so that only samples sharing `dataset_name` are packed together; retain per-packed-row metadata of that dataset. Simple, minimal overhead; per-dataset loss is per packed batch.
  2) **Per-sample tags inside packed rows**: carry a parallel list of `(dataset, length, loss_mask_range)` for each packed element and extend the trainer’s loss reducer to accumulate per-dataset token loss. Heavier change to collate + trainer but more precise when mixed packing is allowed.
- Preferred minimal path: start with group-by-dataset packing to regain dataset-level loss while keeping packing enabled; optionally add fine-grained tagging later.

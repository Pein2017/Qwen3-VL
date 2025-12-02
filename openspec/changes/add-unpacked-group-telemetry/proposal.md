# Proposal: Add per-dataset telemetry when packing is disabled

## Problem
- Packing is enabled by default to gain throughput, but some runs need it off (e.g., expensive augmentation).  
- With packing disabled, the current grouped metrics path (packed_group/packed_segments + GroupedMetricsMixin) is bypassed, so per-dataset metrics (bbu/rru/lvis/lang_chat) disappear.  
- Operators want dataset-level token accuracy/loss even in padded, non-packed batches, logging all datasets in train and only targets in eval unless a source `val` split is provided.

## Goals (in scope)
- Keep `training.packing: false` while emitting per-dataset metrics derived from each sample’s dataset name (e.g., `bbu`, `rru`, `lvis`, `lang_chat`), without extra grouping keys.  
- Preserve standard padded batching (no bin-packing) and default to logging only `{dataset}_loss` / `{dataset}_token_acc`.  
- Config should work out of the box; add a minimal toggle only if needed.

## Non-Goals (out of scope)
- Changing packing algorithms or enabling lazy/piecewise packing.  
- New evaluation splits or dataset samplers.  
- UI/visualization beyond the metrics already logged.

## Approach (high level)
- Add a light collator shim that, when packing is disabled, attaches per-sample dataset labels and single-item segment lengths to each batch before the trainer sees it.  
- Reuse `GroupedMetricsMixin` for logging per-dataset loss/accuracy without invoking `GroupedPackingDataset`.  
- Default behavior unchanged when fusion provides only one dataset or telemetry is not desired.

## Impact/Risks
- Minimal runtime overhead: per-batch tensor length extraction and small extra logging.  
- Metrics shape remains backward compatible; no checkpoint format changes.

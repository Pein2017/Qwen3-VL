# Design — Stage-A Image-Level Distributed Sharding

## Scope
This design defines a two-mode Stage-A execution surface:
- `per_group`: group-level sharding; within-group batching only.
- `per_image`: image-level sharding; per-image intermediates; rank-0 merge into group-level JSONL.

## Data Model

### Group identity
- Stage-A discovery allows duplicate `group_id` values across label folders.
- Stage-B ingestion uses `uid=f\"{group_id}::{label}\"` as a stable unique ticket key (`src/stage_b/ingest/stage_a.py`).
- Stage-A merge should use an internal identifier to avoid collisions:
  - primary: `group_seq` (index in the post-sampling `groups` list)
  - optional debug: `group_uid = group_id::label`

### Image job
Minimal fields required for lossless reconstruction:
- `group_seq`: index into the sampled `groups` list (0-based).
- `image_index`: 1-based index into `GroupInfo.paths` (natural-sorted).
- `path`: filesystem path to the image.

## Data Flow (per_image)

1) Discover groups deterministically.
2) Apply group-level sampling (rank 0) and broadcast selected groups.
3) Flatten groups into deterministic job list:
   - iterate groups in list order, then images in `GroupInfo.paths` order.
4) Shard jobs across ranks:
   - default: round-robin `jobs[rank::world_size]` for load balancing and determinism.
5) Per rank:
   - process `my_jobs` in batches of size `batch_size`
   - write per-image results to a per-rank intermediate JSONL file
6) Barrier.
7) Rank 0 merge:
   - read all per-rank intermediate JSONLs
   - populate `accums[group_seq][image_index] = summary`
   - validate:
     - each filled slot must be within `[1..N]`
     - duplicates are errors
     - missing slots -> group failure
   - write final group JSONL in deterministic group order
   - delete intermediates

## Output Ordering
- Final `{mission}_stage_a.jsonl` record order is NOT a correctness criterion and MAY vary by sharding mode.
- Within each group record, `images` and `per_image` are emitted in the natural-sorted filename order recorded in each `GroupInfo`.

## Failure Semantics
- Per-image failures (decode/infer/empty summary) are recorded as image failures.
- A group is considered failed if any image in the group is missing or failed.
- Failed groups do not emit partial records.
- Merge continues to completion for other groups; failures are logged.

## Memory Bounds
- Per-rank inference keeps at most `batch_size` images in memory at once.
- Rank-0 merge stores only summaries and small metadata; if group counts become large, consider streaming merge with per-group temp buffers (not required initially).

## Notes on “Cross-Group Instability”
Existing within-group vs cross-group outputs show identical schema and coverage but substantial per-image summary text drift. This is consistent with stochastic decoding and/or backend non-determinism. For regression comparisons, `temperature=0` is recommended.

## Critical: Left Padding for Batched Generation (Top Urgency)
Stage-A uses batched generation (not packing). Qwen3-VL is decoder-only, and VL prompts are often variable-length because vision token counts depend on `image_grid_thw` (image resolution).

Therefore, batched inference MUST use **left padding**:
- Right padding can leave `PAD` tokens at the end of shorter prompts, which can destabilize generation in mixed-length batches and manifest as unrelated English prefixes (e.g., `CoreApplication...`).
- `transformers/models/qwen3_vl/processing_qwen3_vl.py` inherits `tokenizer.padding_side` and does not enforce a safe padding side.
- Callers MUST set `tokenizer.padding_side="left"` (and typically `truncation_side="left"`) before calling `processor(..., padding=True)`.

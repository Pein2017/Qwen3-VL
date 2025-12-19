# Add Stage-A Image-Level Distributed Sharding (Per-Image Mode)

## Context
Stage-A (`python -m src.stage_a.cli`, launched via `scripts/stage_a.sh`) performs per-image summarization and writes one JSONL record per group ticket. In distributed runs (`torchrun`), Stage-A currently shards work at the **group** level.

The repository recently introduced a cross-group batching mode, but empirical outputs indicate large text drift compared with within-group processing even when schema/coverage are correct.

## Goals
- Provide an **opt-in** distributed execution mode that shards work at the **image** level to improve load balancing and rank-local batch utilization.
- Reduce surface area by supporting only two execution modes:
  - `per_group`: group-level sharding; batching only within a group; no cross-group mixing in a rank’s forward pass.
  - `per_image`: image-level sharding across ranks; rank 0 merges per-image results into group-level JSONL records at the end.
- Preserve Stage-A output contract (group-level JSONL; strict `per_image` alignment and coverage; deterministic `images` order).
- Preserve existing group sampling semantics (`pass_group_number`, `fail_group_number`, `sample_seed`) by sampling at the group level before image job flattening.
- Preserve fail-fast visibility with clear error reporting; missing images in a group are treated as a group failure (no partial record) without blocking other groups.
- Do not guarantee identical generated text across sharding modes; correctness is defined by group selection + alignment/coverage.
- **Top urgency (stability)**: Stage-A uses batched generation (not packing). For Qwen3-VL decoder-only models with variable-length VL prompts, callers MUST use **left padding** (`tokenizer.padding_side="left"`) for batched inference; right padding can produce unstable/garbage prefixes in mixed-length batches.

## Non-Goals
- Bitwise-identical generated text when decoding is stochastic (e.g., `temperature>0`), or when backend kernels are non-deterministic.
- Multi-node (multi-host) support.
- Automatic retries, work-stealing, or mid-run rescheduling when a rank crashes (single-node `torchrun --max_restarts=0` is assumed).

## Proposed Behavior (User Visible)
- Add a new CLI flag:
  - `--sharding_mode per_group` (default): current behavior (group-level sharding).
  - `--sharding_mode per_image`: new behavior (image-level sharding + rank-0 merge).
- Remove (or deprecate) Stage-A cross-group batching as a first-class mode; avoid cross-group mixing in `per_group`.
  - Backward compatibility is explicitly not supported (legacy `--batching_mode` is removed rather than aliased).

### `per_group` mode
- Distributed sharding: groups are assigned to ranks (implementation detail: can remain `groups[rank::world_size]`).
- Batching: only within a group (no cross-group image batches).
- I/O: each rank writes `{mission}_stage_a.rank{rank}.jsonl`; rank 0 merges into `{mission}_stage_a.jsonl` (output ordering is not a correctness criterion).

### `per_image` mode
- After group discovery and group-level sampling, Stage-A flattens groups into deterministic image jobs.
- Distributed sharding: image jobs are assigned to ranks (recommended: round-robin `jobs[rank::world_size]`).
- Each rank processes its image jobs in batches of size `batch_size` (bounded in-flight images per rank).
- Each rank writes an intermediate per-image JSONL file (e.g., `{mission}_stage_a.images.rank{rank}.jsonl`) containing `(group_seq, image_index) -> summary` (plus error payloads).
- Rank 0 merges all per-image files into the canonical `{mission}_stage_a.jsonl`:
  - output ordering MAY vary
  - `images` is emitted from the original `GroupInfo.paths` ordering
  - `per_image` is reconstructed as `image_{1..N}`
  - missing/failed images cause the group to be marked failed (no record), and merge continues
  - intermediate per-image files are deleted by default and optionally retained for debugging

## Observed Issue: Cross-Group Instability
Comparison of existing outputs shows that the group set and coverage are identical between within-group and cross-group runs, but many per-image summary strings differ substantially. This suggests the instability is primarily **generation drift**, not schema corruption.

One concrete root cause observed in practice: **right padding** during batched generation for decoder-only Qwen3-VL can destabilize decoding when samples have different prompt lengths (common in VL because `image_grid_thw` depends on image resolution). This can manifest as unrelated English prefixes (e.g., `CoreApplication...`). The mitigation is to force **left padding** before `processor(..., padding=True)`.

Implication: image-level sharding inherently increases cross-group batching frequency; therefore, stability expectations must be set explicitly (e.g., recommend `temperature=0` for regression comparisons).

## Risks / Mitigations
- **Output drift under stochastic decoding**: document explicitly; recommend `temperature=0` for comparability.
- **Merge complexity**: implement strict collision/coverage checks; treat inconsistencies as group failures (or optionally run failure in a stricter mode).
- **I/O volume**: intermediate per-image JSONL increases line count; mitigate via small payloads and deleting intermediates after merge.
- **Partial coverage**: missing `(group_seq, image_index)` results become group failures; merge continues; failures are logged with group identifiers.

## Validation Plan
- Unit/utility validation on intermediate merge logic (duplicate keys, missing keys, out-of-range indices).
- Smoke runs:
  - `sharding_mode=per_group` single-GPU and multi-GPU parity on schema and group counts.
  - `sharding_mode=per_image` multi-GPU: verify deterministic group order, strict coverage, and that group sampling counts match requested targets.
- Regression comparison using `temperature=0` to minimize drift.

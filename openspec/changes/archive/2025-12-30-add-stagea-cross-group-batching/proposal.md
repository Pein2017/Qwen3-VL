# Add Stage-A Cross-Group Image Batching

## Context
Stage-A (`python -m src.stage_a.cli`, launched by `scripts/stage_a.sh`) performs per-image summarization and writes one JSONL record per group ticket.

Current execution is **group-local batching**:
- each group is processed end-to-end (load all images in the group → run inference)
- batching (`--batch_size`) applies only to images *within* that group

In real data, many groups contain only a few images. This yields low GPU utilization because `batch_size` is rarely filled.

## Goals
- Provide an **opt-in** execution mode that can batch images **across groups** to better fill GPU batches and improve throughput.
- Preserve **output structure** (no schema change):
  - one JSONL record per group with keys: `group_id`, `mission`, `label`, `images`, `per_image`
  - `per_image` keys remain `image_{i}` with strict coverage and deterministic ordering
  - group failures do not emit partial records and do not stop other groups
- Keep distributed behavior correct:
  - under `torchrun`, each rank runs independently over its shard of groups
  - no cross-rank mixing of work; each rank batches only within its own shard

## Non-Goals
- Bitwise-identical generated text when decoding is stochastic (e.g., `temperature>0`).
- Changing Stage-A output schema/fields.
- Multi-node (multi-host) support.

## Proposed Behavior (User Visible)
- Add a CLI option to choose batching strategy, defaulting to current behavior:
  - `--batching_mode per_group` (default): current group-local batching
  - `--batching_mode cross_group`: accumulate images from successive groups to form full batches when possible
- In `cross_group` mode:
  - batches may contain images from multiple groups
  - the system re-aggregates results so each group’s JSONL record is unchanged in structure
  - output order SHOULD remain stable with respect to the discovered group order per rank (buffer + flush in order; failures count as "done" for ordering)

## Risks / Mitigations
- **Different text outputs** under stochastic decoding: document explicitly; recommend `temperature=0.0` for deterministic outputs.
- **Streaming/error semantics drift**: preserve write-once-per-group behavior and isolate failures (no stuck buffers; failures are treated as completed for ordering).
- **Memory pressure**: in cross-group mode, keep at most `batch_size` images in flight per rank and drop tensors/images immediately after each batch decode (no full-group or multi-group preloading beyond the current in-flight batch).

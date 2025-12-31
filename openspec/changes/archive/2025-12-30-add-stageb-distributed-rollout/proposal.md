# Add Stage-B Distributed Ticket-Parallel Rollout

## Context
Stage-B (`src/stage_b/runner.py`) runs a training-free loop:
rollout (batched) → selection → reflection (sequential, updates shared mission guidance) → next epoch.

Rollout is the dominant cost and we have 8 GPUs available. Reflection MUST remain sequential because it mutates mission guidance (`guidance.json`) that affects later rollouts.

## Goals
- Support **single-GPU** and **multi-GPU** execution from the same entry points (`scripts/stage_b.sh` → `python -m src.stage_b.runner`).
- Implement **ticket-parallel** rollout across GPUs (data-parallel over group tickets) to maximize GPU utilization.
- Preserve **global** semantics for:
  - `runner.per_rank_rollout_batch_size` (per-rank batch size; global effective batch = per_rank × WORLD_SIZE)
  - `reflection.batch_size` (reflection trigger threshold)
- Ensure all ranks use **the same guidance snapshot per global rollout batch** (no intra-batch guidance drift).
- Ensure only **rank 0** writes artifacts to disk.
- Keep backward compatibility: existing single-GPU workflows and configs continue to work without changes.
- Auto-enable distributed mode based on the `gpus` device list passed to `scripts/stage_b.sh` (single GPU → single-process; multiple GPUs → `torchrun`).

## Non-Goals
- Bitwise-identical sampling outputs across different `WORLD_SIZE` values.
- Model-parallel sharding of a single model instance across GPUs (each rank loads the full model on its GPU).
- Multi-node (multi-host) launch support in this change (single-node `torchrun` only).

## Proposed Behavior (User Visible)
- `scripts/stage_b.sh` will launch:
  - single-process mode when `gpus` contains one device (status quo)
  - `torchrun --nproc_per_node=<N>` when `gpus` contains multiple devices
- In multi-GPU mode:
  - `runner.per_rank_rollout_batch_size` is interpreted as **per-rank (per-device) batch size**; global effective batch = `per_rank_rollout_batch_size × WORLD_SIZE`.
  - Each rollout batch is sharded across ranks; each rank runs rollout for up to `per_rank_rollout_batch_size` tickets per step.
  - Rank 0 gathers candidates from all ranks, runs selection and reflection sequentially, updates guidance, and writes artifacts.
  - Guidance is broadcast once per global rollout batch so every shard uses the same guidance step.

## Risks / Mitigations
- **Deadlocks on early stop / exceptions**: rank 0 will broadcast epoch-level continue/stop decisions so all ranks take collectives in the same order.
- **Accidental model-parallelism** with `device_map="auto"`: in distributed mode, the runner will force per-rank single-GPU placement (local-rank device) to avoid cross-process sharding.
- **Communication overhead**: only parsed rollout outputs are gathered (rank0 only), keeping reflection sequential and minimizing additional synchronization.
- **Partial failures**: the system will fail fast if any rank errors during rollout; `torchrun` will tear down the job to avoid silent divergence.

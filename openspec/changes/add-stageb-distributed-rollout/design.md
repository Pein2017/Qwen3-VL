# Design: Stage-B Distributed Ticket-Parallel Rollout

## Summary
Add an optional multi-process distributed mode (single-node `torchrun`) that shards **tickets** across ranks for rollout, while keeping **selection + reflection** on rank 0.

Key invariants:
- `runner.rollout_batch_size` is **global** (effective) per rollout step.
- `reflection.batch_size` is **global** and evaluated only on rank 0 after gathering candidates.
- Guidance is **consistent within a global rollout batch** (broadcast once per batch).
- Only rank 0 writes artifacts; non-zero ranks are rollout workers.
- Single-process behavior remains the default when not launched under `torchrun`.

## Enablement & Launch
- Distributed execution is enabled automatically when Stage-B is launched with multiple GPUs via `scripts/stage_b.sh` (comma-separated `gpus` list), which uses single-node `torchrun --nproc_per_node=<N>`.
- Single-GPU execution remains `python -m src.stage_b.runner ...` (status quo).
- Multi-node is out of scope; the launcher and runner assume `--nnodes=1`.

## Execution Flow

### Initialization
1. Detect distributed mode:
   - `WORLD_SIZE>1` and `torch.distributed` initialized via `env://`.
2. Rank 0:
   - cleans `{output.root}/{run_name}` and resets mission artifacts
   - seeds mission guidance into `{run_dir}/{mission}/guidance.json`
3. All ranks:
   - ingest Stage-A JSONLs (read-only) to build identical `mission_tickets`
   - load the model/tokenizer on their local GPU

### Per Epoch
Rank 0 computes `ordered_indices` (shuffle by `seed+epoch`) and broadcasts it so all ranks iterate the same ticket order.

### Per Global Rollout Batch
For each global batch of size `B = runner.rollout_batch_size`:
1. Rank 0 loads the current guidance snapshot (post-reflection) and broadcasts it.
2. Every rank computes its shard deterministically:
   - `counts[r] = B//W + (r < (B % W))`
   - `offsets = cumsum(counts)`
   - shard = `batch[offsets[rank] : offsets[rank] + counts[rank]]`
3. Each rank runs rollout for its shard using `RolloutSampler.generate_for_batch(local_tickets, guidance_map)`.
4. Rank 0 gathers all shard outputs (`group_id -> ParsedTrajectory[]`) and merges them.
5. Rank 0 executes the existing per-ticket selection + reflection logic, appending trajectories/selections and updating guidance when reflection applies.

### Synchronization Points
Per global batch:
- `broadcast(guidance)` → ensures identical guidance step for all shards
- `gather(rollout_outputs)` → rank 0 merges candidates before selection/reflection

Per epoch boundary:
- `broadcast(ordered_indices)` → consistent ticket order
- `broadcast(continue_flag)` → avoid deadlocks when rank 0 early-stops

## Device Placement
In distributed mode, each rank MUST keep the full model on its local GPU (ticket-parallel data-parallelism).

To avoid accidental model-parallelism when configs use `device_map="auto"`:
- The runner overrides model placement in distributed mode to a local-rank single-GPU mapping.

## Failure Policy (Fail Fast)
- If any rank raises during rollout or communication, the run MUST abort (fail fast).
- This relies on `torchrun`'s default behavior to tear down all ranks on failure and prevents silent divergence.

## Determinism
This change does NOT require rollouts to match single-GPU sampling exactly across different `WORLD_SIZE` values.
Correctness is defined by:
- consistent guidance per global batch,
- sequential reflection updates on rank 0,
- and stable artifact contracts written by rank 0.

For users needing reproducibility, configs SHOULD provide explicit per-decode seeds.

## Artifact Writing
- Rank 0 is the only writer for:
  - `trajectories.jsonl`, `selections.jsonl`, `reflection.jsonl`, `metrics_epoch.jsonl`
  - mission guidance snapshots and `guidance.json`
  - manual review / failure queues
- Non-zero ranks MUST NOT write into `{output.root}/{run_name}`.

## CLI / Entry Points
`scripts/stage_b.sh` becomes a unified launcher:
- 1 GPU → `python -m src.stage_b.runner ...`
- N GPUs → `torchrun --nproc_per_node=N -m src.stage_b.runner ...`

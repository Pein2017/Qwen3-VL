# stage-b-training-free — Distributed Ticket-Parallel Rollout

## ADDED Requirements

### Requirement: Stage-B SHALL support single-node multi-process distributed rollout with sequential reflection on rank 0.
Stage-B SHALL support a `torchrun`-launched mode where multiple ranks cooperate to accelerate rollout, while preserving the sequential guidance update constraint.
#### Scenario: Operators run Stage-B on 8 GPUs
- **WHEN** operators launch Stage-B with multiple GPUs via `scripts/stage_b.sh` (comma-separated `gpus` list)
- **THEN** the launcher SHALL use single-node `torchrun` (`--nnodes=1`) with `WORLD_SIZE>1`
- **AND** the system SHALL shard each global rollout batch across ranks (ticket-parallel rollout)
- **AND** rank 0 SHALL gather all rollout candidates and run selection + reflection sequentially
- **AND** rank 0 SHALL be the only process that writes artifacts to `{output.root}/{run_name}/...`
- **AND** non-zero ranks SHALL NOT write any Stage-B artifacts to disk
- **AND** the run SHALL fail fast if any rank errors during rollout or synchronization

### Requirement: Stage-B distributed mode SHALL replicate the full model per rank on the local GPU.
Distributed rollout is data-parallel over tickets; Stage-B MUST avoid accidental model-parallelism across ranks.
#### Scenario: Config uses `device_map="auto"` under `torchrun`
- **WHEN** Stage-B is launched under `torchrun` with `WORLD_SIZE>1`
- **AND** the config sets `model.device_map="auto"`
- **THEN** Stage-B SHALL force per-rank single-GPU model placement on the local rank device
- **AND** each rank SHALL load the full model weights on its local GPU (no cross-rank sharding)

### Requirement: Stage-B distributed mode SHALL treat `runner.per_rank_rollout_batch_size` as a per-rank (per-device) batch size.
In distributed mode, `runner.per_rank_rollout_batch_size` SHALL represent the number of tickets processed per rollout step on each rank, and the global effective batch size SHALL be `per_rank_rollout_batch_size × WORLD_SIZE`.
#### Scenario: Per-rank batch size with varying GPU count
- **WHEN** `runner.per_rank_rollout_batch_size=P` and the run uses `WORLD_SIZE=W`
- **THEN** the global effective batch size SHALL be `P × W` tickets per rollout step
- **AND** each global batch SHALL be deterministically sharded so each rank receives up to `P` tickets (or fewer for the final batch at end-of-epoch when remaining tickets < P × W)
- **AND** the union of all rank shards SHALL contain exactly `P × W` tickets (or fewer only for the final batch at end-of-epoch when remaining tickets < P × W)

### Requirement: Stage-B distributed mode SHALL enforce a single guidance snapshot per global rollout batch.
Guidance updates from reflection SHALL affect only subsequent global rollout batches; no batch may see multiple guidance steps.
#### Scenario: Reflection applies guidance updates during an epoch
- **WHEN** a reflection cycle applies guidance updates on rank 0
- **THEN** the updated guidance SHALL only be used for the next global rollout batch
- **AND** all ranks SHALL use the same guidance step within a given global rollout batch (no intra-batch version drift)

### Requirement: Stage-B distributed mode SHALL keep `reflection.batch_size` global on rank 0.
Reflection triggering SHALL remain a global decision based on the number of processed tickets, not on per-rank shard sizes.
#### Scenario: Reflection trigger threshold remains unchanged with more GPUs
- **WHEN** `reflection.batch_size=N` and Stage-B runs with `WORLD_SIZE>1`
- **THEN** rank 0 SHALL trigger reflection only after it has accumulated `N` records globally (after gathering rollouts)
- **AND** reflection SHALL NOT be independently triggered on non-zero ranks

## MODIFIED Requirements

### Requirement: Stage-B tooling SHALL expose a fail-fast `run_all()` orchestration entry point.
The orchestration entry point SHALL support both single-process and distributed execution while preserving artifact contracts.
#### Scenario: Single-process execution remains supported
- **WHEN** Stage-B runs without `torchrun` (`WORLD_SIZE` unset or equals 1)
- **THEN** `run_all()` SHALL behave as a single-process pipeline as before
- **AND** it SHALL write all artifacts under `{output.root}/{run_name}/{mission}/`

#### Scenario: Distributed execution is enabled
- **WHEN** Stage-B runs under `torchrun` (`WORLD_SIZE>1`)
- **THEN** `run_all()` SHALL coordinate distributed rollout, gather on rank 0, and keep reflection sequential on rank 0

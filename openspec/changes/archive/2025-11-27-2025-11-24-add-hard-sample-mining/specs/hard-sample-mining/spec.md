## ADDED Requirements

### Requirement: Configurable Hard-Sample Mining (Dynamic Mode)
The system SHALL expose a YAML switch `custom.hard_sample_mining` that, when enabled, activates a **dynamic batch-level** hard-sample mining strategy; when disabled or absent, training behavior is unchanged.

#### Scenario: Disabled by default
- **WHEN** `custom.hard_sample_mining` is omitted or `enabled=false`
- **THEN** no mining callbacks or sampler changes are applied, and training matches current behavior.

#### Scenario: Dynamic mode enabled
- **WHEN** `custom.hard_sample_mining.enabled=true`
- **THEN** the dynamic mining pipeline (token_acc tracking, hard-pool update, and fixed-ratio batch mixing) is active for the target dataset only.

### Requirement: Per-sample token_acc Tracking with EMA
The trainer SHALL record per-sample **token_acc** (masked by `labels==-100`) with stable identifiers `(dataset, base_idx, sample_id)`, updating an EMA difficulty score; if a sample has <3 observations in an epoch, the mean of available observations SHALL be used.

#### Scenario: Rank-safe collection
- **WHEN** training under DDP/DeepSpeed-ZeRO2
- **THEN** per-sample losses are gathered in a rank-safe manner (e.g., aggregation on rank0) without duplicate updates, and identifiers remain stable across augmented views.

### Requirement: Hard-Pool Maintenance (Target-only)
At each epoch end, the system SHALL recompute a hard pool from the **target dataset only**, selecting the highest-loss samples by a configurable fraction or count, while source datasets remain untouched.

#### Scenario: Fractional pool
- **WHEN** `hard_pool_frac=0.30` (default) and no explicit `hard_pool_k` is set
- **THEN** the top 30% (rounded up) of target samples by EMA difficulty form the hard pool for the next epoch; ties break deterministically by `sample_id`.

#### Scenario: Fixed top-K override
- **WHEN** `hard_pool_k` is provided
- **THEN** exactly `hard_pool_k` target samples with highest difficulty form the hard pool.

### Requirement: Fixed-Ratio Batch Mixing after Warmup
During training in mining mode, the sampler SHALL mix hard and regular target samples per batch using a fixed ratio implied by `hard_pool_frac`, activated only after a configurable progress threshold (`activate_after_pct`).

#### Scenario: Warmup then fixed ratio
- **WHEN** `activate_after_pct=0.70`
- **THEN** mining is inactive before 70% of epochs; after that point, batches/schedules use the fixed hard/regular mixing ratio implied by `hard_pool_frac` (e.g., 30% hard, 70% regular) for the target dataset, sources unchanged.

#### Scenario: Hard pool smaller than k_hard
- **WHEN** `k_hard` exceeds the hard-pool size
- **THEN** sampling SHALL wrap with replacement to meet batch size and log the shortfall.

### Requirement: Source-dataset Isolation
Mining SHALL NOT upweight, downweight, or select hard samples from source datasets in fusion; only the target dataset participates in hard/regular mixing, while source quotas/ratios remain unchanged and configurable via `source_ratio`.

#### Scenario: Fusion training
- **WHEN** fusion sources (e.g., lvis/coco/objects365/flickr3k) are present
- **THEN** their sampling ratios are preserved exactly as configured, independent of the hard/regular split on the target dataset.

### Requirement: Distributed Compatibility and Determinism
The mining pipeline (token_acc tracking, hard-pool selection, and batch mixing) SHALL operate correctly and deterministically under distributed training, including DeepSpeed ZeRO-2.

#### Scenario: Epoch-seeded sampling
- **WHEN** running multi-rank training
- **THEN** rank0 broadcasts the hard-pool and schedule; the sampler uses an epoch-seeded RNG so that all ranks draw consistent hard/regular splits per batch.

### Requirement: Telemetry
The system SHALL surface mining activity via standard trainer logging (e.g., TensorBoard/W&B scalars) without requiring JSON dumps of sample lists.

#### Scenario: Per-epoch mining logs
- **WHEN** an epoch ends in mining mode
- **THEN** logs include at least: `hard_pool_size`, `hard_hit_rate`, `hard_pool_coverage`, `train_acc_hard_mean/p90`, `train_acc_regular_mean/p90`, and counts of hard/regular samples seen, under a configurable prefix (default `hsm/`).

## MODIFIED Requirements

### Requirement: Configurable Hard-Sample Mining (Dynamic Mode)
The system SHALL reject `custom.hard_sample_mining` configurations and run standard SFT without mining hooks, samplers, or metadata.

#### Scenario: YAML includes hard_sample_mining
- **WHEN** a training config contains `custom.hard_sample_mining`
- **THEN** validation FAILS with a clear error stating hard-sample mining was removed and to delete the block to proceed.

#### Scenario: Import mining callback
- **WHEN** code attempts to import mining-specific callbacks or datasets
- **THEN** the import fails, indicating the feature was removed.

#### Scenario: Run packed training
- **WHEN** training runs with packing enabled and no `custom.hard_sample_mining`
- **THEN** the trainer uses the default dataloader/shuffle path with no mining metadata or hooks.

## REMOVED Requirements

### Requirement: Per-sample token_acc Tracking with EMA
#### Scenario: Attempt to access token_acc history
- **WHEN** code attempts to read token-level accuracy buffers or EMA difficulty from the trainer or dataset
- **THEN** the attributes are absent, and mining metadata is not produced in logs or checkpoints.

### Requirement: Hard-Pool Maintenance (Target-only)
#### Scenario: End-of-epoch mining step
- **WHEN** an epoch completes
- **THEN** no hard-pool recomputation or persistence occurs, and no hard/regular lists are emitted.

### Requirement: Fixed-Ratio Batch Mixing after Warmup
#### Scenario: Sampler configuration
- **WHEN** a run reaches the previous `activate_after_pct` threshold
- **THEN** batching behavior remains unchanged (no hard/regular ratio), and any legacy mining sampler classes are unavailable.

### Requirement: Source-dataset Isolation
#### Scenario: Fusion training
- **WHEN** training with fusion sources (e.g., lvis/coco/objects365/flickr3k)
- **THEN** sampling ratios stay exactly as configured, with no hard/regular adjustments or mining metadata.

### Requirement: Distributed Compatibility and Determinism
#### Scenario: Multi-rank run
- **WHEN** running with multiple ranks
- **THEN** no mining-specific broadcasts or RNG seeding occurs because mining is disabled.

### Requirement: Telemetry
#### Scenario: Inspect logs
- **WHEN** inspecting TensorBoard/W&B or JSONL logs
- **THEN** fields such as `hsm/hard_pool_size`, `hsm/hard_hit_rate`, and `train_acc_hard_mean` are not present.

## MODIFIED Requirements

### Requirement: Configurable Hard-Sample Mining (Dynamic Mode)
The system SHALL reject `custom.hard_sample_mining` configurations and run standard SFT without mining hooks, samplers, or metadata.

#### Scenario: YAML includes hard_sample_mining
- **WHEN** a training config contains `custom.hard_sample_mining`
- **THEN** validation FAILS with a clear error stating hard-sample mining was removed and to delete the block to proceed.

#### Scenario: Import mining callback
- **WHEN** code attempts to import mining-specific callbacks or datasets
- **THEN** the import fails, indicating the feature was removed.

#### Scenario: Run training without mining
- **WHEN** training runs with the default padded batching path (packing removed) and no `custom.hard_sample_mining`
- **THEN** the trainer uses the standard dataloader/shuffle path with no mining metadata or hooks.

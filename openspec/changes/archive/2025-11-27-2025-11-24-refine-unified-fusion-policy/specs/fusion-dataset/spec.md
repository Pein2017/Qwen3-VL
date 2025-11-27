# fusion-dataset Specification (delta)

## ADDED Requirements

### Requirement: Single-template unified fusion with prompt priority
The fusion loader SHALL use a single template instance and resolve prompts by priority `default < domain < dataset-specific` for both user/system prompts per source.

#### Scenario: Dataset-specific prompt override
- **WHEN** a dataset provides its own prompt override
- **THEN** that override is used for user/system prompts while keeping the shared template (no cloning)
- **AND** if absent, the domain prompt (e.g., target=BBU Chinese, source=aux English) is used; if still absent, the default applies
- **AND** the template system prompt is restored after encoding to avoid cross-sample contamination.

### Requirement: Per-source augmentation and curriculum policy
The fusion loader SHALL honor per-source policies for augmentation and curriculum: sources default to clean unless enabled; targets inherit the global augmentation/curriculum configuration.

#### Scenario: Clean auxiliary sample
- **WHEN** a source-domain sample is fetched and its policy marks augmentation disabled
- **THEN** no augmentation/curriculum preprocessors run, even if the target policy enables them
- **AND** a target-domain sample in the same epoch still receives the configured augmentation/curriculum.

### Requirement: Per-source object cap
The fusion loader SHALL support per-source object caps to control sequence length; no geometry fallback is required (assumed pre-filtered JSONL).

#### Scenario: Source object cap applied
- **WHEN** a source policy sets `max_objects_per_image=5`
- **THEN** the loader enforces the cap deterministically (seeded) before builder/template encoding
- **AND** targets may opt in; fallback to no-cap when unset.

### Requirement: Deterministic per-epoch resampling and shuffling
The fusion loader SHALL rebuild its schedule each epoch: consume all target samples once, sample each source with replacement by `round(ratio * N_target)`, then shuffle with a deterministic seed.

#### Scenario: Epoch boundary resampling
- **WHEN** a new epoch starts
- **THEN** the loader refreshes the source draws with replacement according to ratios, keeps full target coverage, shuffles the combined index deterministically, and exposes counts per source.

### Requirement: No online smart-resize guard
The fusion loader SHALL NOT perform any online smart-resize; inputs are assumed pre-scaled, and resizing occurs only via augmentation ops when configured.

#### Scenario: Oversized input
- **WHEN** a sample exceeds expected dimensions
- **THEN** the fusion loader raises/alerts rather than silently resizing; any resizing must be via explicit augmentation ops.

### Requirement: Telemetry and validation hooks
The fusion loader SHALL emit per-sample debug info and per-epoch aggregates covering dataset name, prompt selection, augmentation enabled/disabled, object-cap hits, smart-resize hits, and input length to support OOD/OOM diagnostics.

#### Scenario: Telemetry capture
- **WHEN** a batch is drawn from mixed sources
- **THEN** the loader records the source name, whether augmentation ran, whether object cap or smart-resize was applied, and the input length for that sample, and exposes aggregate counters per epoch for monitoring.

### Requirement: Optional per-source evaluation splits
The fusion loader SHALL support optional source `val_jsonl` splits; evaluation defaults to target-only and uses fixed (non-shuffled) splits.

#### Scenario: Source eval provided
- **WHEN** a source supplies `val_jsonl`
- **THEN** the eval dataloader can include that split without shuffling, while target evaluation remains the default when source eval is absent.

### Requirement: Unified-only fusion loader
The fusion stack SHALL run exclusively through the unified fusion loader (single shared template); the legacy MultiSourceFusionDataset path is removed.

#### Scenario: No legacy fallback
- **WHEN** a training config declares `custom.fusion_config`
- **THEN** the trainer instantiates the unified fusion loader without a legacy flag or alternate code path
- **AND** attempts to reference the legacy loader are rejected during config parsing.

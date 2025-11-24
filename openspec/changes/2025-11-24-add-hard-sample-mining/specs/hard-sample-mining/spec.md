## ADDED Requirements

### Requirement: Configurable Hard-Sample Mining Stage
The system SHALL allow enabling hard-sample mining via YAML (`custom.hard_sample_mining`) with explicit defaults that leave training behavior unchanged when disabled.

#### Scenario: Disabled by default
- **WHEN** `custom.hard_sample_mining` is absent or `enabled=false`
- **THEN** training runs exactly as today (no extra callbacks, no schedule changes)

#### Scenario: Start after convergence
- **WHEN** `custom.hard_sample_mining.start_epoch` is reached AND loss/metric plateau is detected within the configured `plateau_delta`
- **THEN** the mining stage activates and produces a hard-sample plan for the next epoch

### Requirement: Per-sample Loss Tracking with Augmentation
The trainer SHALL collect per-sample **loss** (masked by `labels==-100`) on the training path (with augmentation applied) and associate it with stable sample identifiers (dataset name + base index).

#### Scenario: Record losses per microbatch
- **WHEN** a training batch finishes
- **THEN** the system records a per-sample loss scalar (after label masking) plus `sample_id`, `dataset`, `base_idx`, `epoch`, and local step; collection works under DDP/DeepSpeed-ZeRO2 by aggregating only on rank 0 while keeping global logging unchanged

### Requirement: Hard-Sample Selection
The system SHALL support selecting hard samples by top-K, percentile, or absolute **loss** threshold using aggregated per-sample losses (mean or EMA).

#### Scenario: Top-K selection
- **WHEN** `selector.mode=top_k` with `k=500`
- **THEN** the 500 samples with highest aggregated loss are marked as hard, ties broken deterministically by `sample_id`

#### Scenario: Percentile selection
- **WHEN** `selector.mode=percentile` with `value=0.1`
- **THEN** all samples whose loss is in the worst 10% are marked as hard

### Requirement: Dataset Reweighting Without Length Change
The dataset SHALL increase the frequency of mined samples in subsequent epochs while keeping the epoch length constant and preserving augmentation behavior.

#### Scenario: Weighted permutation
- **WHEN** a hard-sample plan assigns weight 3 to certain samples
- **THEN** the epoch permutation is rebuilt using weighted sampling so those samples appear ~3× as often, but the total number of samples per epoch remains unchanged

#### Scenario: Fusion schedule compatibility
- **WHEN** using `FusionCaptionDataset`
- **THEN** the per-source quotas are preserved, hard-sample weights are applied **only to the fusion target pool**, and source pools (e.g., lvis/coco/objects365/flickr3k) remain untouched while the target:source ratio is unchanged

### Requirement: Target-only Mining
Hard-sample mining SHALL apply exclusively to the fusion target dataset; auxiliary/source datasets MUST NOT be upweighted, downweighted, or selected as hard samples.

#### Scenario: Skip non-target datasets
- **WHEN** mining runs in a fusion setting with sources present
- **THEN** only samples whose `dataset` equals `fusion_config.target.name` are eligible for hard selection or weighting; all source samples keep their original sampling policy

### Requirement: Distributed Compatibility
The mining pipeline (loss collection, selection, and schedule update) SHALL function under distributed training setups, including DeepSpeed ZeRO-2.

#### Scenario: Rank-safe aggregation
- **WHEN** training uses DeepSpeed ZeRO-2 with multiple ranks
- **THEN** per-sample loss statistics are gathered or reduced in a rank-safe manner (e.g., update tracker on rank 0 only), and no duplicated updates occur across ranks

### Requirement: Telemetry and Persistence
The system SHALL surface mining activity and optional dumps for inspection.

#### Scenario: Logging
- **WHEN** mining triggers or updates a plan
- **THEN** logs include number of hard samples, selection mode, and weight range with prefix `hsm/*`

#### Scenario: JSON dump
- **WHEN** `custom.hard_sample_mining.dump=true`
- **THEN** a JSON file under `output_dir/hard_samples_epoch_{n}.json` records `sample_id`, `dataset`, `base_idx`, and aggregated loss for that epoch

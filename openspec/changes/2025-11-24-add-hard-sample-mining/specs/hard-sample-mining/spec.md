## ADDED Requirements

### Requirement: Configurable Hard-Sample Mining Stage
The system SHALL allow enabling hard-sample mining via YAML (`custom.hard_sample_mining`) with explicit defaults that leave training behavior unchanged when disabled.

#### Scenario: Disabled by default
- **WHEN** `custom.hard_sample_mining` is absent or `enabled=false`
- **THEN** training runs exactly as today (no extra callbacks, no schedule changes)

#### Scenario: Start after convergence
- **WHEN** `custom.hard_sample_mining.start_epoch` is reached AND an **eval/loss** plateau is detected within the configured `plateau_delta`
- **THEN** the mining stage activates and produces a hard-sample plan for the next epoch

### Requirement: Per-sample Loss Tracking with Augmentation
The trainer SHALL collect per-sample **loss** (masked by `labels==-100`) on the training path (with augmentation applied) and associate it with stable sample identifiers (dataset name + base index).

#### Scenario: Record losses per microbatch
- **WHEN** a training batch finishes
- **THEN** the system records a per-sample loss scalar (after label masking) plus `sample_id`, `dataset`, `base_idx`, `epoch`, and local step; collection works under DDP/DeepSpeed-ZeRO2 by aggregating only on rank 0 while keeping global logging unchanged

### Requirement: Hard-Sample Selection
The system SHALL select a fixed number of hard samples using aggregated per-sample **loss** (mean or EMA). When fewer than 3 observations exist for a sample, the mean of available observations SHALL be used.

#### Scenario: Top-K selection
- **WHEN** `hard_sample_size=500`
- **THEN** the 500 samples with highest aggregated loss are marked as hard, ties broken deterministically by `sample_id`

#### Scenario: Percentile selection
- **WHEN** `selector.mode=percentile` with `value=0.1`
- **THEN** all samples whose loss is in the worst 10% are marked as hard

### Requirement: Fixed-size Epoch Sampling (optional downsizing)
The dataset SHALL build each mining epoch from a fixed count of hard samples and regular samples, optionally downsizing the target epoch length.

#### Scenario: Fixed hard+regular counts
- **WHEN** `hard_sample_size=500` and `regular_sample_size=150`
- **THEN** each mining epoch draws 500 hard samples and 150 regular samples (with replacement when needed), while source pools keep their original quotas

#### Scenario: Optional target epoch size
- **WHEN** `target_epoch_size` is provided
- **THEN** the target schedule length equals `target_epoch_size`; otherwise it equals the sum of hard and regular draws (or the full target size if unset)

#### Scenario: Fusion schedule compatibility
- **WHEN** using `FusionCaptionDataset`
- **THEN** the per-source quotas are preserved, hard-sample weights are applied **only to the fusion target pool**, source pools (e.g., lvis/coco/objects365/flickr3k) remain untouched, and if a downsized target length is set it adjusts only the target portion of the schedule

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

### Requirement: Telemetry (no JSON dumps)
The system SHALL surface mining activity via logs; JSON dump of hard lists is not required.

#### Scenario: Logging
- **WHEN** mining triggers or updates a plan
- **THEN** logs include number of hard samples, selection mode, and weight range with prefix `hsm/*`

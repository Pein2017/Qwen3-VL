# summary-grpo-post-training Spec Delta

## ADDED Requirements

### Requirement: CHORD mixing toggle is GRPO-only and fails fast otherwise
The `custom.grpo_chord.enabled` toggle SHALL only be valid when `rlhf.rlhf_type == "grpo"`. If enabled when `rlhf.rlhf_type != "grpo"`, training initialization SHALL fail fast with a clear configuration error.

#### Scenario: Enabled under non-GRPO fails fast
- **GIVEN** a config with `custom.grpo_chord.enabled: true`
- **AND** `rlhf.rlhf_type` is not `grpo`
- **WHEN** training initialization begins
- **THEN** the process fails fast with a clear configuration error message

### Requirement: Optional CHORD SFT mixing provides a supervised fallback signal
Summary GRPO post-training SHALL support an optional CHORD-style mixed loss that combines GRPO loss with a supervised (SFT) loss to provide training signal when rollouts are identical.

#### Scenario: CHORD mixing disabled preserves baseline behavior
- **GIVEN** a summary GRPO config with `custom.grpo_chord.enabled: false` (or absent)
- **WHEN** training runs
- **THEN** the trainer computes GRPO loss only
- **AND** no CHORD SFT dataset is attached

#### Scenario: CHORD mixing enabled attaches expert data and mixes losses
- **GIVEN** a summary GRPO config with `custom.grpo_chord.enabled: true` and valid CHORD schedule parameters
- **WHEN** training runs
- **THEN** the trainer computes `loss = (1 - mu) * grpo_loss + mu * chord_sft_loss`
- **AND** the expert dataset is attached for CHORD SFT loss computation

### Requirement: CHORD expert dataset defaults to the GRPO training dataset
When CHORD mixing is enabled for summary GRPO post-training, the system SHALL default the CHORD expert dataset to the same fusion dataset used for GRPO training so that supervised targets match the summary output contract.

#### Scenario: Expert dataset defaults to fusion train dataset
- **GIVEN** summary GRPO training launched with a fusion dataset and CHORD mixing enabled
- **WHEN** the trainer is constructed
- **THEN** the CHORD expert dataset uses the same fusion train dataset records as the GRPO training dataset
- **AND** irrelevant samples (`metadata._fusion_source == "irrelevant_summary"`) are included without filtering
- **AND** irrelevant targets remain the single-line `无关图片` contract

### Requirement: CHORD mixing is config-toggleable and validated
The system SHALL provide a single YAML toggle to enable/disable CHORD mixing for summary GRPO post-training via `custom.grpo_chord.enabled`. When enabled, required CHORD schedule parameters SHALL be validated at startup, and the trainer SHALL receive the corresponding ms-swift CHORD arguments.

Required fields when enabled:
- `custom.grpo_chord.mu_warmup_steps` (int >= 0)
- `custom.grpo_chord.mu_decay_steps` (int >= 0)
- `custom.grpo_chord.mu_peak` (float in [0, 1])
- `custom.grpo_chord.mu_valley` (float in [0, 1])
- `custom.grpo_chord.sft_per_device_train_batch_size` (int > 0)

Optional fields:
- `custom.grpo_chord.enable_phi_function` (bool; default false)

When enabled, the trainer args SHALL be populated as:
- `chord_mu_warmup_steps = custom.grpo_chord.mu_warmup_steps`
- `chord_mu_decay_steps = custom.grpo_chord.mu_decay_steps`
- `chord_mu_peak = custom.grpo_chord.mu_peak`
- `chord_mu_valley = custom.grpo_chord.mu_valley`
- `chord_sft_per_device_train_batch_size = custom.grpo_chord.sft_per_device_train_batch_size`
- `chord_enable_phi_function = custom.grpo_chord.enable_phi_function`

#### Scenario: Missing schedule parameters fails fast
- **GIVEN** CHORD mixing is enabled but one or more required schedule fields are missing
- **WHEN** training initialization begins
- **THEN** the process fails fast with a clear configuration error message

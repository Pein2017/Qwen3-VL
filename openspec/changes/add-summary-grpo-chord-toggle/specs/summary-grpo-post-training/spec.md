# summary-grpo-post-training Spec Delta

## ADDED Requirements

### Requirement: Optional CHORD SFT mixing provides a supervised fallback signal
Summary GRPO post-training SHALL support an optional CHORD-style mixed loss that combines GRPO loss with a supervised (SFT) loss to provide training signal when rollouts are identical.

#### Scenario: CHORD mixing disabled preserves baseline behavior
- **GIVEN** a summary GRPO config with CHORD mixing disabled
- **WHEN** training runs
- **THEN** the trainer computes GRPO loss only
- **AND** no CHORD SFT dataset is attached

#### Scenario: CHORD mixing enabled attaches expert data and mixes losses
- **GIVEN** a summary GRPO config with CHORD mixing enabled and valid CHORD schedule parameters
- **WHEN** training runs
- **THEN** the trainer computes `loss = (1 - mu) * grpo_loss + mu * chord_sft_loss`
- **AND** the expert dataset is attached for CHORD SFT loss computation

### Requirement: CHORD expert dataset defaults to the GRPO training dataset
When CHORD mixing is enabled for summary GRPO post-training, the system SHALL default the CHORD expert dataset to the same fusion dataset used for GRPO training so that supervised targets match the summary output contract.

#### Scenario: Expert dataset defaults to fusion train dataset
- **GIVEN** summary GRPO training launched with a fusion dataset and CHORD mixing enabled
- **WHEN** the trainer is constructed
- **THEN** the CHORD expert dataset uses the same fusion train dataset records as the GRPO training dataset

### Requirement: CHORD mixing is config-toggleable and validated
The system SHALL provide a single YAML toggle to enable/disable CHORD mixing for summary GRPO post-training. When enabled, required CHORD schedule parameters SHALL be validated at startup.

#### Scenario: Missing schedule parameters fails fast
- **GIVEN** CHORD mixing is enabled but one or more required schedule fields are missing
- **WHEN** training initialization begins
- **THEN** the process fails fast with a clear configuration error message


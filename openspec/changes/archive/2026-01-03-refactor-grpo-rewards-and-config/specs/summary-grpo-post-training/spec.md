# summary-grpo-post-training Specification (Delta)

## ADDED Requirements

### Requirement: Reward identifiers use namespaced dot form
Summary GRPO configs SHALL use namespaced dot identifiers for reward functions (e.g., `summary.format`, `summary.header`, `summary.parse`). Legacy snake-case identifiers (e.g., `summary_format`) are unsupported and SHALL fail validation.

#### Scenario: Legacy reward identifier is rejected
- **GIVEN** a GRPO config whose `rlhf.reward_funcs` includes `summary_format`
- **WHEN** configuration validation runs
- **THEN** validation fails with a clear error that legacy reward identifiers are unsupported

### Requirement: GRPO CHORD configuration uses `custom.grpo.chord`
When CHORD is enabled for summary GRPO, configs SHALL define it under `custom.grpo.chord` and SHALL include required fields: `sft_per_device_train_batch_size`, `mu_warmup_steps`, `mu_decay_steps`, `mu_peak`, `mu_valley`, and `enable_phi_function`. The legacy `custom.grpo_chord` field is unsupported and SHALL fail validation.

#### Scenario: CHORD config is validated
- **GIVEN** a GRPO config with `custom.grpo.chord.enabled: true`
- **WHEN** configuration validation runs
- **THEN** all required CHORD fields must be present and valid
- **AND** validation fails if only `custom.grpo_chord` is provided

# dense-grpo-post-training (Delta)

## ADDED Requirements

### Requirement: Batch plan shorthand is supported for dense GRPO post-training
Dense GRPO post-training SHALL support the same `custom.grpo.batch_plan` shorthand as summary GRPO, expanding it into the concrete training + rollout knobs.

#### Scenario: Dense GRPO uses unified batch plan
- **GIVEN** a dense GRPO config with `custom.grpo.batch_plan.enabled: true`
- **WHEN** the config is loaded via `ConfigLoader.load_yaml_with_extends`
- **THEN** the resolved config contains `training.effective_batch_size` and `rlhf.generation_batch_size` set to the same `unified_batch_size`.


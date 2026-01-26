# summary-grpo-post-training (Delta)

## ADDED Requirements

### Requirement: Batch plan shorthand is supported for summary GRPO (operator interface)
When running summary-mode GRPO, the system SHALL support an optional shorthand configuration block `custom.grpo.batch_plan` that expands into the concrete ms-swift/Qwen3-VL knobs.

The shorthand MUST be disabled by default and MUST preserve legacy behavior when absent.

#### Scenario: Shorthand is disabled by default (legacy knobs remain source of truth)
- **GIVEN** a summary GRPO config that does not include `custom.grpo.batch_plan`
- **WHEN** the config is loaded via `ConfigLoader.load_yaml_with_extends`
- **THEN** the loader performs no batch-plan expansion
- **AND** any configured legacy knobs (e.g. `training.effective_batch_size`, `rlhf.generation_batch_size`, `custom.extra.rollout_server.*`) remain unchanged.

#### Scenario: Shorthand expands into concrete knobs
- **GIVEN** a summary GRPO config with:
  ```yaml
  custom:
    grpo:
      batch_plan:
        enabled: true
        per_device_train_batch_size: 8
        per_device_eval_batch_size: 8
        unified_batch_size: 48
        rollout_server:
          force_vllm_tensor_parallel_size: 1
          force_vllm_data_parallel_size: 2
          max_num_seqs_per_gpu: 4
  ```
- **WHEN** the config is loaded via `ConfigLoader.load_yaml_with_extends`
- **THEN** the resolved config includes:
  - `training.per_device_train_batch_size == 8`
  - `training.per_device_eval_batch_size == 8`
  - `training.effective_batch_size == 48`
  - `rlhf.generation_batch_size == 48`
  - `custom.extra.rollout_server.vllm_tensor_parallel_size == 1`
  - `custom.extra.rollout_server.vllm_data_parallel_size == 2`
  - `custom.extra.rollout_server.vllm_max_num_seqs == 4`

#### Scenario: Conflicting legacy knobs fail fast
- **GIVEN** shorthand is enabled with `unified_batch_size: 48`
- **AND** the config also sets `rlhf.generation_batch_size: 96`
- **WHEN** configuration validation runs
- **THEN** loading fails fast with an error that names both `custom.grpo.batch_plan.unified_batch_size` and `rlhf.generation_batch_size` as conflicting sources of truth.

#### Scenario: Conflicting rollout-server knobs fail fast
- **GIVEN** shorthand is enabled with a rollout server plan forcing `force_vllm_data_parallel_size: 2`
- **AND** the config also sets `custom.extra.rollout_server.vllm_data_parallel_size: 1`
- **WHEN** the config is loaded via `ConfigLoader.load_yaml_with_extends`
- **THEN** loading fails fast with an error that names both:
  - `custom.grpo.batch_plan.rollout_server.force_vllm_data_parallel_size`
  - `custom.extra.rollout_server.vllm_data_parallel_size`
  as conflicting sources of truth.

#### Scenario: World-size-aware alignment validation
- **GIVEN** shorthand is enabled with:
  - `per_device_train_batch_size = 8`
  - `unified_batch_size = 48`
- **AND** `WORLD_SIZE=6` in the environment during initialization/validation
- **WHEN** training initializes (or config validation runs with world size available)
- **THEN** configuration validation confirms `48 % (8*6) == 0`
- **AND** the derived `gradient_accumulation_steps` equals the derived `steps_per_generation`.

#### Scenario: Unified batch respects num_generations divisibility constraints
- **GIVEN** shorthand is enabled with `unified_batch_size: 48`
- **AND** `rlhf.num_generations: 5`
- **WHEN** GRPO configuration validation runs
- **THEN** validation fails because `generation_batch_size % num_generations == 0` is required by GRPO.

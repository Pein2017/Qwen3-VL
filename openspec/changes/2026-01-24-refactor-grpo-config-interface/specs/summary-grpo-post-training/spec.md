# summary-grpo-post-training (Delta)

## MODIFIED Requirements

### Requirement: Batch plan shorthand is supported for summary GRPO (operator interface)
When running summary-mode GRPO, the system SHALL support an optional shorthand configuration block `custom.grpo.batch_plan` that expands into the concrete ms-swift/Qwen3-VL knobs.

The shorthand MUST be disabled by default and MUST preserve legacy behavior when absent.

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

#### Scenario: World-size-aware alignment validation
- **GIVEN** shorthand is enabled with:
  - `per_device_train_batch_size = 8`
  - `unified_batch_size = 48`
- **AND** the runtime world size is 6
- **WHEN** training initializes
- **THEN** configuration validation confirms `48 % (8*6) == 0`
- **AND** the derived `gradient_accumulation_steps` equals the derived `steps_per_generation`.


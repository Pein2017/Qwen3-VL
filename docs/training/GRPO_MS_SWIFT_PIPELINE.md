# ms-swift GRPO Training Pipeline (Code-Derived)

Status: Internal reference for Qwen3-VL GRPO usage with ms-swift.

## Scope and Sources
- Source of truth: ms-swift repository at `/data/ms-swift`.
- This document only records behavior observed in code; external dependencies (e.g., TRL / Transformers) are noted where applicable.
- Operational notes (launch patterns, weight-sync caveats) reference ms-swift documentation in
  `/data/ms-swift/docs/source_en/Instruction/GRPO/GetStarted/GRPO.md`.

Key code anchors (non-exhaustive):
- GRPO argument defaults and validation: `swift/trainers/rlhf_arguments.py:77-110`
- Base argument fields: `swift/trainers/arguments.py:278-335`
- GRPO trainer rollout buffer and regeneration: `swift/trainers/rlhf_trainer/grpo_trainer.py:133-177`
- Grouped advantages by `num_generations`: `swift/trainers/rlhf_trainer/grpo_trainer.py:358-489`
- Generation mini-batch splitting by `steps_per_generation`: `swift/trainers/rlhf_trainer/grpo_trainer.py:674-745`
- Sequence-parallel sampler repeat logic: `swift/trainers/rlhf_trainer/grpo_trainer.py:139-147`
- vLLM engine capacity sizing: `swift/trainers/rlhf_trainer/rollout_mixin.py:181-182`
- vLLM mode selection + server validation: `swift/llm/argument/rlhf_args.py:243-268`
- External vLLM warning (max_model_len ignored in server mode): `swift/llm/argument/rlhf_args.py:384-392`
- Server weight-sync client: `swift/trainers/rlhf_trainer/vllm_client.py:178-235`

## Pipeline Outline (Code Path)

1. **Args + trainer wiring**
   - `rlhf_type=grpo` selects `GRPOTrainer` and `GRPOConfig` via `TrainerFactory`.
   - Locations: `swift/trainers/trainer_factory.py:13-42`
2. **RLHF entrypoint prepares models + template**
   - Reward models are prepared (if provided), and GRPO rewards are passed via `reward_funcs`.
   - Template mode uses `train` for GRPO.
   - Locations: `swift/llm/train/rlhf.py:124-175`, `swift/llm/train/rlhf.py:206-219`
3. **GRPO trainer initialization**
   - Sets GRPO algorithm params, prepares rollout engine, and validates reward sources.
   - Locations: `swift/trainers/rlhf_trainer/grpo_trainer.py:57-88`
4. **Rollout engine setup**
   - Colocate vLLM uses `max_num_seqs` derived from batch size and `steps_per_generation`.
   - Locations: `swift/trainers/rlhf_trainer/rollout_mixin.py:175-182`
5. **Training loop**
   - Rollout outputs are buffered and reused across steps; regeneration is periodic.
   - Locations: `swift/trainers/rlhf_trainer/grpo_trainer.py:133-177`

## Parameter Meanings and Defaults

### `gradient_accumulation_steps` / `effective_batch_size`
- Default for GRPO (when unset): forced to `1`.
- In Qwen3-VL configs, prefer setting `training.effective_batch_size`; the loader derives
  `gradient_accumulation_steps = ceil(effective_batch_size / (per_device_train_batch_size * world_size))`
  for both SFT and GRPO.
- Locations:
  - GRPO default: `swift/llm/argument/rlhf_args.py:331-334`
  - Loader auto-calc: `src/config/loader.py:261-306`

### `generation_batch_size` (recommended rollout size knob)
- Optional argument declared in GRPO mixin.
- **Recommended**: set `generation_batch_size` directly and do not set `steps_per_generation`.
- If unset: computed as global generation size.
- Formula (global):
  - `generation_batch_size = per_device_train_batch_size * world_size * steps_per_generation`
- Locations:
  - Field: `swift/trainers/arguments.py:333-334`
  - Defaulting: `swift/trainers/rlhf_arguments.py:80-81`

### `steps_per_generation` (derived when `generation_batch_size` is set)
- Optional argument declared in GRPO mixin.
- If unset: defaults to `gradient_accumulation_steps`.
- If `generation_batch_size` is set: recomputed as
  - `steps_per_generation = generation_batch_size // (per_device_train_batch_size * world_size)`
- Locations:
  - Field: `swift/trainers/arguments.py:333-334`
  - Defaulting: `swift/trainers/rlhf_arguments.py:78-79`
  - Recompute: `swift/trainers/rlhf_arguments.py:94`

### Coupling and Overrides
- `steps_per_generation` is recomputed from `generation_batch_size` during validation:
  - `steps_per_generation = generation_batch_size // (per_device_train_batch_size * world_size)`
- Location: `swift/trainers/rlhf_arguments.py:94`
- Implication: when both are provided, `generation_batch_size` takes precedence and overwrites `steps_per_generation`.

### `num_generations`
- Must satisfy divisibility constraints:
  - `generation_batch_size % num_generations == 0`
- Location: `swift/trainers/rlhf_arguments.py:101-109`

### `num_generations` (default)
- Default is `8`.
- Location: `swift/llm/argument/rlhf_args.py:59-61`

### `num_iterations`
- Default is `1` (multi-step GRPO, mu in the paper).
- Location: `swift/llm/argument/rlhf_args.py:65-66`

### `num_iterations` (trainer usage)
- Multi-step GRPO parameter `mu` from the paper.
- Location: `swift/trainers/rlhf_trainer/grpo_trainer.py:1844-1846`

### GRPO-specific defaults and constraints
- `cached_dataset` is not supported for GRPO.
  - Location: `swift/llm/argument/rlhf_args.py:195-199`
- `truncation_strategy` defaults to `left` and must be `left` or `delete`.
  - Location: `swift/llm/argument/rlhf_args.py:206-210`
- `remove_unused_columns` is forced to `False` for GRPO.
  - Location: `swift/llm/argument/rlhf_args.py:204-205`
- `beta` defaults to `0.04` when unset for GRPO.
  - Location: `swift/llm/argument/rlhf_args.py:211-212`
- `kl_in_reward` defaults by advantage estimator (`grpo` -> `False`).
  - Location: `swift/llm/argument/rlhf_args.py:225-229`
- `scale_rewards` defaults by advantage estimator (`grpo` -> `group`).
  - Location: `swift/llm/argument/rlhf_args.py:233-236`
- `dataloader_drop_last` is forced to `True`.
  - Location: `swift/trainers/rlhf_arguments.py:74-75`
- A reward source is required: either `reward_funcs` or `reward_model`.
  - Location: `swift/trainers/rlhf_trainer/grpo_trainer.py:86-87`

## Batch Size Semantics

### Global generation batch size
- Counts total rollout samples (prompt + completion pairs) across all processes for one generation cycle.
- Formula:
  - `generation_batch_size = per_device_train_batch_size * world_size * steps_per_generation`
- Location: `swift/trainers/rlhf_arguments.py:80-81`

### Prompt batch size (grouped rollouts)
- Prompt count is derived as:
  - `prompt_batch_size = generation_batch_size // num_generations`
- Location: `swift/trainers/rlhf_trainer/grpo_trainer.py:145`

### vLLM capacity
- vLLM colocate engine sets a maximum sequence budget based on `steps_per_generation`:
  - `max_num_seqs = per_device_train_batch_size * vllm_tensor_parallel_size * steps_per_generation`
- Location: `swift/trainers/rlhf_trainer/rollout_mixin.py:181-182`

## Rollout Integration Modes (Colocate vs Server)

### Colocate (internal vLLM)
- `use_vllm: true` and `vllm_mode: colocate` start vLLM inside the trainer process.
- `vllm_tensor_parallel_size` must divide the trainer world size evenly; otherwise the rollout mixin raises.
  - Location: `swift/trainers/rlhf_trainer/rollout_mixin.py:153-170`
- vLLM capacity is sized from training batch and `steps_per_generation` (see Batch Size Semantics).
- Memory relief knobs (from ms-swift docs): `sleep_level`, `offload_optimizer`, `offload_model`, and lower
  `vllm_gpu_memory_utilization`. See `/data/ms-swift/docs/source_en/Instruction/GRPO/GetStarted/GRPO.md`.

### Server (external `swift rollout`)
- `vllm_server_host` or `vllm_server_base_url` forces `vllm_mode=server`; missing host forces `colocate`.
  - Location: `swift/llm/argument/rlhf_args.py:243-256`
- `async_generate` requires `vllm_mode=server`.
  - Location: `swift/llm/argument/rlhf_args.py:262-264`
- External rollout uses `swift rollout` and vLLM backend only. Recommended launch (single node):
  ```bash
  CUDA_VISIBLE_DEVICES=0,1 \
  swift rollout \
    --model <MODEL_PATH> \
    --vllm_tensor_parallel_size 2 \
    --vllm_data_parallel_size 1
  ```
  - Source: `/data/ms-swift/docs/source_en/Instruction/GRPO/GetStarted/GRPO.md`
- Training side connects via `vllm_server_host` / `vllm_server_port` / `vllm_server_timeout`.
  - Location: `swift/llm/argument/rlhf_args.py:303-313`
- `vllm_max_model_len` set in training config is ignored in server mode; set it on `swift rollout`.
  - Location: `swift/llm/argument/rlhf_args.py:384-392`
- Weight sync uses an NCCL communicator created by `VLLMClient` after `/init_communicator/`:
  - Location: `swift/trainers/rlhf_trainer/vllm_client.py:178-235`
  - Operational implication: the trainer process must reach the rollout host and its group port.

### Server-mode guardrails (from ms-swift docs)
- `use_async_engine` with only DP may fail; use both TP and DP or upgrade vLLM.
  - Source: `/data/ms-swift/docs/source_en/Instruction/GRPO/GetStarted/GRPO.md`
- Weight-sync acceleration for LoRA:
  - Rollout: `--vllm_enable_lora true --vllm_max_lora_rank <lora_rank>`
  - Colocate: `--vllm_enable_lora true`
  - Not supported when training multimodal ViT layers or MoE models.
  - Source: `/data/ms-swift/docs/source_en/Instruction/GRPO/GetStarted/GRPO.md`

## `num_generations` Semantics (Grouping vs Sampling)

### Default grouped mode
- Advantages assume a strict group of `num_generations` samples per prompt.
- Rewards are reshaped as `rewards.view(-1, num_generations)`.
- Location: `swift/trainers/rlhf_trainer/grpo_trainer.py:358-443`

### Prompt repetition behavior
- In sequence-parallel mode, the training sampler uses:
  - `mini_repeat_count = num_generations`
  - `batch_size = generation_batch_size // num_generations`
- Location: `swift/trainers/rlhf_trainer/grpo_trainer.py:142-145`
- This indicates explicit prompt repetition to ensure K completions per prompt in grouped mode.

### Request-aware (dynamic) mode
- When `dynamic_num_samples` is active, grouping is done by `request_id` and `prompt_id` rather than fixed K.
- Location: `swift/trainers/rlhf_trainer/grpo_trainer.py:496-520`

## Rollout Lifecycle (Buffer, Regeneration, Training Steps)

### Buffer storage
- Buffered rollout inputs are stored in `self._buffered_inputs` for reuse across steps.
- Location: `swift/trainers/rlhf_trainer/grpo_trainer.py:136-176`

### Regeneration condition
- For training mode:
  - `num_rollout_samples = steps_per_generation * sequence_parallel_size`
  - `generate_every = num_rollout_samples * num_iterations`
  - Regenerate if: `_step % generate_every == 0` or buffer is empty.
- Location: `swift/trainers/rlhf_trainer/grpo_trainer.py:171-175`

### Mini-batch slicing for training
- After rollout, generated inputs are split into `steps_per_generation` chunks.
- Location: `swift/trainers/rlhf_trainer/grpo_trainer.py:674-707`

### Sequence-parallel note
- In sequence-parallel mode, the split count is `steps_per_generation * sequence_parallel.world_size` during training.
- Location: `swift/trainers/rlhf_trainer/grpo_trainer.py:723-725`

### Buffer reuse and regeneration interval
- `_prepare_inputs` buffers the generated rollout once per generation cycle and reuses it for multiple steps.
- Regeneration happens every `generate_every = (steps_per_generation * sequence_parallel_size) * num_iterations` steps,
  or when the buffer is empty.
- Location: `swift/trainers/rlhf_trainer/grpo_trainer.py:171-176`

### Pseudocode of the outer loop
```
initialize _step = 0
initialize _buffered_inputs = None

for each training step:
  num_rollout_samples = steps_per_generation * sequence_parallel_size
  generate_every = num_rollout_samples * num_iterations

  if _step % generate_every == 0 or _buffered_inputs is None:
    inputs = generate_completions(generation_batch)
    rewards = score_completions(inputs)
    batch_encoded = prepare_batch_inputs(inputs)
    advantages = compute_advantages(inputs, rewards, batch_encoded)
    split into steps_per_generation mini-batches
    store as _buffered_inputs

  inputs = _buffered_inputs[_step % num_rollout_samples]
  _step += 1
  run forward/backward on inputs
```

## Iteration Counters and Logging

### Training-step counter used for rollout
- `_step` tracks forward/backward iterations (including accumulation) and gates rollout regeneration.
- Location: `swift/trainers/rlhf_trainer/grpo_trainer.py:133-177`

### Progress bar and logged step
- Logging and progress use `state.global_step` (optimizer steps).
- Location: `swift/trainers/callback.py:27-48`
- Completion tables also record `state.global_step`.
- Location: `swift/trainers/rlhf_trainer/grpo_trainer.py:1616-1618`

## Why Increasing `steps_per_generation` Can Increase Total Iterations

Observed in code (sequence-parallel path):
- Dataloader repetition scales with `steps_per_generation`:
  - `repeat_count = num_iterations * steps_per_generation * sequence_parallel.world_size`
  - Location: `swift/trainers/rlhf_trainer/grpo_trainer.py:146`
- Larger `steps_per_generation` increases `generation_batch_size`, which increases the number of rollout samples and mini-batches processed per rollout.
  - Location: `swift/trainers/rlhf_arguments.py:80-81`
  - Location: `swift/trainers/rlhf_trainer/grpo_trainer.py:674-707`

Implication:
- When training is epoch-driven (not max-steps driven), the dataloader length can expand with `steps_per_generation`, raising total optimizer steps and therefore logged iterations (`state.global_step`).
- The precise mapping from dataloader length to `state.max_steps` is implemented in external Transformers/TRL Trainer code.

## Dynamic Sampling vs Dynamic Num-Samples

### `dynamic_sample` (DAPO / resampling)
- Configurable GRPO argument for resampling behavior.
- Related fields: `dynamic_sample`, `max_resample_times`, `overlong_filter`.
- Locations: `swift/trainers/arguments.py:310-312`, `swift/trainers/rlhf_trainer/grpo_trainer.py:1847-1852`

### `dynamic_num_samples` (server multi-turn runtime)
- Internal runtime flag used for **server multi-turn** rollouts when the vLLM server returns a different
  number of outputs than requests.
- Trigger: in server mode, if `outputs_count != len(all_requests)`, the trainer sets
  `dynamic_num_samples = True` and switches to even data distribution (non-uniform per-rank sizes).
- Constraints: padding-free mode is **not supported** in this path; a mismatch raises `NotImplementedError`.
- Impact:
  - Advantage grouping switches to `request_id` / `prompt_id` rather than fixed `num_generations`.
  - Mini-batch splitting and chunking are guarded for variable batch sizes.
- Locations:
  - Trigger + constraints: `swift/trainers/rlhf_trainer/rollout_mixin.py:733-749`
  - Request-aware advantages: `swift/trainers/rlhf_trainer/grpo_trainer.py:441-520`
  - Chunking/splitting guards: `swift/trainers/rlhf_trainer/grpo_trainer.py:674-745`

## Config Mapping (Qwen3-VL Examples)

### `configs/grpo/summary_grpo_base.yaml`
- `rlhf.rlhf_type: grpo` routes to `GRPOTrainer` / `GRPOConfig`.
- `rlhf.reward_funcs` must be non-empty (or set `reward_model`).
- `rlhf.reward_weights` length must match `reward_funcs`.
- `training.effective_batch_size` controls backward accumulation; `per_device_train_batch_size` remains the micro-batch.
- `rlhf.generation_batch_size` is the global rollout size (total trajectories per generation cycle).
- `rlhf.num_generations` must divide the global `generation_batch_size`.
- `training.eval_strategy: no` bypasses GRPO eval batch divisibility checks.

Locations:
- Config file: `configs/grpo/summary_grpo_base.yaml:8-40`
- Requirement for reward source: `swift/trainers/rlhf_trainer/grpo_trainer.py:86-87`
- Divisibility checks: `swift/trainers/rlhf_arguments.py:89-110`
- Eval strategy check: `swift/trainers/rlhf_arguments.py:111-121`

## External Dependencies (Not in ms-swift Tree)
- Non-sequence-parallel training sampler behavior depends on the TRL/Transformers GRPO trainer base class.
- This repository references TRL symbols (e.g., `RepeatSampler`), but base trainer logic is not duplicated in ms-swift.

## Cross-Checks and Guardrails
- `generation_batch_size` and `steps_per_generation` require TRL >= 0.18 (assertion in argument checks).
  - Location: `swift/llm/argument/rlhf_args.py:378-382`
- Old-policy path activation when `gradient_accumulation_steps % steps_per_generation != 0`.
  - Location: `swift/trainers/rlhf_trainer/grpo_trainer.py:1506-1507`
- Sequence-parallel old-policy condition uses `steps_per_generation * sequence_parallel.world_size`.
  - Location: `swift/trainers/rlhf_trainer/grpo_trainer.py:1509-1511`

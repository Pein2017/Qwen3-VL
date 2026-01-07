# ms-swift GRPO Training Pipeline (Code-Derived)

Status: Active
Scope: Code-derived GRPO pipeline notes for ms-swift integration (internal reference).
Owners: Training
Last updated: 2026-01-07
Related: [REFERENCE.md](REFERENCE.md), [configs/train/grpo/summary_base.yaml](../../configs/train/grpo/summary_base.yaml)

## Scope and Sources
- Source of truth: local ms-swift repository (external to this repo; see [docs/ops/UPSTREAM_DEPENDENCIES.md](../ops/UPSTREAM_DEPENDENCIES.md)).
- This document only records behavior observed in code; external dependencies (e.g., TRL / Transformers) are noted where applicable.
- Operational notes (launch patterns, weight-sync caveats) reference the ms-swift GRPO guide (see upstream doc pointers in [docs/ops/UPSTREAM_DEPENDENCIES.md](../ops/UPSTREAM_DEPENDENCIES.md)).

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

## CHORD (SFT-mixed) Loss for GRPO (Qwen3-VL Toggle)

ms-swift GRPO supports CHORD-style loss mixing: a supervised (SFT) loss is mixed into GRPO loss to provide a fallback gradient even when advantages collapse (e.g., `reward_std == 0` groups).

### How ms-swift CHORD mixing works
- When enabled, GRPOTrainer combines losses:
  - `loss = (1 - mu) * grpo_loss + mu * chord_sft_loss`
- `mu` is scheduled by warmup + cosine decay (or constant if decay steps are 0).
- Code locations:
  - Mix point: `swift/trainers/rlhf_trainer/grpo_trainer.py:1085-1087`
  - Implementation: `swift/trainers/rlhf_trainer/utils.py:945-987`

### Qwen3-VL config surface
Qwen3-VL exposes a config-only toggle under `custom.grpo.chord` (GRPO-only):

```yaml
custom:
  grpo:
    chord:
      enabled: true
      sft_per_device_train_batch_size: 1
      mu_warmup_steps: 100
      mu_decay_steps: 0
      mu_peak: 0.05
      mu_valley: 0.05
      enable_phi_function: false
```

Mapping to ms-swift trainer args:
- `custom.grpo.chord.sft_per_device_train_batch_size` → `chord_sft_per_device_train_batch_size`
- `custom.grpo.chord.mu_*` → `chord_mu_*`
- `custom.grpo.chord.enable_phi_function` → `chord_enable_phi_function`

Qwen3-VL passes `chord_sft_dataset` to the GRPO trainer using the same fusion train dataset stream as expert targets (including irrelevant samples with target `无关图片`).

### Batch size guidance for tight VRAM
- Keep GRPO micro-batch minimal: `training.per_device_train_batch_size: 1`.
- Control effective update size via gradient accumulation (`training.effective_batch_size`).
- Keep CHORD SFT micro-batch minimal: `custom.grpo.chord.sft_per_device_train_batch_size: 1`.

Note: CHORD adds an extra supervised forward/backward per train step when `mu > 0`, so overall VRAM and step time increase.
Note: With DeepSpeed ZeRO-2, CHORD introduces a second forward pass per step. If DeepSpeed asserts on duplicated gradient
reductions, override DeepSpeed config to disable `zero_optimization.reduce_scatter` (example: `configs/deepspeed/zero2_chord_no_reduce_scatter.json`).

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
  `vllm_gpu_memory_utilization`. See the ms-swift GRPO guide referenced in [docs/ops/UPSTREAM_DEPENDENCIES.md](../ops/UPSTREAM_DEPENDENCIES.md).

#### CUDA OOM after `eval_steps` (reserved-memory creep)

In GRPO + vLLM **colocate** mode, evaluation can temporarily allocate extra CUDA workspaces (PyTorch + vLLM). Even when
those tensors go out of scope, CUDA **reserved** memory (allocator cache) can remain high and drift upward across
eval cycles, reducing free headroom and eventually triggering CUDA OOM (often after a few evals).

Mitigation in Qwen3-VL:
- Qwen3-VL enables an **eval/save CUDA cache cleanup hook by default** (runs `gc.collect()` + `torch.cuda.empty_cache()`
  after evaluation / checkpoint saves; also best-effort resets vLLM prefix cache).
- This does **not** change GRPO batch size or vLLM GPU settings; it only releases unused reserved memory back to the
  driver at eval/save boundaries to prevent monotonic headroom loss.

Controls (all under `custom.cuda_memory`):
- Disable completely: set `custom.cuda_memory.enabled: false`
- Disable cleanup only (keep hook for optional profiling): set `custom.cuda_memory.cleanup: false` or export
  `QWEN3VL_DISABLE_CUDA_EVAL_CLEANUP=1`
- Enable profiling trace: set `custom.cuda_memory.profile: true` (writes `cuda_memory_trace.rank{RANK}.jsonl` into
  `training.output_dir`; defaults to rank-0 logging only via `rank0_only: true`)

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
  - Source: ms-swift GRPO guide (see [docs/ops/UPSTREAM_DEPENDENCIES.md](../ops/UPSTREAM_DEPENDENCIES.md))
- Training side connects via `vllm_server_host` / `vllm_server_port` / `vllm_server_timeout`.
  - Location: `swift/llm/argument/rlhf_args.py:303-313`
- `vllm_max_model_len` set in training config is ignored in server mode; set it on `swift rollout`.
  - Location: `swift/llm/argument/rlhf_args.py:384-392`
- Weight sync uses an NCCL communicator created by `VLLMClient` after `/init_communicator/`:
  - Location: `swift/trainers/rlhf_trainer/vllm_client.py:178-235`
  - Operational implication: the trainer process must reach the rollout host and its group port.

### Server-mode guardrails (from ms-swift docs)
- `use_async_engine` with only DP may fail; use both TP and DP or upgrade vLLM.
  - Source: ms-swift GRPO guide (see [docs/ops/UPSTREAM_DEPENDENCIES.md](../ops/UPSTREAM_DEPENDENCIES.md))
- Weight-sync acceleration for LoRA:
  - Rollout: `--vllm_enable_lora true --vllm_max_lora_rank <lora_rank>`
  - Colocate: `--vllm_enable_lora true`
  - Not supported when training multimodal ViT layers or MoE models.
  - Source: ms-swift GRPO guide (see [docs/ops/UPSTREAM_DEPENDENCIES.md](../ops/UPSTREAM_DEPENDENCIES.md))

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

### `configs/train/grpo/summary_base.yaml`
- `rlhf.rlhf_type: grpo` routes to `GRPOTrainer` / `GRPOConfig`.
- `rlhf.reward_funcs` must be non-empty (or set `reward_model`).
- `rlhf.reward_weights` length must match `reward_funcs`.
- Summary-mode reward funcs used by the base template are implemented in `src/rlhf/grpo/rewards/summary/rewards.py` and include:
  - Contract + guardrails: `summary.format`, `summary.header`, `summary.strict`, `summary.parse`
  - Hard JSON correctness: `summary.no_dup_keys` (hard-penalize duplicate JSON keys, including nested dicts)
  - Core content alignment (strict-format gated, GT treated as lower bound):
    - `summary.dataset` (header domain matches `_fusion_template` mapping)
    - `summary.category_recall` (category recall over `统计[*].类别`)
    - `summary.content_structured_tversky` (recall-biased Tversky on structured facts; BBU excludes `文本/备注`; RRU is stricter)
  - BBU free-text handling:
    - `summary.text_bbu` (OCR `文本` lower-bound recall with punctuation normalization and `+2` unique-string slack)
    - `summary.notes_bbu` (recall when GT has `备注`; hard-penalize spurious notes when GT has none)
  - Conditional domain structure: `summary.group_stats_presence` (RRU-only, only when GT has `分组统计`)
- `training.effective_batch_size` controls backward accumulation; `per_device_train_batch_size` remains the micro-batch.
- `rlhf.generation_batch_size` is the global rollout size (total trajectories per generation cycle).
- `rlhf.num_generations` must divide the global `generation_batch_size`.
- `training.eval_strategy: no` bypasses GRPO eval batch divisibility checks.

Locations:
- Config file: `configs/train/grpo/summary_base.yaml`
- Requirement for reward source: `swift/trainers/rlhf_trainer/grpo_trainer.py:86-87`
- Divisibility checks: `swift/trainers/rlhf_arguments.py:89-110`
- Eval strategy check: `swift/trainers/rlhf_arguments.py:111-121`

### `configs/train/grpo/summary_server.yaml`
- Extends `summary_2048.yaml` (which extends `summary_base.yaml`) and switches vLLM to server mode.
- Sets `rlhf.vllm_mode: server` plus server host/port/timeouts and tensor parallel sizing.

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

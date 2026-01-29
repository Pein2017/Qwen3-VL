# Proposal: Add GRPO Batch Plan Shorthand (Unified Batch + Rollout Server)

## Context / Problem
GRPO configs currently require operators to reason about multiple coupled knobs across:
- learner micro-batching (`training.per_device_train_batch_size`, `training.effective_batch_size` → derived `gradient_accumulation_steps`)
- rollout buffering (`rlhf.generation_batch_size` → derived `steps_per_generation`)
- rollout server topology and concurrency (`custom.extra.rollout_server.*`)

In practice, operators usually want a simple, stable, “no-thinking” interface:
- set per-device micro-batch sizes (train/eval) to control CUDA memory
- set a single global batch size `B` used for both training updates and rollout refreshes
- set rollout server DP/TP and a per-GPU parallel decoding cap

Today this requires manual arithmetic and can accidentally trigger mismatched accumulation vs rollout buffering (e.g. `gradient_accumulation_steps % steps_per_generation != 0`), which may activate ms-swift GRPO fallback paths and increase instability.

## Goals
- Provide a high-level YAML interface for GRPO runs that:
  1) removes the need to manually set/think about `gradient_accumulation_steps`
  2) removes the need to manually set/think about `steps_per_generation`
  3) keeps `training.effective_batch_size` and `rlhf.generation_batch_size` aligned by construction
  4) can force a rollout-server plan (TP=1, DP=2) and a low per-GPU concurrency cap (e.g. 4)
- Keep backward compatibility: existing configs without the shorthand must behave exactly as before.
- Make the behavior traceable: resolved configs and logs should clearly show derived values.

## Non-Goals
- Changing ms-swift behavior.
- Removing existing knobs (only provide a safer shorthand).
- Multi-node rollout server support changes.

## Proposed User-Facing YAML Interface
Introduce a new optional block under `custom.grpo`:

```yaml
custom:
  grpo:
    batch_plan:
      enabled: true

      # Learner micro-batch sizes (CUDA memory control)
      per_device_train_batch_size: 8
      per_device_eval_batch_size: 8  # optional; defaults to train

      # Unified global batch size used for BOTH:
      # - training.effective_batch_size
      # - rlhf.generation_batch_size
      unified_batch_size: 48

      # Rollout server plan (server-mode GRPO)
      rollout_server:
        force_vllm_tensor_parallel_size: 1
        force_vllm_data_parallel_size: 2
        max_num_seqs_per_gpu: 4
```

Behavior:
- When enabled, the loader populates:
  - `training.per_device_train_batch_size`
  - `training.per_device_eval_batch_size` (defaulting to train)
  - `training.effective_batch_size = unified_batch_size`
  - `rlhf.generation_batch_size = unified_batch_size`
- When world size is known (distributed run), the loader validates:
  - `unified_batch_size % (per_device_train_batch_size * world_size) == 0`
  - This ensures `gradient_accumulation_steps == steps_per_generation` and avoids the GRPO “old_policy” fallback.
- When rollout_server plan is provided, the loader ensures `custom.extra.rollout_server` contains:
  - `vllm_tensor_parallel_size = 1`
  - `vllm_data_parallel_size = 2`
  - `vllm_max_num_seqs = max_num_seqs_per_gpu`

## Compatibility / Precedence Rules
- If `custom.grpo.batch_plan.enabled=true` and the user also sets any of:
  - `training.effective_batch_size`
  - `training.gradient_accumulation_steps`
  - `rlhf.generation_batch_size`
  - `rlhf.steps_per_generation`
  - `custom.extra.rollout_server.vllm_tensor_parallel_size`
  - `custom.extra.rollout_server.vllm_data_parallel_size`
  - `custom.extra.rollout_server.vllm_max_num_seqs`
  with conflicting values,
  the loader fails fast with a clear error describing the conflict and the intended shorthand.

## Traceability
- `scripts/config_tools/inspect_config.py inspect` should show the resolved config (including derived values) so operators can diff configs deterministically.
- Training startup logs should include a concise summary of derived batch values:
  - world_size, per_device_train_batch_size, unified_batch_size
  - derived `gradient_accumulation_steps` and derived `steps_per_generation`

## Risks
- Some workflows call `ConfigLoader.load_yaml_with_extends()` outside a distributed env (world_size defaults to 1). The shorthand must therefore:
  - still populate the obvious derived fields (effective/generation batch sizes)
  - defer strict divisibility validation until world_size is known (or validate only when `WORLD_SIZE` is set).

## Rollout / Migration
- Add docs showing the new shorthand and explaining the mapping.
- Update at least one runnable GRPO preset to demonstrate the shorthand (while keeping existing presets valid).


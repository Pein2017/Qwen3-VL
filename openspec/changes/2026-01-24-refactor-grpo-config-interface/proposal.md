# Proposal: Add GRPO Batch Plan Shorthand (Unified Batch + Rollout Server)

## Why
GRPO configs currently require operators to reason about multiple coupled knobs across:
- learner micro-batching (`training.per_device_train_batch_size`, `training.effective_batch_size` → derived `gradient_accumulation_steps`)
- rollout buffering (`rlhf.generation_batch_size` → derived `steps_per_generation`)
- rollout server topology and concurrency (`custom.extra.rollout_server.*`)

In practice, operators usually want a simple, stable, “no-thinking” interface:
- set per-device micro-batch sizes (train/eval) to control CUDA memory
- set a single global batch size `B` used for both training updates and rollout refreshes
- set rollout server DP/TP and a per-GPU parallel decoding cap

Today this requires manual arithmetic and can accidentally trigger mismatched accumulation vs rollout buffering (e.g. `gradient_accumulation_steps % steps_per_generation != 0`), which may activate ms-swift GRPO fallback paths and increase instability.

## What Changes
- Provide a high-level YAML interface for GRPO runs that:
  1) removes the need to manually set/think about `gradient_accumulation_steps`
  2) removes the need to manually set/think about `steps_per_generation`
  3) keeps `training.effective_batch_size` and `rlhf.generation_batch_size` aligned by construction
  4) can force a rollout-server plan (TP=1, DP=2) and a low per-GPU concurrency cap (e.g. 4)
- Keep backward compatibility: existing configs without the shorthand must behave exactly as before.
- Make the behavior traceable: resolved configs and logs should clearly show derived values.

## Acceptance Criteria
- A config that enables `custom.grpo.batch_plan.enabled: true` can omit:
  - `training.gradient_accumulation_steps`
  - `rlhf.steps_per_generation`
  and still runs without manual batch arithmetic.
- Resolved configs loaded via `ConfigLoader.load_yaml_with_extends()` contain the expanded concrete knobs:
  - `training.per_device_train_batch_size`, `training.per_device_eval_batch_size`
  - `training.effective_batch_size`
  - `rlhf.generation_batch_size`
  - `custom.extra.rollout_server.*` when rollout_server plan is provided
- If shorthand is enabled and a legacy knob is set to a different value, config loading fails fast with a path-qualified error.
- When `WORLD_SIZE` is available, shorthand validates that the unified batch produces:
  - `gradient_accumulation_steps == steps_per_generation`
  to avoid ms-swift GRPO fallback behavior.

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
- When world size is known (distributed run), the loader validates divisibility:
  - Let `denom = per_device_train_batch_size * world_size`.
  - Require: `unified_batch_size % denom == 0`.
  - Rationale: ms-swift derives rollout buffering with integer division while the learner side may round up.
    With exact divisibility, the implied values match exactly:
    - `gradient_accumulation_steps = unified_batch_size / denom`
    - `steps_per_generation = unified_batch_size / denom`
    and avoids the GRPO “old_policy” fallback triggered by accumulation/buffering mismatch.
- Because `unified_batch_size` is also used as `generation_batch_size`, it must satisfy ms-swift divisibility constraints:
  - `unified_batch_size % rlhf.num_generations == 0`
- When rollout_server plan is provided, the loader ensures `custom.extra.rollout_server` contains:
  - `vllm_tensor_parallel_size = 1`
  - `vllm_data_parallel_size = 2`
  - `vllm_max_num_seqs = max_num_seqs_per_gpu`

## Compatibility / Precedence Rules
- If `custom.grpo.batch_plan.enabled=true` and the user also sets any of:
  - `training.per_device_train_batch_size`
  - `training.per_device_eval_batch_size`
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
- `scripts/config_tools/inspect_config.py inspect` shows the resolved config after shorthand expansion (i.e. the concrete knobs populated by the batch plan).
  - Note: ms-swift runtime-derived values (e.g. `gradient_accumulation_steps` when computed inside `build_train_arguments`, and `steps_per_generation` when recomputed from `generation_batch_size`) may not be materialized back into the YAML mapping; the goal is that the expanded knobs are visible and deterministic.
- Training startup logs should include a concise summary of derived batch values:
  - world_size, per_device_train_batch_size, unified_batch_size
  - derived `gradient_accumulation_steps` and derived `steps_per_generation`

## Impact
- Operator experience: fewer coupled knobs; easier to tune learner micro-batching vs rollout throughput without manual arithmetic.
- Backward compatibility: opt-in; configs without `custom.grpo.batch_plan.enabled: true` behave exactly as before.
- Failure mode: invalid combinations fail early with path-qualified errors instead of silently running with unexpected ms-swift defaults/overrides.

## Risks
- Some workflows call `ConfigLoader.load_yaml_with_extends()` outside a distributed env (world_size defaults to 1). The shorthand must therefore:
  - still populate the obvious derived fields (effective/generation batch sizes)
  - defer strict divisibility validation until world_size is known (or validate only when `WORLD_SIZE` is set).

## Rollout / Migration
- Add docs showing the new shorthand and explaining the mapping.
- Update at least one runnable GRPO preset to demonstrate the shorthand (while keeping existing presets valid).

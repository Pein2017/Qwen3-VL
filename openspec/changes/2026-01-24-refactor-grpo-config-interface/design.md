# Design: GRPO Batch Plan Shorthand

## Principle
Make the operator-facing interface small, but keep the internal representation explicit.

- Users set a small `custom.grpo.batch_plan` structure.
- The loader expands it into the existing knobs (`training.*`, `rlhf.*`, `custom.extra.rollout_server.*`).
- Downstream code remains unchanged and sees the explicit, fully-resolved config.

## Expansion Stage
Expansion should occur in `ConfigLoader.load_yaml_with_extends()` so it applies to:
- training runs (`src.sft.py`)
- config inspection/diff tooling (`scripts/config_tools/inspect_config.py`)
- server-mode rollout launcher extraction (`scripts/grpo_server_mode.sh` uses `ConfigLoader.load_yaml_with_extends`)

## Validation Strategy
Two-phase validation:

1) Always-on shape validation (independent of world size)
- validate types and required keys in `custom.grpo.batch_plan`.
- populate missing defaults (e.g., per_device_eval defaults to train).

2) World-size-aware semantic validation
- when `WORLD_SIZE` (or `_PATCH_WORLD_SIZE`) is present, validate divisibility:
  - `unified_batch_size % (per_device_train_batch_size * world_size) == 0`
- rationale: in non-distributed contexts, `world_size` may be unknown and default to 1.

## Conflict Detection
If shorthand is enabled, legacy knobs become “read-only”.
- If legacy knobs are absent: OK.
- If legacy knobs are present and equal to the derived values: OK.
- If legacy knobs are present and differ: fail fast with a path-qualified error.

## Rollout Server Plan
For server-mode GRPO, the plan can also force rollout server settings:
- TP=1, DP=2
- per-worker `vllm_max_num_seqs` set from `max_num_seqs_per_gpu`

This is implemented by populating/overriding `custom.extra.rollout_server` in the resolved config.

## Traceability
- The inspection tool shows the final expanded config.
- Training startup logs should include a single line summarizing:
  - per_device_train_batch_size, per_device_eval_batch_size, unified_batch_size
  - derived gradient_accumulation_steps and derived steps_per_generation (when world size known)


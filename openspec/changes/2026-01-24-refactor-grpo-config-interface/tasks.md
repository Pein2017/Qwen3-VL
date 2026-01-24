# Tasks: Add GRPO Batch Plan Shorthand

- [ ] Add schema support for `custom.grpo.batch_plan` (dataclass + validation) in `src/config/schema.py`.
- [ ] Implement shorthand expansion in `ConfigLoader.load_yaml_with_extends()` so resolved configs contain:
  - `training.per_device_train_batch_size`
  - `training.per_device_eval_batch_size`
  - `training.effective_batch_size`
  - `rlhf.generation_batch_size`
  - `custom.extra.rollout_server` overrides for TP/DP/max_num_seqs when rollout_server plan is provided.
- [ ] Implement strict conflict detection with actionable errors when shorthand is enabled but legacy knobs disagree.
- [ ] Implement world-size-aware validation ensuring `gradient_accumulation_steps == steps_per_generation`:
  - validate `unified_batch_size % (per_device_train_batch_size * world_size) == 0` when `WORLD_SIZE` (or `_PATCH_WORLD_SIZE`) is available.
- [ ] Add tests under `tests/rlhf/grpo/`:
  - [ ] Shorthand expands correctly for a synthetic config.
  - [ ] Conflicting legacy knobs fail fast.
  - [ ] World-size divisibility validation works using `_PATCH_WORLD_SIZE`.
  - [ ] Rollout server config extracted via `extract_rollout_server_launch_config` uses forced TP/DP/max_num_seqs.
- [ ] Update docs:
  - [ ] `docs/training/GRPO_MS_SWIFT_PIPELINE.md` add a “Batch Plan Shorthand” section with an intuitive example.
  - [ ] Mention how to inspect the resolved config via `scripts/config_tools/inspect_config.py`.
- [ ] Update at least one GRPO preset to demonstrate the new shorthand interface.
- [ ] Run `pytest -q` (or at least the GRPO config tests) and ensure no regressions.


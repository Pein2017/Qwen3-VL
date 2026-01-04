# Tasks

- [x] Implement new GRPO reward module layout under `src/rlhf/grpo/` with shared parsing/context utilities and consolidated helper functions.
- [x] Update reward registry to use namespaced dot identifiers and a single explicit registration entrypoint.
- [x] Remove legacy reward identifier support and update reward tests to the new module paths and names.
- [x] Standardize GRPO config validation (including CHORD) under a dedicated validator and replace inline validation in `src/sft.py`.
- [x] Update `configs/grpo/*` to use the new reward identifiers and the new `custom.grpo.chord` config shape.
- [x] Update docs: `docs/training/GRPO_MS_SWIFT_PIPELINE.md` and `docs/training/REFERENCE.md` to reflect the new config surface and reward identifiers.
- [x] Run config validation (e.g., `conda run -n ms python -m src.sft --config <grpo-config> --debug` or existing validator) and `pytest tests/test_summary_grpo_rewards.py`.

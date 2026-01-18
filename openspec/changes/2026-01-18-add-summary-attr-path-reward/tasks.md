# Tasks

- [x] Implement `summary.attr_path_recall` reward for summary GRPO (nested attribute paths, depth=2).
- [x] Register the reward in `src/rlhf/grpo/rewards/registry.py` and allow it in `src/rlhf/grpo/rewards/names.py`.
- [x] Add unit tests for nested-path extraction and reward behavior.
- [x] Update GRPO config preset(s) to include the new reward (non-breaking, optional).
- [x] Update documentation to describe when/how to use the reward.
- [x] Run `ruff` + `basedpyright` + relevant unit tests under conda env `ms`.

# Tasks: Augmentation Curriculum Scheduling

- [x] Add YAML schema support for `custom.augmentation.curriculum` (monotonic step/epoch boundaries, op/field validation, non-negative probs, ascending ranges) with strict startup failure on invalid configs.
- [x] Implement step-driven scheduler that linearly interpolates `bypass_prob` and numeric op fields (prob, scalars, ranges) between phase targets; holds final values after the last phase.
- [x] Add TrainerCallback + shared-state propagation so all ranks/workers read identical effective params each step; mutate existing op objects without rebuilding dataloaders.
- [x] Update docs (docs/AUGMENTATION.md and config examples) to describe curriculum usage, linear ramps, and constraints.
- [x] Add tests: scheduler selection/interpolation/validation, propagation sanity across workers, and a smoke test covering phase transition correctness without extra logging.

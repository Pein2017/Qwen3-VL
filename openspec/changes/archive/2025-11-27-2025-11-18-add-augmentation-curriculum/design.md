# Design: Augmentation Curriculum Scheduling

## Current State
- Augmentations configured once via `custom.augmentation` → `Compose` pipeline → `AugmentationPreprocessor` in `DenseCaptionDataset` (train only).
- Dataset RNG seeded per epoch; dataloader workers get copies of the dataset/pipeline (persistent workers on by default).
- No hook to change augmentation parameters mid-training.

## Proposed Approach
1. **Config Schema**
   - Optional `custom.augmentation.curriculum`: ordered phases keyed by `until_percent` (0–1 or 0–100; preferred) or `until_step`. Phases define target `bypass_prob` and per-op overrides for numeric fields (`prob`, scalars, numeric ranges like `[lo, hi]`, `max_deg`). Non-numeric fields are ignored for scheduling. Mixing step and percent in one config is disallowed.
   - Linear interpolation between the previous phase target and current phase target over the step interval. Piecewise-constant behavior can be expressed by repeating targets across adjacent phases.
   - After the last phase, hold final target values.
2. **Scheduler Object**
   - Pure helper that, given global step (and total steps when using percentages), selects the active phase and computes interpolated effective values for `bypass_prob` and each op field present in overrides. Leaves unspecified fields at their base values.
   - Validates monotonic boundaries (after resolving percentages to steps), non-negative probabilities, ascending ranges, and referenced op names/fields existing in the base augmentation config.
3. **Propagation Strategy**
   - Instantiate the scheduler at trainer setup. Maintain a small shared, read-only state (e.g., multiprocessing-safe dict/Value) storing current effective params.
   - A TrainerCallback (rank 0) resolves percent milestones to steps using trainer-reported `max_steps`, then updates the shared state each step or when entering a new phase. Workers read the state inside `__getitem__`/preprocessor and apply effective params to existing op objects (mutating `prob`/range fields) without rebuilding Compose. No dataloader rebuild unless drift is observed.
4. **Determinism & Safety**
   - Phase selection driven by trainer-reported global step to avoid rank drift; workers do not derive phase locally from their own RNG.
   - Strict startup validation; invalid configs abort before training. Runtime logging remains minimal (phase changes need not emit unless debug).
5. **Logging/Debugging**
   - Default: no extra logs beyond a single startup acknowledgment. Optional debug mode may log phase transitions for troubleshooting but is off by default.

## Alternatives Considered
- **Dataloader Rebuild per Phase**: simpler propagation but restarts workers; avoided to keep training steady.
- **Per-batch Compose rebuild**: too slow and risks nondeterminism; rejected.

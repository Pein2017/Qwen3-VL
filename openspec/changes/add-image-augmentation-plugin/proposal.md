## Why
Image overfitting is observed on narrow domains; we need configurable, image-level augmentations to improve generalization without entangling dataset logic.

## What Changes
- Introduce a plugin-style image augmentation module under `src/datasets/augmentation/` with a minimal, typed interface.
- Add config-driven registry to enable/disable built-in ops (flip/rotate/scale/color) and third-party plugins.
- Integrate augmentation at the preprocessor layer (`AugmentationPreprocessor`) with explicit RNG injection.
- Provide validation and fail-fast behavior for geometry consistency after transforms.

## Impact
- Affected specs: `data-augmentation`
- Affected code: `src/datasets/augment.py`, `src/datasets/preprocessors/augmentation.py`, new `src/datasets/augmentation/` plugin registry

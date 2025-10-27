---
title: Robust geometry transforms for augmentation (bbox→quad, clipping, CW order)
author: core
created: 2025-10-27
change-id: robust-geometry-aug-2025-10-27
status: completed
completed: 2025-10-27
---

## Why
Rotation augmentation caused visual quad mismatches due to canvas cropping while geometry was clamped. This led to training on incorrect object annotations after geometric augmentations.

## What Changes
- ✅ **Pre-flush hook mechanism**: Barriers can modify affine matrix and canvas dimensions before warping
- ✅ **Canvas expansion**: `ExpandToFitAffine` computes AABB of rotated corners, translates to top-left origin, pads to multiple of 32
- ✅ **Pixel limit safety**: Enforces max_pixels (default 921,600 for Qwen3-VL), scales down proportionally if exceeded
- ✅ **Code cleanup**: Removed ~360 lines of duplicate geometry code from affine ops
- ✅ **Centralized logging**: Integrated with `src/utils/logger.py` for rank-aware distributed training
- ✅ **Protocol documentation**: Comprehensive docstrings explaining pre-flush hook contract
- ✅ **Test coverage**: 18 tests including rotation expansion and pixel limit enforcement

## Impact
- **Affected specs**: `specs/augmentation-geometry/spec.md`
- **Affected code**:
  - `src/datasets/augmentation/base.py` - Pre-flush hook support in Compose
  - `src/datasets/augmentation/ops.py` - ExpandToFitAffine with safety checks, simplified affine ops
  - `src/datasets/geometry.py` - Typed value objects (BBox, Quad, Polyline) and transform_geometry()
  - `configs/stage_3_*.yaml` - Added expand_to_fit_affine in augmentation pipeline
  - `tests/` - Comprehensive test coverage

## Results
- **Visual alignment**: Quads now match rotated images perfectly (verified via visualization)
- **No cropping**: Full rotated content visible in expanded canvas
- **Safety**: Automatic scaling prevents OOM from extreme expansions
- **Code quality**: ~225 net lines removed while adding functionality
- **Performance**: All 18 tests pass; no significant slowdown

## Non-goals
- No change to template normalization or training loss; strictly data geometry.

## Risks Mitigated
- OOM from extreme rotations → Pixel limit with proportional scaling
- Performance hit from polygon clipping → Negligible in practice (<5%)
- Code duplication → Eliminated via centralized transform_geometry()



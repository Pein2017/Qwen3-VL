## Context
Augmentation geometry drift occurs when image warps and geometry transforms diverge (pivot, matrix, rounding, clipping). The primary issue was rotation causing canvas cropping while geometry was only clamped, creating visual mismatches in training data.

## Goals / Non-Goals
- Goals: Canvas expansion for rotations; pixel limit safety; eliminate code duplication; centralized logging; comprehensive documentation
- Non-Goals: Changing training loss or template normalization; debug IoU tooling (deferred)

## Decisions

### 1. Pre-Flush Hook Architecture ⭐
**Decision**: Add optional `pre_flush_hook(M_total, width, height, rng)` to barrier ops that runs BEFORE affine flush.

**Why**: Allows barrier ops to modify the accumulated affine matrix and canvas dimensions before warping occurs, enabling canvas expansion without retroactive image manipulation.

**Implementation**:
```python
# In Compose.apply() - base.py:94-97
if hasattr(op, 'pre_flush_hook'):
    M_total, current_width, current_height = op.pre_flush_hook(
        M_total, current_width, current_height, rng
    )
_flush_affine()  # Then warp with updated M and dimensions
```

**Alternatives considered**:
- Post-warp expansion: Rejected - would require re-warping images
- Inline in Rotate op: Rejected - breaks separation of concerns
- New expansion op without hook: Rejected - can't access accumulated M_total

### 2. Canvas Expansion with Top-Left Translation
**Decision**: Compute AABB of corners under M_total, translate by `(-minX, -minY)` to shift to non-negative coordinates.

**Why**: Top-left alignment avoids rounding drift from centering and maintains simple coordinate space.

**Algorithm**:
```python
corners = [0, 0, W-1, 0, W-1, H-1, 0, H-1]
transformed = apply_affine(corners, M_total)
minX, minY, maxX, maxY = points_to_xyxy(transformed)
T = translate(-minX, -minY)
M_total_updated = compose_affine(T, M_total)
new_width = ceil(maxX - minX + 1)
new_height = ceil(maxY - minY + 1)
```

### 3. Pixel Limit with Proportional Scaling
**Decision**: Enforce `max_pixels` (default 921,600 = 960×960 for Qwen3-VL). If exceeded, scale down proportionally while maintaining aspect ratio.

**Why**: 
- Prevents OOM from extreme rotations or large images
- Qwen3-VL training has strict memory constraints
- Proportional scaling maintains visual quality better than cropping

**Implementation**:
```python
if new_width * new_height > max_pixels:
    scale_factor = sqrt(max_pixels / (new_width * new_height))
    S = scale_matrix(scale_factor, scale_factor)
    M_total = compose_affine(S, M_total)
    new_width = int(new_width * scale_factor)
    new_height = int(new_height * scale_factor)
    logger.warning(...)  # Actionable warning for user
```

**Alternatives**:
- Skip expansion: Rejected - defeats purpose (cropping returns)
- Fail-fast exception: Rejected - crashes training mid-run
- Adaptive rotation reduction: Rejected - too complex, changes augmentation intent

### 4. Affine Ops Simplification
**Decision**: Remove `apply()` methods from HFlip, VFlip, Rotate, Scale. Keep only `affine()`.

**Why**: `Compose` already handles geometry via centralized `transform_geometry()`. The 90-100 line `apply()` methods per op were dead code (never called by Compose).

**Result**: ~360 lines of duplicate geometry code removed.

### 5. Centralized Logging
**Decision**: Use `src/utils/logger.py` with `get_logger("augmentation.expand")` for rank-aware logging.

**Why**:
- Integrates with ms-swift's logging infrastructure
- Only rank 0 logs by default in distributed training
- Provides actionable warnings with dimensions and scale factors

### 6. Protocol Documentation
**Decision**: Comprehensive docstrings in `ImageAugmenter` protocol explaining all three op types and pre-flush hook contract.

**Why**: Enables future developers to create custom barrier ops without reverse-engineering.

**Content**:
- 3 operator types (affine, color, barrier)
- Pre-flush hook signature and execution flow
- Example use case (ExpandToFitAffine)
- Constraints and requirements

## Risks / Trade-offs

### Canvas Size Growth
**Risk**: Very large rotations could create unwieldy canvases.
**Mitigation**: Pixel limit with proportional scaling; warning messages suggest reducing augmentation strength.

### Multiple-of-32 Alignment After Scaling
**Risk**: Padding to 32 after scaling might slightly exceed pixel limit.
**Mitigation**: Acceptable - overhead is at most `32 * max(W,H)` pixels (~30K), negligible compared to 921K limit.

### Performance
**Risk**: Additional AABB computation and matrix composition per barrier.
**Mitigation**: Negligible - O(1) operations; tests show <1% overhead.

## Migration Plan

### Stage 1: Core Implementation ✅
- Add pre-flush hook support to Compose
- Implement ExpandToFitAffine with pixel limit
- Simplify affine ops
- Add centralized logging

### Stage 2: Configuration Updates ✅
- Add `expand_to_fit_affine` to stage_3 configs after geometric ops
- Remove redundant `pad_to_multiple` at end (now handled by expand)

### Stage 3: Testing & Validation ✅
- Unit tests for rotation + expansion
- Mixed affines test (rotate + scale + flip)
- Pixel limit enforcement test
- Visualization verification

### Rollback Plan
If issues arise:
1. Remove `expand_to_fit_affine` from configs
2. Re-add final `pad_to_multiple`
3. Original behavior restored (with cropping)

## Open Questions - RESOLVED

### Q: Should expansion be opt-in or default?
**A**: Opt-in via config. Not all use cases need it; users can add to their pipeline.

### Q: What if pixel limit is too restrictive?
**A**: Users can override via `max_pixels` parameter. Default of 921,600 works for most Qwen3-VL training scenarios. Warnings guide users to adjust.

### Q: Debug IoU tooling?
**A**: Deferred. Visual verification sufficient for now. Can add in future if needed.

## Implementation Summary

**Files Modified**:
- `src/datasets/augmentation/base.py` (+51 lines)
- `src/datasets/augmentation/ops.py` (-360 lines, +90 lines for ExpandToFitAffine improvements)
- `configs/stage_3_*.yaml` (+3 lines)
- `tests/test_augmentation_geometry.py` (+60 lines)

**Net Change**: ~225 lines removed with expanded functionality

**Test Coverage**: 18 tests, all passing

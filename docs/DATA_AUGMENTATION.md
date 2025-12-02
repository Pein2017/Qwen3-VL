# Data Augmentation Guide

This document describes the robust geometry-preserving augmentation system for Qwen3-VL training.

## Overview

The augmentation pipeline handles 3 geometry types (bbox, poly, polyline) with proper coordinate transforms, clipping, and canvas expansion to prevent cropping artifacts during rotation.

### Source-Aware Augmentation Control

- **Target-domain wrappers (e.g., `bbu`, `rru`)** attach the configured augmentation pipeline and curriculum scheduler. Every record flowing through those datasets is eligible for geometry/color transforms.
- **Source-domain wrappers (e.g., `coco`, `lvis`, `objects365`, `flickr3k`)** default to clean images— the fusion loader simply omits the augmentation preprocessor for those datasets so auxiliary samples stay untouched.
- Wrappers can override this behavior via `params.augmentation_enabled`/`params.curriculum_enabled` if a particular dataset should opt into augmentation.

## Key Features

### ✅ **Robust Rotation Without Cropping**
- Canvas automatically expands to contain full rotated image
- Geometry polygons align perfectly with visual content
- No training on partially-cropped objects

### ✅ **Safety Limits**
- Enforces max pixel count (921,600 by default for Qwen3-VL)
- Proportional scaling when limits exceeded (recomputed after 32× alignment to guarantee the pixel cap)
- Actionable warnings guide configuration tuning

### ✅ **Efficient Implementation**
- Accumulates affine transforms for single-warp efficiency
- Centralized geometry transform logic (no duplication)
- <1% performance overhead

### ✅ **Smart Cropping with Label Filtering** 🆕 *(v1.1 - Oct 2025)*
- Automatic object filtering based on visibility (min_coverage threshold)
- Geometry truncation at crop boundaries (bbox/poly/line)
- Polygon-aware coverage for polygons (Sutherland–Hodgman clipping plus shoelace area computation) to avoid elongated false positives
- Completeness field updates: `显示完整` → `只显示部分` for partially visible objects (structured metadata supported)
- Skip conditions: preserves dense scenes (<4 objects) and line objects (cables/fibers)
- Perfect visual-label alignment for dense detection captioning

## Recent Updates

### October 2025: Middle Gray Padding (v1.1.2)
**Status**: ✅ Deployed  
**Change ID**: `2025-10-27-middle-gray-padding`

**What Changed**:
- Changed padding color from black (0,0,0) to middle gray (128,128,128)
- Applies to: `_pad_to_multiple()`, `Image.transform()` affine warps, canvas expansion
- Achieves zero in normalized space after Qwen3-VL's symmetric normalization

**Why This Matters**:
- Black padding normalized to -1.0, creating artificial boundaries that harm training
- Middle gray (128) normalizes to ~0, appearing neutral to the model
- Minimizes distribution shift and prevents the model from learning spurious edge artifacts
- Improves training stability when using aggressive rotation/expansion augmentation

**Impact**: Better model generalization, especially with rotation and canvas expansion operations.

---

### October 2025: Smart Crop with Label Filtering (v1.1)
**Status**: ✅ Deployed  
**Change ID**: `2025-10-27-add-smart-crop-with-label-filter`

**What Changed**:
- Added `RandomCrop` operator with automatic label filtering and geometry truncation
- Removed redundant operators: `CenterCrop` (use RandomCrop with fixed scale), `Equalize` (use AutoContrast)
- Fixed polygon rotation preservation (no unnecessary AABB conversion for polygons inside canvas)
- Enhanced visualization script with extreme testing mode for all augmentation operations

**v1.1.1 Patch (Same Day)**: Refined polygon truncation for rotate+crop scenarios
- Added `simplify_polygon()` to remove redundant vertices from axis-aligned clipping
- Added `choose_four_corners()` to select true polygon corners before AABB fallback
- Result: Rotated polygons maintain rotation after crop, with accurate boundary intersections

**Key Capabilities**:
1. **Coverage-based filtering**: Objects <30% visible are dropped from GT
2. **Geometry truncation**: Bbox/poly/line clipped to crop boundaries
3. **Completeness tracking**: Automatic `显示完整` ↔ `只显示部分` updates based on coverage (including `attributes.completeness` when present)
4. **Smart skip conditions**: Preserves dense scenes (<4 objects) and line objects; emits telemetry on skip reasons

**Configuration Example**:
```yaml
- name: random_crop
  params:
    scale: [0.7, 1.0]               # Crop 70-100% of image
    min_coverage: 0.3               # Drop if <30% visible
    completeness_threshold: 0.95    # Mark partial if <95% visible
    min_objects: 4                  # Skip crop if <4 objects
    skip_if_line: true              # Preserve cables/fibers
    prob: 0.3                       # 30% of samples
```

**Impact**: Eliminates visual-label misalignment for dense captioning tasks.

---

### October 2025: Telemetry & Validation Enhancements (v1.1.3)
**Status**: ✅ Deployed  
**Change ID**: `2025-10-28-dense-augmentation-telemetry`

- **What Changed**:
- Pixel-cap enforcement re-runs after 32× alignment and rounds down to the nearest legal multiple, guaranteeing `max_pixels` is never exceeded.
- RandomCrop now measures polygons via polygon clipping (shoelace area computation) rather than AABB overlap.
- Completeness updates propagate to structured metadata fields (`attributes.completeness`, `attributes.完整性`), with debug logging when replacements are skipped.
- Compose skips redundant identity warps; telemetry captures padding ratios, crop skip reasons, and kept-object coverage summaries via logger `augmentation.telemetry`.
- Regression tests cover pixel-cap enforcement (including identity alignment) and polygon coverage edge cases.

**Impact**:
- Prevents neutral padding from exceeding the token budget and diluting supervision.
- Dense-caption labels drop nearly invisible objects, reducing hallucinations.
- Debug/test runs can inspect augmentation telemetry without additional instrumentation.
- Updated visualization tooling (`vis_tools/vis_augment_compare.py`) now overlays per-object captions alongside geometry, matching training captions.

---

## Quick Start

### Basic Configuration

```yaml
custom:
  augmentation:
    enabled: true
    bypass_prob: 0.1  # 10% of samples remain clean (no augmentation)
    ops:
      # Geometric transforms (accumulated into single warp)
      - name: hflip
        params: { prob: 0.5 }
      - name: rotate
        params: { max_deg: 25.0, prob: 0.4 }
      - name: scale
        params: { lo: 1.1, hi: 1.4, prob: 0.25 }
      
      # Resolution changes (barrier, changes canvas size)
      - name: resize_by_scale
        params: { lo: 0.75, hi: 1.2, align_multiple: 32, prob: 0.7 }
      
      # Color transforms (deferred, no geometry changes)
      - name: color_jitter
        params: { brightness: [0.75, 1.25], contrast: [0.75, 1.25], prob: 0.5 }
      - name: gamma
        params: { gamma: [0.8, 1.3], prob: 0.3 }

      # Small-object recall (single-image crop→scale→translate for tiny targets)
      - name: small_object_zoom_paste
        params:
          prob: 0.2
          max_targets: 1
          scale: [1.4, 1.8]
          max_size: 96          # bbox long side threshold
          max_line_length: 128  # line length threshold
          overlap_threshold: 0.1
          max_attempts: 20
          context: 4            # extra pixels around patch
          line_buffer: 4        # buffer when computing IoU for lines
      
      # ✅ Final padding: MUST be last to ensure all images are multiple of 32
      # Applied after all size-changing operations to guarantee final alignment
      - name: expand_to_fit_affine
        params: 
          multiple: 32
          max_pixels: 921600  # 960×960, adjust for your GPU memory
```

### Clean Sample Preservation (`bypass_prob`)

Set `bypass_prob` to preserve a percentage of training samples without any augmentation:

- **`0.1`** (recommended): 10% clean samples - balances diversity with stability
- **`0.0`** (default): All samples augmented
- **`0.15-0.2`**: More conservative, use with aggressive augmentation

Clean samples help prevent overfitting to augmented patterns and maintain model familiarity with original data. Validation data is always clean regardless of this setting.

### Curriculum Scheduling (`custom.augmentation.curriculum`)

You can gradually intensify augmentation by adding a `curriculum` block to your augmentation config. The scheduler operates on training **progress** and linearly interpolates `bypass_prob` plus any numeric operator fields (probabilities or two-element ranges) between phase boundaries. Prefer percent milestones (`until_percent`) so schedules stay stable when batch size or accumulation changes; absolute steps (`until_step`) remain available.

```yaml
custom:
  augmentation:
    enabled: true
    bypass_prob: 0.05
    ops:
      - name: rotate
        params: { max_deg: 8.0, prob: 0.1 }
    curriculum:
      phases:
        - until_percent: 20    # 20% of total training steps
          bypass_prob: 0.1
          ops:
            rotate:
              prob: 0.3
              max_deg: 10.0
        - until_percent: 50    # halfway point
          bypass_prob: 0.2
          ops:
            rotate:
              prob: 0.5
              max_deg: 12.0
```

Each phase declares a target `bypass_prob` and overrides for numeric op parameters. Between milestones, the scheduler interpolates linearly from the previous target to the next, then holds the final targets for all later steps. Repeat targets across consecutive phases for piecewise-constant behavior.

Curriculum validation happens before training: non-monotonic milestones, unknown op names, or overriding non-numeric fields abort training. Operators without numeric parameters are ignored by the scheduler. Percent milestones require total steps from the trainer (resolved automatically); mixing steps and percents in one curriculum is disallowed.

No additional logging is emitted by default; the training loop applies the shared state inside the `AugmentationPreprocessor` so all dataloader workers see the same effective values.

### Order Matters!

1. **Affine ops** (hflip, vflip, rotate, scale) - accumulated
2. **Size-changing barriers** (random_crop, resize_by_scale) - flush affines, change dimensions
3. **Color ops** - applied after size changes (don't affect geometry)
4. **expand_to_fit_affine** - **MUST be last** to ensure final padding to multiple of 32

**Important**: `expand_to_fit_affine` should be placed at the **end** of the pipeline (after all size-changing operations) to guarantee that all images are padded to a multiple of 32. This ensures compatibility with Qwen3-VL's ViT requirements. If placed earlier, subsequent operations like `random_crop` or `resize_by_scale` may break the 32× alignment.

## Operator Types

### 1. Affine Operators (`kind="affine"`)
**Examples**: `HFlip`, `VFlip`, `Rotate`, `Scale`

**How they work**:
- Each op returns a 3×3 affine matrix
- `Compose` accumulates all consecutive affine ops
- Single warp applied when hitting a barrier or end of pipeline
- Geometry transformed via centralized `transform_geometry()`

**Geometry behavior**:
- BBox → Quad under general affines (rotation, shear)
- BBox → BBox under axis-aligned (flips, uniform scale)
- Quads transform all 4 corners, enforcing clockwise order
- Lines clipped with Cohen-Sutherland algorithm

### 2. Color Operators (`kind="color"`)
**Examples**: `ColorJitter`, `Gamma`, `HSV`, `CLAHE`

**How they work**:
- Deferred until after all affines flushed
- Applied to images only (geometry unchanged)
- Efficient: no redundant warps

**Configuration Notes**:
- **CLAHE**: `tile_grid_size` accepts a list `[8, 8]` in YAML (automatically converted to tuple internally). Both list and tuple formats are supported.

### 3. Barrier Operators (`kind="barrier"`)
**Examples**: `ExpandToFitAffine`, `ResizeByScale`, `PadToMultiple`

**How they work**:
- Forces affine flush before and after
- Can change canvas dimensions
- Optional `pre_flush_hook()` to modify affines before warp

### 4. Crop Operators with Label Filtering (`kind="barrier"`)
**Example**: `RandomCrop`

**New in v1.1**: Smart cropping with automatic label filtering and completeness tracking for dense captioning tasks.

**How they work**:
- Crop image to random or center region
- **Filter objects** based on visibility (drop if <30% visible by default)
- **Truncate geometries** of partially visible objects to crop boundary
- **Update completeness field**: `"显示完整"` → `"只显示部分"` for objects <95% visible
- **Skip crop** if <4 objects remain or line objects present (preserves cable/fiber integrity)
- Translate retained geometries to crop-relative coordinates

**Business Rules**:
- `min_coverage` (default: 0.3): Drop objects with <30% area inside crop
- `completeness_threshold` (default: 0.95): Mark "只显示部分" if <95% visible
- `min_objects` (default: 4): Skip crop if <4 objects would remain (dense scenes requirement)
- `skip_if_line` (default: true): Skip crop if any line object present (preserve cable/fiber paths)

**Perfect Visual-Label Alignment**: Only describes objects that are actually visible in the cropped image.

**Configuration Example**:
```yaml
# Random crop for scale variation and small object focus
- name: random_crop
  params:
    scale: [0.7, 1.0]               # Crop 70-100% of image
    aspect_ratio: [0.9, 1.1]        # Nearly square
    min_coverage: 0.3               # Drop if <30% visible
    completeness_threshold: 0.95    # Mark partial if <95% visible
    min_objects: 4                  # Skip crop if <4 objects (dense scenes)
    skip_if_line: true              # Skip if line objects present
    prob: 0.3                       # 30% of samples

```

**When to Use**:
- ✅ Dense captioning with object descriptions
- ✅ Need to focus on small objects via zooming
- ✅ Want perfect visual-label alignment (no hallucinated objects)
- ✅ Have `显示完整`/`只显示部分` completeness fields in your data
- ❌ Single-object detection (use `min_objects=1`)
- ❌ Images with <4 objects total (crop will always skip)

**Metadata Propagation**:
Crop operators store metadata that the preprocessor uses to filter and update objects:
- `last_kept_indices`: Indices of retained objects
- `last_object_coverages`: Coverage ratio [0.0, 1.0] for each retained object
- `allows_geometry_drops`: Flag to relax validation (geometry count can decrease)

## Padding Strategy

### Why Padding Matters

Augmentation operations like rotation, canvas expansion, and alignment to 32-multiples introduce padding areas. The fill color significantly impacts training quality and distribution shift.

### Qwen3-VL Normalization

Qwen3-VL uses **symmetric normalization** that maps RGB [0, 255] to [-1, 1]:
```python
# From preprocessor_config.json
image_mean = [0.5, 0.5, 0.5]
image_std = [0.5, 0.5, 0.5]

# Normalization formula
normalized = (pixel/255 - 0.5) / 0.5 = 2*(pixel/255) - 1
```

### Optimal Padding Color: Middle Gray (128, 128, 128)

**Why this value**:
- **Zero in normalized space**: `(128/255 - 0.5) / 0.5 ≈ 0.003 ≈ 0`
- **Neutral point**: Model sees padding as "neutral" rather than strong negative values
- **Minimal distribution shift**: Avoids artificial high-contrast boundaries
- **Better than black (0,0,0)**: Black normalizes to -1.0, creating strong artifacts
- **Better than white (255,255,255)**: White normalizes to +1.0, equally problematic

**Implementation locations**:
1. `_pad_to_multiple()`: Padding to 32-multiple for ViT requirements
2. `Image.transform()`: Fill color for affine warps (rotation, scale)
3. Canvas expansion: Background for expanded areas

**Visual impact**: Gray padding is visually neutral and doesn't create distracting edges during augmentation visualization.

---

## Canvas Expansion Deep Dive

### Problem
Rotation about image center causes corners to extend beyond original bounds:
```
Original 800×600     Rotated 30°
┌──────────┐        
│          │           ◢────◣
│    ●     │    →      │  ● │  (corners clipped!)
│          │           ◥────◤
└──────────┘        
```

### Solution: Pre-Flush Hook
`ExpandToFitAffine` implements `pre_flush_hook()` that:

1. **Computes AABB** of corners under accumulated affine M_total
2. **Translates** affine by (-minX, -minY) to shift to non-negative coords
3. **Expands** canvas to ceil(maxX-minX+1) × ceil(maxY-minY+1)
4. **Pads** to multiple of 32 (Qwen3-VL requirement)
5. **Scales** proportionally if exceeds max_pixels

```python
corners = [0, 0, W-1, 0, W-1, H-1, 0, H-1]
transformed = apply_affine(corners, M_total)
minX, minY, maxX, maxY = points_to_xyxy(transformed)

# Translate to top-left origin
T = translate(-minX, -minY)
M_total = compose_affine(T, M_total)

# Compute new dimensions
new_W = ceil(maxX - minX + 1)
new_H = ceil(maxY - minY + 1)

# Pad to multiple of 32
new_W = ((new_W + 31) // 32) * 32
new_H = ((new_H + 31) // 32) * 32

# Safety: scale down if too large
if new_W * new_H > max_pixels:
    scale = sqrt(max_pixels / (new_W * new_H))
    M_total = compose_affine(scale_matrix(scale, scale), M_total)
    new_W = int(new_W * scale)
    new_H = int(new_H * scale)
    # Re-align to 32
    new_W = ((new_W + 31) // 32) * 32
    new_H = ((new_H + 31) // 32) * 32
```

## Pixel Limit Safety

### Why Limit Pixels?
- Extreme rotations can cause OOM: 1024×1024 @ 45° → 1448×1448 = 2.1M pixels
- Qwen3-VL training with 2GB VRAM per sample needs constraints
- Default: 921,600 pixels (960×960)

### What Happens When Exceeded?
1. System computes scale_factor = sqrt(max_pixels / pixel_count)
2. Applies scaling to affine matrix via `scale_matrix()`
3. Scales down dimensions proportionally
4. Re-aligns to 32-multiple
5. Logs warning with actionable advice

### Example Warning
```
[WARNING:swift.custom.augmentation.expand] ExpandToFitAffine: Canvas expansion 
(1024×1344 = 1376256 pixels) exceeds max_pixels=921600. Scaling down to 
864×1120 (967680 pixels, factor=0.818). Consider reducing rotation/scale 
augmentation strength.
```

### Tuning max_pixels

**Conservative** (smaller GPUs, safe):
```yaml
- name: expand_to_fit_affine
  params: 
    multiple: 32
    max_pixels: 614400  # 768×800
```

**Moderate** (default, Qwen3-VL):
```yaml
- name: expand_to_fit_affine
  params: 
    multiple: 32
    max_pixels: 921600  # 960×960
```

**Aggressive** (larger GPUs, more detail):
```yaml
- name: expand_to_fit_affine
  params: 
    multiple: 32
    max_pixels: 1474560  # 1184×1248
```

## Geometry Transform Rules

### BBox → Polygon Promotion
Under **general affines** (rotation, shear, non-uniform scale):
- BBox converted to 4-point polygon
- All 4 corners transformed
- Clipped with Sutherland-Hodgman
- Enforced clockwise order

Under **axis-aligned affines** (flips, uniform scale):
- BBox stays BBox
- Coordinates transformed directly
- Min/max envelope maintained

### Polygon Handling
- Flat, even-length list (≥6 values / ≥3 points)
- All vertices transformed by affine
- Clipped to image bounds
- After clipping the pipeline may simplify to stable corners (e.g., choose four representative points or a min-area rectangle) while preserving polygon orientation

### Polyline (Line) Handling
- Variable-length point sequence
- Segment-wise Cohen-Sutherland clipping
- Consecutive duplicates removed
- Dropped if <2 points remain after clipping

## Logging

### Centralized Logger
All augmentation uses `src/utils/logger.py`:
```python
from ...utils.logger import get_logger
logger = get_logger("augmentation.expand")
logger.warning("...")
```

### Distributed Training
- Only rank 0 logs by default (reduces noise)
- Set `QWEN3VL_VERBOSE=1` to see logs from all ranks
- Helpful for debugging augmentation issues

## Performance

### Measurements
- **Affine accumulation**: ~0.1ms overhead per op
- **Canvas expansion AABB**: <0.5ms
- **Geometry transform**: ~0.2ms per 100 objects
- **Total overhead**: <1% of training step time

### Optimization Tips
1. **Padding-only**: Packing is removed; rely on standard padding and adjust `global_max_length` if needed.
2. **Tune max_length**: Reduce `global_max_length` if seeing OOM
3. **Reduce augmentation strength**: Lower rotation angles if seeing scaling warnings
4. **Monitor logs**: Watch for frequent pixel limit warnings

## Testing

### Run Tests
```bash
# From repo root
conda run -n ms python -m pytest tests/augmentation/ tests/test_augmentation_geometry.py -v
```

### Test Coverage
- ✅ Affine accumulation and composition
- ✅ Rotation with canvas expansion
- ✅ Mixed affines (rotate + scale + flip)
- ✅ Pixel limit enforcement
- ✅ Geometry type preservation (bbox/poly/line)
- ✅ Determinism and bounds checking

### Visualization
```bash
# From repo root
conda run -n ms python vis_tools/vis_augment_compare.py
```
Outputs to `vis_out/augment_stage3_exact/` showing original vs augmented with overlaid geometry.

## Troubleshooting

### "Canvas expansion exceeds max_pixels" warnings
**Cause**: Rotation + scale combination creates large canvas
**Fix**: 
1. Reduce `rotate.max_deg` (e.g., 25° → 15°)
2. Reduce `scale.hi` (e.g., 1.4 → 1.2)
3. Increase `max_pixels` if GPU memory allows

### Geometry not aligning with image
**Cause**: Missing `expand_to_fit_affine` after rotation, or placed in wrong position
**Fix**: Add `expand_to_fit_affine` at the **end** of the pipeline (after all size-changing operations):
```yaml
- name: rotate
  params: { max_deg: 25.0, prob: 0.4 }
- name: random_crop
  params: { scale: [0.7, 1.0], prob: 0.3 }
- name: resize_by_scale
  params: { lo: 0.9, hi: 1.1, prob: 0.5 }
# ... color ops ...
- name: expand_to_fit_affine  # MUST be last
  params: { multiple: 32 }
```

**Note**: Placing `expand_to_fit_affine` before size-changing operations (like `random_crop` or `resize_by_scale`) will cause the final image to not be a multiple of 32, breaking ViT requirements.

### OOM during training
**Cause**: max_pixels too high or batch too large
**Fix**:
1. Lower `max_pixels` (921600 → 614400)
2. Reduce `per_device_train_batch_size`
3. Shorten `global_max_length` or clip sequences earlier

### Slow augmentation
**Cause**: Too many barrier ops breaking affine accumulation
**Fix**: Group affine ops together, minimize barriers:
```yaml
# Good: affines accumulated
- hflip
- rotate
- scale
- expand_to_fit_affine  # Single barrier

# Bad: barriers interrupt accumulation
- hflip
- pad_to_multiple       # Barrier
- rotate
- pad_to_multiple       # Barrier (redundant)
```

### Crop always skipping (high skip rate)
**Cause**: Too few objects, line objects present, or min_objects threshold too high

**Symptoms**:
```
[DEBUG] Crop would filter to 2 < 4 objects. Skipping crop.
[DEBUG] Crop region contains line object. Skipping crop to preserve cable/fiber integrity.
```

**Fix**:
1. **For datasets with few objects** (2-3 per image):
   ```yaml
   - name: random_crop
     params:
       min_objects: 2  # Lower threshold (default: 4)
   ```

2. **For datasets with line objects** (cables/fibers):
   ```yaml
   - name: random_crop
     params:
       skip_if_line: false  # Allow line truncation if needed
       # WARNING: May lose cable routing information
   ```

3. **Check your data**: If average object count < min_objects, disable crop or lower threshold

### Too many objects dropped by crop
**Cause**: min_coverage threshold too high or crops too small

**Symptoms**:
```
[DEBUG] Crop applied: 15 → 4 objects (region: [200, 150, 600, 450])
```

**Fix**:
1. **Lower coverage threshold** (more lenient):
   ```yaml
   - name: random_crop
     params:
       min_coverage: 0.2  # Keep objects with >20% visible (default: 0.3)
   ```

2. **Use larger crops** (less aggressive):
   ```yaml
   - name: random_crop
     params:
       scale: [0.8, 1.0]  # Crop 80-100% (default: [0.6, 1.0])
   ```

3. **Monitor logs**: If consistently dropping >50% of objects, tune parameters

### Completeness field not updating
**Cause**: Field name mismatch or coverage above threshold

**Check**:
1. Verify your data uses exact field names: `"显示完整"` and `"只显示部分"`
2. Coverage may be ≥95% (above completeness_threshold):
   ```yaml
   - name: random_crop
     params:
       completeness_threshold: 0.90  # Lower threshold (default: 0.95)
   ```

3. Ensure crop is actually applied (check for skip conditions)

### Visual-label misalignment persists
**Cause**: Using `scale` zoom-in instead of `center_crop`

**Fix**: Replace scale with center_crop:
```yaml
# OLD (causes misalignment):
# - name: scale
#   params: { lo: 1.1, hi: 1.4, prob: 0.25 }

# NEW (perfect alignment):
- name: center_crop
  params:
    scale: 0.75  # 1.33x zoom (1 / 0.75 = 1.33)
    min_coverage: 0.3
    completeness_threshold: 0.95
    min_objects: 4
    skip_if_line: true
    prob: 0.25
```

### Crop with rotation produces unexpected results
**Cause**: Crop applied BEFORE rotation (wrong order), or `expand_to_fit_affine` placed incorrectly

**Fix**: Apply crop AFTER affine transforms, and place `expand_to_fit_affine` at the END:
```yaml
# CORRECT order:
- name: rotate
  params: { max_deg: 25.0, prob: 0.4 }
- name: random_crop           # After affines, before final padding
  params: { scale: [0.7, 1.0], prob: 0.3 }
- name: resize_by_scale
  params: { lo: 0.9, hi: 1.1, prob: 0.5 }
# ... color ops ...
- name: expand_to_fit_affine  # MUST be last for final 32× padding
  params: { multiple: 32 }

# WRONG order (crop before rotation):
# - name: random_crop
# - name: rotate  # Will re-crop rotated content!

# WRONG order (expand_to_fit_affine too early):
# - name: expand_to_fit_affine  # Too early!
# - name: resize_by_scale       # Breaks 32× alignment
```

## Implementation Reference

### Key Files
- `src/datasets/augmentation/base.py` - Compose with pre-flush hook
- `src/datasets/augmentation/ops.py` - Augmentation operators
- `src/datasets/geometry.py` - Typed geometry (BBox, Quad, Polyline) and transform_geometry()
- `configs/stage_3_*.yaml` - Example configurations

### Extension Points
Create custom barrier ops:
```python
@register("my_custom_barrier")
class MyCustomBarrier(ImageAugmenter):
    def __init__(self, param: int = 42):
        self.param = param
        self.kind = "barrier"  # Forces affine flush
    
    def pre_flush_hook(self, M_total, width, height, rng):
        """Optional: modify M and dimensions before warp"""
        # ... your logic ...
        return M_total, new_width, new_height
    
    def apply(self, images, geoms, *, width, height, rng):
        """Required: transform images/geoms after flush"""
        # ... your logic ...
        return images, geoms
```

## References

- **Proposal**: `openspec/changes/2025-10-27-robust-geometry-aug/proposal.md`
- **Design Doc**: `openspec/changes/2025-10-27-robust-geometry-aug/design.md`
- **Spec**: `openspec/changes/2025-10-27-robust-geometry-aug/specs/augmentation-geometry/spec.md`
- **Tests**: `tests/test_augmentation_geometry.py`, `tests/augmentation/`

## Summary

The augmentation system provides:
- ✅ **Correctness**: Geometry aligns perfectly with transformed images
- ✅ **Safety**: Pixel limits prevent OOM with automatic scaling
- ✅ **Performance**: <1% overhead via affine accumulation
- ✅ **Maintainability**: Centralized transform logic, comprehensive tests
- ✅ **Visibility**: Rank-aware logging with actionable warnings

Properly configured, it enables aggressive geometric augmentation without sacrificing annotation quality or training stability.

---

## Appendix A: Crop Quick Reference

### Basic Usage

Replace `scale` zoom-in with smart cropping:

```yaml
# ❌ OLD (broken for dense captioning)
- name: scale
  params: { lo: 1.1, hi: 1.4, prob: 0.25 }

# ✅ NEW (perfect alignment, use RandomCrop with fixed scale for center-crop behavior)
- name: random_crop
  params:
    scale: [0.75, 0.75]          # Fixed 75% = 1.33x zoom
    aspect_ratio: [1.0, 1.0]     # Fixed aspect
    min_coverage: 0.3
    completeness_threshold: 0.95
    min_objects: 4
    skip_if_line: true
    prob: 0.25
```

### Parameter Cheat Sheet

| Parameter | Default | What It Does | Tune If... |
|-----------|---------|--------------|------------|
| **scale** | `[0.6, 1.0]` | Crop size (0.7 = 70% of image) | Images too large/small |
| **aspect_ratio** | `[0.8, 1.2]` | Width/height variation | Need square crops |
| **min_coverage** | `0.3` | Drop if <30% visible | Too many/few dropped |
| **completeness_threshold** | `0.95` | Mark partial if <95% | Completeness field tuning |
| **min_objects** | `4` | Skip crop if <4 remain | Samples being skipped |
| **skip_if_line** | `true` | Skip if line objects present | Cable/fiber integrity |
| **prob** | `1.0` | Crop probability | Sample diversity |

### Common Configurations

**Conservative** (start here):
```yaml
scale: [0.7, 1.0]           # 70-100% crop
min_coverage: 0.3           # Drop <30%
min_objects: 4              # Preserve dense scenes
skip_if_line: true          # Preserve cables
prob: 0.3                   # 30% of samples
```

**Aggressive** (small object focus):
```yaml
scale: [0.5, 0.8]           # 50-80% crop
min_coverage: 0.3           # Drop <30%
min_objects: 2              # Allow sparse scenes
skip_if_line: false         # Allow line truncation
prob: 0.5                   # 50% of samples
```

### Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Too many skipped crops | `min_objects` too high | Lower to 2-3 |
| Objects hallucinated | `min_coverage` too low | Raise to 0.4-0.5 |
| Line objects broken | `skip_if_line=false` | Set to `true` |
| No completeness updates | Threshold too low | Check coverage distribution |

---

## Appendix B: Migration from Scale Zoom-In

### Problem with `scale` (lo > 1.0)

The `scale` operator with `lo > 1.0` performs center crop **without label filtering**:

```
Original: 10 objects (all visible)
After scale=1.3x:
  - Image: 6 objects visible, 4 pushed to edges
  - Labels: Still describes all 10 objects ❌
  - Result: Model hallucinates 4 invisible objects
```

### Migration Steps

**Step 1**: Convert zoom factor to crop scale
```
crop_scale = 1 / zoom_factor
```

Examples:
- `scale lo: 1.2` → `crop scale: 0.83`
- `scale lo: 1.3` → `crop scale: 0.77`
- `scale lo: 1.4` → `crop scale: 0.71`

**Step 2**: Replace with `random_crop` (fixed scale for center-crop behavior)
```yaml
# Before
- name: scale
  params: { lo: 1.3, hi: 1.3, prob: 0.25 }

# After
- name: random_crop
  params:
    scale: [0.77, 0.77]          # 1/1.3 = 0.77
    aspect_ratio: [1.0, 1.0]     # Fixed aspect for center-crop
    min_coverage: 0.3
    min_objects: 4
    skip_if_line: true
    prob: 0.25
```

**Step 3**: Test with visualization
```bash
python vis_tools/vis_augment_compare.py
# Check: object counts match between visual and labels
```

### What's Fixed

✅ **Labels filtered**: GT only describes visible objects  
✅ **Completeness updated**: "显示完整" → "只显示部分" for partial objects  
✅ **Geometry truncated**: Coordinates clipped to crop boundaries  
✅ **No hallucination**: Model trained on aligned data

---

## Recent Updates

### November 2025: Final Padding Placement & Type Safety (v1.1.3)
**Status**: ✅ Deployed  
**Change ID**: `2025-11-19-final-padding-placement`

**What Changed**:
- `expand_to_fit_affine` must now be placed at the **end** of the augmentation pipeline (after all size-changing operations) to guarantee final padding to multiple of 32
- Added explicit integer type conversions for PIL operations (prevents float dimension errors)
- Enhanced CLAHE configuration validation (accepts list or tuple for `tile_grid_size`)

**Why This Matters**:
- Placing `expand_to_fit_affine` before size-changing operations (like `random_crop` or `resize_by_scale`) breaks the 32× alignment requirement
- Final padding ensures all images are compatible with Qwen3-VL's ViT requirements
- Type safety fixes prevent runtime errors from float dimensions in PIL operations

**Migration Guide**:
If your config has `expand_to_fit_affine` before size-changing operations, move it to the end:
```yaml
# OLD (incorrect):
- name: expand_to_fit_affine
  params: { multiple: 32 }
- name: resize_by_scale
  params: { lo: 0.9, hi: 1.1, prob: 0.5 }

# NEW (correct):
- name: resize_by_scale
  params: { lo: 0.9, hi: 1.1, prob: 0.5 }
# ... color ops ...
- name: expand_to_fit_affine  # MUST be last
  params: { multiple: 32 }
```

**Impact**: Ensures consistent 32× alignment across all training samples, preventing dimension-related errors.

---

**Last Updated**: 2025-11-19 (v1.1.3)

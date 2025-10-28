# Changelog

All notable changes to the Qwen3-VL training system are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.1.3] - 2025-10-28

### Fixed - Dense Augmentation Telemetry & Safety

**Change ID**: `2025-10-28-dense-augmentation-telemetry`

#### Summary
Addressed two dense-caption regressions by enforcing canvas pixel limits post-alignment and by replacing AABB coverage heuristics with exact polygon clipping. Added telemetry hooks to troubleshoot crop skips, padding ratios, and completeness updates.

#### Technical Details
- `ExpandToFitAffine` now recomputes scaling after 32× alignment, floors to legal multiples, updates padding ratio telemetry, and skips redundant resampling for identity transforms.
- `compute_polygon_coverage` performs Sutherland–Hodgman clipping + shoelace area for quads, while `RandomCrop` records coverage, skip reasons, and structured completeness metadata updates.
- `Compose` avoids unnecessary warps when the affine matrix is identity and surfaces image size/padding data for downstream consumers.
- Added debugging logs in `apply_augmentations` for retained indices, coverage histograms, skip reasons, and padding ratios under logger `augmentation.telemetry`.
- Regression tests: ensured polygon coverage thresholds and pixel-cap enforcement (including identity-alignment scaling) are exercised.

#### Files Changed
- `src/datasets/geometry.py`
- `src/datasets/augmentation/ops.py`
- `src/datasets/augmentation/base.py`
- `src/datasets/preprocessors/augmentation.py`
- `src/datasets/augment.py`
- `tests/augmentation/test_crop_coverage.py`
- `tests/test_augmentation_geometry.py`
- `docs/AUGMENTATION.md`

#### Impact
- ✅ Canvas expansion never exceeds configured `max_pixels`
- ✅ Dense crops drop near-invisible objects and update completeness metadata reliably
- ✅ Telemetry exposes skip reasons, padding ratios, and coverage stats for tuning
- ✅ Regression tests guard against future regressions in coverage and pixel limits

---

## [1.1.2] - 2025-10-27

### Changed - Middle Gray Padding Strategy

**Change ID**: `2025-10-27-middle-gray-padding`

#### Summary
Changed padding color from black (0,0,0) to middle gray (128,128,128) to achieve zero in Qwen3-VL's normalized space, minimizing distribution shift during augmentation.

#### Technical Details
- **Affected operations**: `_pad_to_multiple()`, `Image.transform()` affine warps, canvas expansion
- **Normalization context**: Qwen3-VL uses symmetric normalization `(pixel/255 - 0.5) / 0.5`
- **Rationale**: 
  - Black (0) normalizes to -1.0, creating artificial high-contrast boundaries
  - Middle gray (128) normalizes to ~0.003 ≈ 0, appearing neutral to the model
  - Prevents model from learning spurious edge artifacts at padding boundaries
  
#### Files Changed
- `src/datasets/augmentation/ops.py`: Updated `_pad_to_multiple()` canvas creation
- `src/datasets/augmentation/base.py`: Added `fillcolor=(128, 128, 128)` to `Image.transform()`

#### Impact
- ✅ Reduced distribution shift from padding areas
- ✅ Improved training stability with rotation/expansion augmentation
- ✅ Better model generalization (no spurious edge detection)
- ✅ Visually neutral padding in augmentation visualizations

---

## [1.1.0] - 2025-10-27

### Added - Smart Cropping with Label Filtering

**Change ID**: `2025-10-27-add-smart-crop-with-label-filter`  
**OpenSpec**: Archived as `2025-10-27-2025-10-27-add-smart-crop-with-label-filter`

#### New Features
- **RandomCrop Operator** 🆕
  - Random region selection with scale [0.7, 1.0] and aspect ratio control
  - Automatic object filtering based on visibility (min_coverage=30%)
  - Geometry truncation at crop boundaries for bbox/quad/line types
  - Completeness field updates: `显示完整` → `只显示部分` for partially visible objects
  - Skip conditions: <4 objects (preserves dense scenes) or line objects present (preserves cable/fiber integrity)
  - Metadata propagation through pipeline for preprocessor integration

- **Coverage Utilities** (`src/datasets/geometry.py`)
  - `get_aabb()`: Extract bounding box from any geometry type
  - `intersect_aabb()`: Compute AABB intersection
  - `aabb_area()`: Calculate area
  - `compute_coverage()`: Object visibility ratio in crop region [0.0, 1.0]
  - `translate_geometry()`: Coordinate translation by offset

- **Enhanced Visualization** (`vis_tools/vis_augment_compare.py`)
  - Extreme testing mode with all augmentation operations
  - Comprehensive parameter coverage (geometric, color, effects)
  - Improved output messages showing test scenarios
  - Support for crop visualization and object count tracking

#### Changed
- **Quad Rotation Preservation**
  - Added fast-path for quads fully inside canvas (epsilon=0.5 tolerance)
  - Prevents unnecessary AABB conversion for rotated quads
  - Preserves exact rotation angles when no clipping needed

- **Degenerate Geometry Handling**
  - Improved fallback for collapsed geometries
  - Clamping to image bounds when clipping fails
  - No geometry drops during global rescaling

- **Preprocessor Integration** (`src/datasets/preprocessors/augmentation.py`)
  - Filter objects by crop operator's `last_kept_indices`
  - Update completeness field based on coverage threshold (95%)
  - Ensure single geometry type per object (cleanup redundant fields)
  - Automatic label sync with filtered objects

- **Validation System** (`src/datasets/augment.py`)
  - Conditional validation via `allows_geometry_drops` flag
  - Strict for non-crop ops (preserve all geometries)
  - Relaxed for crop ops (allow filtering)
  - Debug logging for geometry count changes

#### Removed - Redundancy Cleanup 🧹
- **CenterCrop Operator** (~160 lines)
  - **Reason**: Redundant with RandomCrop
  - **Migration**: Use `RandomCrop` with fixed scale `[0.75, 0.75]` and `aspect_ratio: [1.0, 1.0]`
  
- **Equalize Operator** (~14 lines)
  - **Reason**: Redundant with AutoContrast (both global histogram operations)
  - **Migration**: Use `AutoContrast` instead

#### Fixed
- Quad visualization issue (removed incorrect `canonicalize_quad` call)
- Rotated quads appearing as axis-aligned boxes when hitting boundaries
- Geometry count mismatches during augmentation validation

#### Documentation
- Added [CROP_QUICK_REFERENCE.md](CROP_QUICK_REFERENCE.md): Quick reference for crop operators
- Added [MIGRATION_SCALE_TO_CROP.md](MIGRATION_SCALE_TO_CROP.md): Migration guide from scale zoom-in
- Updated [AUGMENTATION.md](AUGMENTATION.md): Comprehensive crop operator documentation
- Added completion report in `openspec/changes/archive/`

#### Configuration
- Updated `configs/stage_3_vision_last6_lora.yaml` with RandomCrop example
- Added `expand_to_fit_affine` after rotation for canvas expansion
- Removed redundant augmentation operators from config examples

#### Metrics
- **Lines Added**: ~485 (including documentation)
- **Lines Removed**: ~175 (redundant operators)
- **Net Change**: +310 lines
- **Files Modified**: 11
- **Linter Errors**: 0

#### Impact
- ✅ Perfect visual-label alignment for dense detection captioning
- ✅ Intelligent geometry handling (truncation + completeness tracking)
- ✅ Cleaner codebase (redundancy removed)
- ✅ Better testing tools (extreme mode visualization)
- ✅ Improved geometry transforms (quad rotation fix)

---

## [1.1.1] - 2025-10-27 (Patch)

### Fixed - Quad Truncation for Rotate+Crop

**Issue**: When a rotated quad was cropped, Sutherland-Hodgman clipping introduced redundant vertices (5-8 vertices for partially cut quads), causing `min_area_rect` fallback to sometimes produce axis-aligned boxes instead of preserving rotated shape.

**Root Cause**:
- Sutherland-Hodgman against axis-aligned crop boundaries adds intersection vertices
- Many vertices were collinear (on crop boundary edges)
- Min-area-rect heuristic would choose AABB over true rotated shape

**Solution**:
Added polygon simplification pipeline:
1. **`simplify_polygon()`**: Remove duplicate and near-collinear vertices (cross-product test, eps=1e-6)
2. **`choose_four_corners()`**: Select 4 most salient corners from simplified polygon
   - Rank by corner strength (normalized cross product)
   - Order clockwise around centroid
   - Applied before `min_area_rect` fallback

**Result**:
- ✅ Rotated quads maintain rotation after crop
- ✅ Inside corners preserve exact rotated coordinates
- ✅ Cropped corners become accurate boundary intersections
- ✅ No spurious AABB conversion

**Code Changes**:
- `src/datasets/geometry.py`: +110 lines (simplification utilities)
- `src/datasets/augmentation/ops.py`: +8 lines (updated quad truncation)

---

## [1.0.0] - Initial Release

### Added
- Core augmentation pipeline with affine accumulation
- Geometry-aware transforms for bbox/quad/line types
- Canvas expansion to prevent rotation cropping
- Safety limits with max pixel enforcement
- Multi-scale training support via `resize_by_scale`
- Color augmentations (jitter, gamma, HSV, CLAHE, etc.)
- Visualization tools for augmentation testing
- Comprehensive documentation

### Features
- **Affine Operators**: HFlip, VFlip, Rotate, Scale
- **Barrier Operators**: ExpandToFitAffine, ResizeByScale, PadToMultiple
- **Color Operators**: ColorJitter, Gamma, HSV, CLAHE, AutoContrast, Solarize, Posterize, Sharpness
- **Geometry Support**: BBox (axis-aligned), Quad (4-corner polygon), Polyline (multi-segment)
- **Clipping Algorithms**: Sutherland-Hodgman (polygon), Cohen-Sutherland (line)

---

## Version Numbering

- **Major** (X.0.0): Breaking changes, architecture shifts
- **Minor** (0.X.0): New features, capabilities
- **Patch** (0.0.X): Bug fixes, refinements

---

## Links

- **Documentation**: [docs/](.)
- **OpenSpec Changes**: [openspec/changes/archive/](../openspec/changes/archive/)
- **Training Configs**: [configs/](../configs/)
- **Source Code**: [src/](../src/)

---

**Last Updated**: 2025-10-27


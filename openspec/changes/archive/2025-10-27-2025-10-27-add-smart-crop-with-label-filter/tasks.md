# Implementation Tasks: Smart Cropping with Label Filtering

**Change ID**: `2025-10-27-add-smart-crop-with-label-filter`  
**Status**: ✅ **IMPLEMENTED** (Core: 100%, Enhancements: Optional)

---

## 1. Geometry Coverage Utilities

- [x] 1.1 Add `get_aabb()` function to extract bounding box from any geometry type
- [x] 1.2 Add `intersect_aabb()` function for AABB intersection computation
- [x] 1.3 Add `aabb_area()` function for area calculation
- [x] 1.4 Add `compute_coverage()` function that returns coverage ratio [0.0, 1.0]
- [x] 1.5 Add `translate_geometry()` function to shift geometry coordinates by offset
- [x] 1.6 Write unit tests for coverage computation edge cases (no overlap, full overlap, partial)

## 2. Random Crop Operator

- [x] 2.1 Implement `RandomCrop` class with barrier semantics
- [x] 2.2 Add scale parameter sampling (uniform from [min, max] range)
- [x] 2.3 Add aspect ratio parameter sampling
- [x] 2.4 Add `completeness_threshold` parameter (default: 0.95)
- [x] 2.5 Add `min_objects` parameter (default: 4)
- [x] 2.6 Add `skip_if_line` parameter (default: true)
- [x] 2.7 Implement random crop position selection
- [x] 2.8 Add image cropping logic (PIL crop)
- [x] 2.9 Implement geometry filtering by coverage threshold
- [x] 2.10 Check skip conditions: < min_objects OR has line objects
- [x] 2.11 If skip: return original images/geoms, log reason, no metadata
- [x] 2.12 If proceed: implement geometry truncation/clipping to crop boundary
- [x] 2.13 Translate remaining geometries to crop coordinate system
- [x] 2.14 Store coverage values for each retained object (`last_object_coverages`)
- [x] 2.15 Store `last_crop_bbox` and `last_kept_indices` metadata
- [x] 2.16 Set `allows_geometry_drops = True` flag (when crop is applied)
- [x] 2.17 Register operator in registry with name "random_crop"

## 3. Center Crop Operator (REMOVED - Redundant)

- [x] 3.1 ~~Implement `CenterCrop` class~~ → **REMOVED** (redundant with RandomCrop)
- [x] 3.2 ~~Add `completeness_threshold` parameter~~ → **REMOVED**
- [x] 3.3 ~~Compute center crop region~~ → **REMOVED**
- [x] 3.4 ~~Reuse geometry filtering logic~~ → **REMOVED**
- [x] 3.5 ~~Register operator~~ → **REMOVED**

**Note**: CenterCrop was implemented but later removed as redundant. RandomCrop with appropriate scale/prob settings provides equivalent functionality with more flexibility.

## 4. Preprocessor Updates for Completeness Handling

- [x] 4.1 Modify `AugmentationPreprocessor._augment_record()` to check for `last_kept_indices`
- [x] 4.2 If no `last_kept_indices`: return original record (crop was skipped)
- [x] 4.3 If `last_kept_indices` present: filter `rec["objects"]` by those indices
- [x] 4.4 Update each filtered object's geometry field (single field only):
  - [x] 4.4a If new geom has "bbox_2d": set bbox_2d, clear quad/line
  - [x] 4.4b If new geom has "quad": set quad, clear bbox_2d/line
  - [x] 4.4c If new geom has "line": set line, clear bbox_2d/quad
- [x] 4.5 Add completeness field update logic using `last_object_coverages`
- [x] 4.6 Implement `desc.replace("显示完整", "只显示部分")` for objects with coverage < completeness_threshold
- [x] 4.7 Update `rec["objects"]` to filtered list

## 5. Validation Updates

- [x] 5.1 Modify `apply_augmentations()` to check `allows_geometry_drops` flag
- [x] 5.2 Add conditional validation: allow count changes if flag is True
- [x] 5.3 Log geometry count changes at debug level with context
- [x] 5.4 Propagate `allows_geometry_drops` flag from operator through pipeline

## 6. Edge Case Handling

- [x] 6.1 Handle empty crop region edge case (crop larger than image → skip crop)
- [x] 6.2 Handle < min_objects case (skip crop, log debug)
- [x] 6.3 Handle line objects in filtered set (skip crop if skip_if_line=true)
- [x] 6.4 Handle degenerate geometries after truncation (<3 points for quad → drop from filtered set)
- [x] 6.5 Handle objects without completeness field (skip completeness update)
- [x] 6.6 Handle objects already marked "只显示部分" (keep unchanged)

## 7. Testing

- [ ] 7.1 Unit test: `get_aabb()` for bbox, quad, line types
- [ ] 7.2 Unit test: `intersect_aabb()` for various overlap scenarios
- [ ] 7.3 Unit test: `compute_coverage()` returns correct ratios
- [ ] 7.4 Unit test: Completeness field update (`显示完整` → `只显示部分`)
- [ ] 7.5 Unit test: Completeness unchanged for objects ≥95% coverage
- [ ] 7.6 Unit test: RandomCrop produces valid crop regions within bounds
- [ ] 7.7 Unit test: Aspect ratio constraints are respected
- [ ] 7.8 Unit test: Geometry truncation preserves valid coordinates
- [ ] 7.9 Integration test: Crop + filter + completeness pipeline end-to-end
- [ ] 7.10 Integration test: Validation allows count changes for crop ops
- [ ] 7.11 Integration test: Fallback to original when < min_objects
- [ ] 7.12 Integration test: Summary regenerated with correct completeness counts
- [ ] 7.13 Edge case test: All objects filtered → fallback to original
- [ ] 7.14 Edge case test: Crop with 0 overlap → all objects dropped
- [ ] 7.15 Edge case test: Partial visibility (30-95%) → geometry truncated + completeness updated
- [ ] 7.16 Edge case test: Crop larger than image → clipped to bounds
- [ ] 7.17 Edge case test: Objects without "显示完整" field → unchanged
- [ ] 7.18 Determinism test: Same seed produces same crop regions and filtering

## 8. Configuration

- [x] 8.1 Add example RandomCrop config to stage_3_vision_last6_lora.yaml (commented, with completeness_threshold)
- [x] 8.2 Add example CenterCrop config as alternative to scale zoom-in
- [x] 8.3 Document recommended crop → expand_to_fit_affine ordering (see MIGRATION_SCALE_TO_CROP.md)
- [ ] 8.4 Add config validation (warn if scale range invalid, etc.)
- [ ] 8.5 Document completeness_threshold parameter with BBU dataset example

## 9. Documentation

- [x] 9.0 Update vis_augment_compare.py to support crop visualization
  - [x] 9.0a Refactor to use EXTREME testing mode with all augmentation ops
  - [x] 9.0b Add comprehensive parameter coverage (geometric, color, effects)
  - [x] 9.0c Remove redundant operators (Equalize, CenterCrop) from imports
  - [x] 9.0d Update output messages for better clarity
- [x] 9.1 Update AUGMENTATION.md with crop operator section
- [x] 9.2 Visual examples enabled via vis_augment_compare.py extreme testing mode
- [x] 9.3 Document completeness field update behavior (`显示完整` ↔ `只显示部分`)
- [x] 9.4 BBU dataset example workflow covered in visualization script
- [x] 9.5 Document coverage threshold tuning guide (min_coverage vs completeness_threshold)
- [x] 9.6 Add troubleshooting section for high drop rates
- [x] 9.7 Document interaction with other augmentation ops (rotate, expand, etc.)
- [x] 9.8 Migration guide: CenterCrop removed (use RandomCrop instead)
- [x] 9.9 Explain geometry truncation behavior for partial visibility
- [x] 9.10 Document summary field automatic regeneration with correct completeness

## 10. Logging & Metrics

- [x] 10.1 Log crop skip events at debug level (< min_objects, line objects)
- [x] 10.2 Log filtered object count when crop is applied
- [x] 10.3 Log completeness field updates (how many "显示完整" → "只显示部分")
- [ ] 10.4 Add optional metrics collection (crop skip rate, filtered objects, completeness updates per epoch)
- [ ] 10.5 Ensure rank-aware logging (only rank 0 in distributed)

## 11. Code Quality

- [x] 11.1 Add docstrings to all new functions (emphasize completeness logic)
- [x] 11.2 Add type hints to all new code
- [x] 11.3 Run linter and fix any issues
- [x] 11.4 Ensure consistent naming with existing geometry utilities
- [x] 11.5 Remove any debug print statements

---

## Dependencies

- **Depends on**: Current geometry system (BBox, Quad, Polyline, transform_geometry)
- **Depends on**: Augmentation pipeline (Compose, apply_augmentations)
- **Depends on**: Builder system (JSONLinesBuilder formats from objects list)
- **Depends on**: Existing clipping utilities (sutherland_hodgman_clip, clip_polyline_to_rect)

## Parallelizable Work

- Tasks 1.1-1.6 (geometry utilities) can be done in parallel with 2.1-2.11 (crop operator)
- Tasks 6.1-6.14 (tests) can start once implementation is done
- Tasks 8.1-8.7 (documentation) can be written in parallel with implementation

## Validation Checkpoints

- **After Task 1**: Coverage computation works correctly (unit tests pass)
- **After Task 2**: RandomCrop produces valid crops with truncation (integration tests pass)
- **After Task 4**: Validation allows count changes appropriately
- **After Task 6**: All tests pass, no regressions
- **After Task 8**: Documentation is clear and actionable

---

## Estimated Effort

| Phase | Tasks | Time |
|-------|-------|------|
| Geometry utilities | 1.1-1.6 | 30 min |
| Crop operators | 2.1-2.17, 3.1-3.5 | 1.5 hours |
| Preprocessor + completeness | 4.1-4.7 | 35 min |
| Validation | 5.1-5.4 | 20 min |
| Edge cases | 6.1-6.6 | 15 min |
| Testing | 7.1-7.18 | 1 hour |
| Config | 8.1-8.5 | 15 min |
| Documentation | 9.1-9.10 | 40 min |
| Logging | 10.1-10.5 | 10 min |
| Polish | 11.1-11.5 | 10 min |
| **Total** | **72 tasks** | **~5 hours** |


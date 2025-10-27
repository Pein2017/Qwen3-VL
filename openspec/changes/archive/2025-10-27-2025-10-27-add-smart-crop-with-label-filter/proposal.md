# Proposal: Smart Cropping with Label Filtering for Dense Captioning

**Change ID**: `2025-10-27-add-smart-crop-with-label-filter`  
**Status**: Proposed  
**Date**: October 27, 2025  
**Author**: AI Assistant

---

## Why

### Problem
Current augmentation preserves all geometries even when `scale(>1.0)` zooms in, causing **visual-label misalignment** in dense captioning tasks:

```
After scale(1.3x) zoom-in:
  Visual: Only 3 objects visible in center region
  Labels: All 11 objects present, including 8 collapsed to edge pixels
  Result: Model learns to "see" invisible objects → hallucination
```

This is particularly problematic for structured scene analysis (BBU installation checks) where:
- Each object has detailed hierarchical captions (类型/属性/条件属性/备注)
- Model must learn precise object-caption associations
- Degenerate edge pixels provide no meaningful training signal

### Current Workarounds
- Remove `scale` zoom-in entirely → loses scale variation
- Use zoom-out only → doesn't help with small object focus
- Keep misaligned labels → trains hallucination behavior

### Opportunity
Proper **random cropping with label filtering** is standard in object detection and provides:
- **Perfect visual-label alignment**: Only describe what's visible
- **Scale variation**: Small objects get focused attention via zooms
- **Robustness**: Model learns to handle partial visibility and occlusion
- **Efficiency**: No wasted compute on degenerate geometries

---

## What Changes

### New Capabilities

**1. Smart Crop Operators**
- `RandomCrop`: Crop from random position with configurable aspect ratios
- `CenterCrop`: Crop from center (replacement for current `scale` zoom-in)
- Both support min/max coverage thresholds and object count constraints

**2. Geometry Visibility Filtering with Completeness Tracking** ⭐ **CRITICAL**
```python
# After crop, automatically filter objects AND update completeness field:
original: 11 objects (8 "显示完整", 3 "只显示部分")
crop region: contains 5 fully, 2 partially (>30% but <95%), 4 outside

Filtering logic:
- 4 objects <30% coverage → DROPPED
- 2 objects 30-95% coverage → KEPT, "显示完整" → "只显示部分"
- 5 objects 95-100% coverage → KEPT, completeness unchanged

filtered: 7 objects retained
completeness updated: 6 now "只显示部分" (2 changed + 3 original)
→ Builder formats only these 7 into JSON with correct completeness
```

**Key Business Logic**: The system intelligently updates the `显示完整`/`只显示部分` field in object descriptions based on crop-induced truncation, ensuring GT labels match visual reality.

**3. Validation Updates**
- Relax strict count preservation for crop operations
- Require logging when objects are dropped
- Add min_objects safeguard (skip sample if too few objects remain)

**4. Configuration Interface**
```yaml
- name: random_crop
  params:
    scale: [0.6, 1.0]               # Crop 60-100% of image
    aspect_ratio: [0.8, 1.2]        # Aspect ratio variation
    min_coverage: 0.3               # Drop objects <30% visible
    completeness_threshold: 0.95    # Mark "只显示部分" if <95% visible
    min_objects: 4                  # Skip crop if <4 objects remain
    skip_if_line: true              # Skip crop if any line object present
    prob: 0.5
```

**Thresholds and Business Rules**:
- `min_coverage` (0.3): Objects <30% visible are dropped from filtered set
- `completeness_threshold` (0.95): Objects <95% visible get "显示完整" → "只显示部分"
- `min_objects` (4): Skip entire crop operation if <4 objects would remain
- `skip_if_line` (true): Skip crop if any filtered object is a line (preserve cable/fiber integrity)
- Objects ≥95% visible keep original completeness field

### Modified Systems
- `src/datasets/augmentation/ops.py`: New crop operators
- `src/datasets/augment.py`: Relax validation, add filtering logic
- `src/datasets/geometry.py`: Add coverage computation utilities
- `configs/*.yaml`: Add crop configs for experimentation

### No Breaking Changes
- Existing augmentation ops unchanged
- Opt-in via config (disabled by default)
- Backward compatible with current pipelines

---

## Impact

### Affected Specs
- **data-augmentation**: ADDED requirements for smart cropping and label filtering

### Affected Code
- `src/datasets/augmentation/ops.py` (~150 lines added)
- `src/datasets/augment.py` (~30 lines modified)
- `src/datasets/geometry.py` (~50 lines added)
- `configs/stage_3_vision_last6_lora.yaml` (example config)

### Dependencies
- Requires current geometry system (BBox, Quad, Polyline)
- Uses existing clipping utilities
- Compatible with expand_to_fit_affine and resize_by_scale

### Testing Strategy
- Unit tests for coverage computation
- Integration tests for crop + filter pipeline
- Edge case tests (0 objects, all outside, partial visibility)
- Visual verification with overlay tool

### Migration Path
**For users NOT using crops**: No action needed (feature disabled by default)

**For users wanting cropping**:
1. Add crop operator to augmentation config
2. Tune coverage thresholds and min_objects
3. Verify alignment with visualization tool
4. Monitor dropped object logs

---

## Alternatives Considered

**1. Keep current scale zoom-in, ignore degenerate labels**
- ❌ Trains hallucination behavior
- ❌ Wastes compute on meaningless geometries

**2. Manual crop at dataset creation time**
- ❌ No variation across epochs
- ❌ Requires regenerating entire dataset
- ❌ Can't tune thresholds dynamically

**3. Filter geometries but keep original captions**
- ❌ Still misaligned (text mentions dropped objects)
- ❌ Confuses model

**4. Use IoU-based filtering with complex logic**
- ❌ Overkill for axis-aligned crops
- ❌ Coverage ratio simpler and sufficient

---

## Success Criteria

### Functional
- ✅ Crop operators produce valid crops with correct aspect ratios
- ✅ Filtered object counts match visible objects in crop
- ✅ Generated JSON only describes visible objects
- ✅ Edge cases handled gracefully (0 objects → skip sample with log)

### Quality
- ✅ No visual-label misalignment (verified with visualization)
- ✅ Model learns to focus on small objects via crops
- ✅ No hallucination of invisible objects

### Performance
- ✅ Filtering overhead <2% of total batch time
- ✅ No memory regressions

### Usability
- ✅ Clear config documentation with examples
- ✅ Actionable logs when samples dropped
- ✅ Easy to disable/tune thresholds

---

## Questions for Review

1. **Coverage threshold**: Default 0.5 (50% visible) or 0.3 (30%)?
2. **Min objects**: Skip sample if <N objects, or allow 0 objects + summary-only?
3. **Aspect ratio**: Fixed 1.0 (square crops) or variable [0.8, 1.2]?
4. **Integration with expand_to_fit_affine**: Disable expansion when cropping, or keep independent?
5. **Crop position**: Purely random, or bias towards object-dense regions?

---

## Timeline Estimate

- **Spec writing**: 30 minutes
- **Core implementation**: 2 hours
  - Crop operators: 1 hour
  - Filtering logic: 30 minutes
  - Validation updates: 30 minutes
- **Testing**: 1 hour
  - Unit tests: 30 minutes
  - Integration tests: 30 minutes
- **Documentation**: 30 minutes
- **Total**: ~4 hours

---

## References

- Current augmentation spec: `openspec/specs/data-augmentation/spec.md`
- Geometry system: `src/datasets/geometry.py`
- Builder system: `src/datasets/builders/jsonlines.py`
- Related change: `2025-10-27-robust-geometry-aug` (canvas expansion)


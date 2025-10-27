# Design: Smart Cropping with Label Filtering

**Change ID**: `2025-10-27-add-smart-crop-with-label-filter`  
**Date**: October 27, 2025

---

## Context

### Current Architecture

```
Augmentation Pipeline (src/datasets/augment.py):
  1. Load images + geometries from JSONL
  2. Apply augmentation ops (rotate, scale, color, etc.)
  3. Transform geometries via transform_geometry()
  4. Validate: len(output_geoms) == len(input_geoms)  ← STRICT
  5. Return augmented record

Builder (src/datasets/builders/jsonlines.py):
  1. Receive record with images + objects list
  2. Format objects into grouped JSON structure
  3. Generate assistant response from objects
  → Perfect coupling: objects list → JSON text
```

### Key Insight
**Text generation is automatic** - builders format whatever's in the `objects` list. So filtering geometries from `objects` automatically filters the generated text. No explicit caption manipulation needed!

### Constraint: Dynamic per-Epoch Augmentation
- Same sample gets different crops each epoch (randomized)
- Cannot pre-generate cropped datasets with fixed captions
- Must filter labels at augmentation time (training pipeline)

---

## Goals / Non-Goals

### Goals
1. **Perfect alignment**: Only describe visible objects after crop
2. **Configurable coverage**: Tune what counts as "visible" (50%? 30%?)
3. **Graceful degradation**: Handle edge cases (0 objects → skip or summary-only)
4. **Simple integration**: Minimal changes to existing pipeline
5. **Performance**: <2% overhead from filtering

### Non-Goals
1. **Complex IoU computation**: Coverage ratio sufficient for axis-aligned crops
2. **Object-aware crop positioning**: Start with uniform random, can enhance later
3. **Partial object descriptions**: Either keep object (if >threshold) or drop entirely
4. **Multi-label handling**: Current system is single object → single caption
5. **Inference-time cropping**: Training-only feature

---

## Decisions

### Decision 1: Coverage-Based Filtering

**Chosen**: Axis-Aligned Bounding Box (AABB) coverage ratio

```python
def compute_coverage(geom, crop_bbox):
    """
    Coverage = (area_inside_crop) / (area_original)
    
    For bbox: Simple AABB intersection
    For quad: Compute AABB of quad, then intersect
    For line: Count points inside crop / total points
    """
    geom_bbox = get_aabb(geom)  # [x1, y1, x2, y2]
    intersection = intersect_aabb(geom_bbox, crop_bbox)
    return area(intersection) / area(geom_bbox)
```

**Alternatives Considered**:
- **Pixel-perfect IoU** (rasterize geometry → mask IoU)
  - ❌ Too slow (need to warp masks for every geometry)
  - ❌ Overkill for simple crops
- **Centroid-based** (keep if centroid inside crop)
  - ❌ Drops large objects partially visible
  - ❌ Keeps tiny objects barely clipped
- **Vertex count** (keep if >N vertices inside)
  - ❌ Doesn't work for bboxes
  - ❌ Inconsistent across geometry types

**Rationale**: AABB coverage is:
- ✅ Fast (pure arithmetic, no rendering)
- ✅ Intuitive (70% coverage = 70% of object visible)
- ✅ Works for all geometry types (bbox, quad, line)
- ✅ Good enough for axis-aligned crops

### Decision 2: Filter at Augmentation Level

**Chosen**: Modify `apply_augmentations()` to return filtered geometries

```python
# src/datasets/augment.py
def apply_augmentations(...):
    out_imgs, out_geoms = pipeline.apply(...)
    
    # NEW: If pipeline has crop metadata, filter geometries
    if hasattr(pipeline, 'last_crop_bbox'):
        out_geoms = filter_by_coverage(
            out_geoms, 
            pipeline.last_crop_bbox, 
            min_coverage=0.5
        )
    
    return images_bytes, out_geoms  # May have fewer geoms now
```

**Alternatives Considered**:
- **Filter in preprocessor** (before augmentation)
  - ❌ Doesn't know crop region until augmentation runs
  - ❌ Wrong layer (preprocessor should be stateless)
- **Filter in builder** (during message construction)
  - ❌ Too late (objects already in record)
  - ❌ Breaks separation of concerns
- **Filter in crop operator's apply()** method
  - ❌ Validation check happens in apply_augmentations, not in ops
  - ✅ Could work, but cleaner to centralize in apply_augmentations

**Rationale**: Filtering in `apply_augmentations()` because:
- ✅ Already handles validation (knows geometry counts)
- ✅ Has access to both original and transformed geometries
- ✅ Centralized logic (all ops benefit, not duplicated)
- ✅ Can log drops with full context

### Decision 3: Relax Validation Conditionally

**Chosen**: Allow count changes only for crop operations

```python
# src/datasets/augment.py (modified validation)
if len(geoms) != len(per_object_geoms):
    # Check if count change is expected (from cropping)
    if not getattr(pipeline, 'allows_geometry_drops', False):
        raise ValueError(...)
    # Crop operation: count change OK, but log it
    logger.debug(f"Crop filtered {len(per_object_geoms)} → {len(geoms)} objects")
```

**Alternatives Considered**:
- **Remove validation entirely**
  - ❌ Loses safety net for bugs in other ops
  - ❌ Silent failures hard to debug
- **Add metadata to each op** (can_drop_geometries flag)
  - ❌ More complex (every op needs metadata)
  - ✅ More flexible (could allow other ops to drop)
- **Separate validation paths** (crop vs non-crop)
  - ❌ Code duplication
  - ❌ Hard to maintain

**Rationale**: Conditional validation because:
- ✅ Preserves safety for existing ops
- ✅ Simple flag check (low overhead)
- ✅ Clear semantics (pipeline declares capability)
- ✅ Easy to extend (other ops can opt-in later)

### Decision 4: Crop Implementation - Barrier Operator

**Chosen**: Crop as `kind="barrier"` with custom apply() logic

```python
@register("random_crop")
class RandomCrop(ImageAugmenter):
    kind = "barrier"
    allows_geometry_drops = True  # Signal to validation
    
    def apply(self, images, geoms, *, width, height, rng):
        # 1. Choose crop region
        crop_w, crop_h = self._sample_size(width, height, rng)
        x, y = self._sample_position(width, height, crop_w, crop_h, rng)
        crop_bbox = [x, y, x + crop_w, y + crop_h]
        
        # 2. Crop images
        out_imgs = [img.crop(crop_bbox) for img in images]
        
        # 3. Translate geometries to new coordinate system
        out_geoms = []
        for g in geoms:
            # Translate by (-x, -y)
            g_translated = translate_geometry(g, -x, -y)
            # Compute coverage
            coverage = compute_coverage(g, crop_bbox)
            if coverage >= self.min_coverage:
                out_geoms.append(g_translated)
        
        # 4. Store crop metadata for validation
        self.last_crop_bbox = crop_bbox
        
        return out_imgs, out_geoms
```

**Alternatives Considered**:
- **Crop as affine transform**
  - ❌ Cropping is not an affine (changes canvas size)
  - ❌ Would need complex flush logic
- **Crop as preprocessing step**
  - ❌ Needs to interact with other augmentations (rotate→crop order matters)
  - ❌ Wrong abstraction layer

**Rationale**: Barrier operator because:
- ✅ Crops change canvas size (flushes affines)
- ✅ Can filter geometries in apply() method
- ✅ Consistent with resize_by_scale pattern
- ✅ Clear order semantics (barriers create checkpoints)

### Decision 5: Edge Case Handling - Skip Crop Operation

**Chosen**: Skip crop operation entirely for edge cases, return original images/geometries

**Skip Conditions**:
1. Filtered objects < 4 (min_objects threshold)
2. Any retained object is a `line` (preserve cable/fiber integrity)
3. Crop region would be invalid (larger than image, etc.)

```python
# In crop operator apply() method:
# After computing filtered geometries
if len(filtered_geoms) < 4:
    logger.debug(f"Crop would filter to {len(filtered_geoms)} < 4 objects. Skipping crop.")
    return images, geoms  # Return original (no crop applied)

# Check for line objects in filtered set
has_line = any("line" in g for g in filtered_geoms)
if has_line:
    logger.debug("Crop region contains line object. Skipping crop to preserve cable/fiber integrity.")
    return images, geoms  # Return original (no crop applied)

# Otherwise proceed with crop
```

**Business Rationale**:
- **min_objects=4**: Dense captioning needs multi-object scenes for learning
- **No line truncation**: Cables/fibers are critical infrastructure; preserve their full paths
- **Skip vs filter**: Simpler than maintaining "crop was attempted" state

**Alternatives Considered**:
- **Fallback to full image** (same effect, more complex state management)
- **Allow line truncation** (❌ loses critical cable routing information)
- **Lower min_objects threshold** (❌ single-object crops not useful for scene understanding)

**Rationale**: Skip-and-preserve because:
- ✅ Simple semantics (crop succeeds or doesn't happen)
- ✅ Preserves training data (sample still used, just not cropped)
- ✅ No geometry count changes (validation stays simple)
- ✅ Business rule enforcement (protect critical objects)

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ Augmentation Pipeline (apply_augmentations)                 │
│                                                              │
│  Input: images, geometries                                  │
│    ↓                                                         │
│  ┌──────────────────────────────────────────────┐           │
│  │ Compose.apply()                              │           │
│  │  - Affine ops (rotate, scale, flip)          │           │
│  │  - Color ops (jitter, gamma, hsv)            │           │
│  │  - Barrier: RandomCrop ←────────────┐        │           │
│  │    → Crops images                   │        │           │
│  │    → Translates geometries          │        │           │
│  │    → Filters by coverage ───────────┘        │           │
│  │    → Sets allows_geometry_drops flag         │           │
│  └──────────────────────────────────────────────┘           │
│    ↓                                                         │
│  Validation (conditional):                                  │
│    if allows_geometry_drops:                                │
│      OK (log count change)                                  │
│    else:                                                    │
│      Strict: len(out) == len(in)                            │
│    ↓                                                         │
│  Output: augmented images, filtered geometries              │
└─────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────┐
│ Builder (JSONLinesBuilder.build_many)                       │
│                                                              │
│  Input: record with filtered objects                        │
│    ↓                                                         │
│  Format objects → grouped JSON                              │
│    {                                                         │
│      "图片_1": {                                            │
│        "object_1": {"bbox_2d": [...], "desc": "..."},       │
│        "object_2": {"quad": [...], "desc": "..."}           │
│      }                                                       │
│    }                                                         │
│    ↓                                                         │
│  Output: messages with aligned visual-label content         │
└─────────────────────────────────────────────────────────────┘
```

---

## API Design

### Geometry Coverage Utilities

```python
# src/datasets/geometry.py

def get_aabb(geom: Dict[str, Any]) -> List[float]:
    """Get axis-aligned bounding box [x1, y1, x2, y2] from any geometry."""
    if "bbox_2d" in geom:
        return geom["bbox_2d"]
    elif "quad" in geom:
        pts = geom["quad"]
        xs = pts[0::2]
        ys = pts[1::2]
        return [min(xs), min(ys), max(xs), max(ys)]
    elif "line" in geom:
        pts = geom["line"]
        xs = pts[0::2]
        ys = pts[1::2]
        return [min(xs), min(ys), max(xs), max(ys)]
    else:
        raise ValueError(f"Unknown geometry type: {geom.keys()}")

def intersect_aabb(bbox_a: List[float], bbox_b: List[float]) -> List[float]:
    """Compute intersection of two AABBs. Returns [0,0,0,0] if no overlap."""
    x1 = max(bbox_a[0], bbox_b[0])
    y1 = max(bbox_a[1], bbox_b[1])
    x2 = min(bbox_a[2], bbox_b[2])
    y2 = min(bbox_a[3], bbox_b[3])
    if x2 <= x1 or y2 <= y1:
        return [0.0, 0.0, 0.0, 0.0]
    return [x1, y1, x2, y2]

def aabb_area(bbox: List[float]) -> float:
    """Compute area of AABB."""
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])

def compute_coverage(geom: Dict[str, Any], crop_bbox: List[float]) -> float:
    """
    Compute fraction of geometry inside crop region.
    
    Returns: coverage ratio in [0.0, 1.0]
    
    Used for two purposes:
    1. Filtering: Drop if < min_coverage (e.g., 0.3)
    2. Completeness: Mark "只显示部分" if < completeness_threshold (e.g., 0.95)
    """
    geom_bbox = get_aabb(geom)
    intersection = intersect_aabb(geom_bbox, crop_bbox)
    geom_area = aabb_area(geom_bbox)
    if geom_area <= 0:
        return 0.0
    return aabb_area(intersection) / geom_area
```

### Crop Operators

```python
# src/datasets/augmentation/ops.py

@register("random_crop")
class RandomCrop(ImageAugmenter):
    """
    Random crop with label filtering and completeness field update.
    
    Business Logic: Updates "显示完整" → "只显示部分" when objects are
    partially truncated by crop boundary (30-95% visible).
    
    Params:
      scale: [min, max] - crop size as fraction of original (default: [0.6, 1.0])
      aspect_ratio: [min, max] - aspect ratio range (default: [0.8, 1.2])
      min_coverage: float - minimum overlap to keep object (default: 0.3)
      completeness_threshold: float - mark partial if < threshold (default: 0.95)
      min_objects: int - skip sample if < N objects remain (default: 1)
      prob: float - probability of applying crop (default: 1.0)
    """
    
    def __init__(
        self,
        scale: List[float] = [0.6, 1.0],
        aspect_ratio: List[float] = [0.8, 1.2],
        min_coverage: float = 0.3,
        completeness_threshold: float = 0.95,
        min_objects: int = 1,
        prob: float = 1.0,
    ):
        self.scale = scale
        self.aspect_ratio = aspect_ratio
        self.min_coverage = min_coverage
        self.completeness_threshold = completeness_threshold
        self.min_objects = min_objects
        self.prob = prob
        self.kind = "barrier"
        self.allows_geometry_drops = True
        self.last_crop_bbox = None
        self.last_kept_indices = None
        self.last_object_coverages = None  # For completeness update
    
    def apply(
        self,
        images: List[Any],
        geoms: List[Dict[str, Any]],
        *,
        width: int,
        height: int,
        rng: Any,
    ) -> Tuple[List[Any], List[Dict[str, Any]]]:
        """Apply random crop and filter geometries by coverage."""
        # Implementation details...


@register("center_crop")
class CenterCrop(ImageAugmenter):
    """
    Center crop (replacement for scale zoom-in) with label filtering and
    completeness field update.
    
    Business Logic: Updates "显示完整" → "只显示部分" when objects are
    partially truncated by crop boundary (30-95% visible).
    
    Params:
      scale: crop size as fraction of original (default: 0.75 = 1.33x zoom)
      min_coverage: float - minimum overlap to keep object (default: 0.3)
      completeness_threshold: float - mark partial if < threshold (default: 0.95)
      min_objects: int - skip sample if < N objects remain (default: 1)
      prob: float - probability of applying crop (default: 1.0)
    """
    # Similar to RandomCrop but fixed center position
```

### Modified Validation

```python
# src/datasets/augment.py (modified)

def apply_augmentations(...) -> Tuple[List[Dict], List[Dict]]:
    ...
    out_imgs, geoms = pipeline.apply(...)
    
    # Check if pipeline allows geometry drops (from crop operations)
    allows_drops = getattr(pipeline, 'allows_geometry_drops', False)
    
    if len(geoms) != len(per_object_geoms):
        if not allows_drops:
            raise ValueError(f"pipeline.apply returned {len(geoms)} geometries, expected {len(per_object_geoms)}")
        # Crop operation: log but allow
        logger.debug(f"Crop filtered {len(per_object_geoms)} → {len(geoms)} objects")
    
    return images_bytes, geoms
```

---

## Preprocessor Integration: Completeness Update and Geometry Cleanup

### Overview

After augmentation, the preprocessor must:
1. Check if cropping was applied (metadata present)
2. Filter objects to only kept indices
3. Update each object's geometry field (ensure **single** field only)
4. Update completeness field based on coverage

### Geometry Field Cleanup Requirement

**Problem**: Augmentation can change geometry types (e.g., bbox → quad after rotation). Objects may end up with **multiple** geometry fields (`bbox_2d`, `quad`, `line`), leading to ambiguity.

**Solution**: After filtering, ensure each object has exactly ONE geometry field by:
- Setting the new geometry type from the augmented result
- Explicitly removing other geometry fields with `.pop()`

### Implementation

```python
# src/datasets/preprocessors/augmentation.py

def _augment_record(self, rec: Dict) -> Optional[Dict]:
    # ... apply augmentation pipeline ...
    images_bytes, per_obj_geoms_new = apply_augmentations(...)
    
    # Check if crop was applied
    kept_indices = getattr(self.pipeline, "last_kept_indices", None)
    if kept_indices is None:
        # No crop (or crop was skipped) - return with updated geometries only
        for obj, new_geom in zip(rec["objects"], per_obj_geoms_new):
            # Still need geometry cleanup for non-crop augmentations!
            _update_geometry_field(obj, new_geom)
        return rec
    
    # Crop was applied - filter objects
    filtered_objects = [rec["objects"][i] for i in kept_indices]
    coverages = getattr(self.pipeline, "last_object_coverages", [])
    completeness_threshold = getattr(self.pipeline, "completeness_threshold", 0.95)
    
    # Update geometries and completeness
    for obj, cov, new_geom in zip(filtered_objects, coverages, per_obj_geoms_new):
        # 1. Update geometry field (single field only!)
        _update_geometry_field(obj, new_geom)
        
        # 2. Update completeness field if below threshold
        if cov < completeness_threshold and "显示完整" in obj.get("desc", ""):
            obj["desc"] = obj["desc"].replace("显示完整", "只显示部分")
    
    rec["objects"] = filtered_objects
    return rec


def _update_geometry_field(obj: Dict, new_geom: Dict) -> None:
    """
    Update object's geometry field, ensuring only ONE geometry type exists.
    
    Critical for downstream consistency - builders expect exactly one of:
    bbox_2d, quad, or line per object.
    """
    if "bbox_2d" in new_geom:
        obj["bbox_2d"] = new_geom["bbox_2d"]
        obj.pop("quad", None)   # Remove if exists
        obj.pop("line", None)   # Remove if exists
    elif "quad" in new_geom:
        obj["quad"] = new_geom["quad"]
        obj.pop("bbox_2d", None)
        obj.pop("line", None)
    elif "line" in new_geom:
        obj["line"] = new_geom["line"]
        obj.pop("bbox_2d", None)
        obj.pop("quad", None)
    else:
        raise ValueError(f"Unknown geometry type in {new_geom.keys()}")
```

### Why This Matters

**Before cleanup (BUG)**:
```python
obj = {
    "bbox_2d": [10, 10, 50, 50],      # Old (from JSONL)
    "quad": [10, 10, 50, 10, ...],    # New (from rotation)
    "desc": "Cable A 显示完整"
}
# ❌ Builder sees both bbox_2d AND quad - undefined behavior!
```

**After cleanup (CORRECT)**:
```python
obj = {
    "quad": [10, 10, 50, 10, ...],    # Only new geometry
    "desc": "Cable A 显示完整"
}
# ✅ Builder sees exactly one geometry type
```

**After crop + completeness update (CORRECT)**:
```python
obj = {
    "quad": [10, 10, 45, 10, ...],    # Truncated + translated
    "desc": "Cable A 只显示部分"      # Updated completeness
}
# ✅ Geometry and description match visual state
```

---

## Data Flow Example

### Before (Current System with scale zoom-in)

```
Input JSONL:
{
  "images": ["img.jpg"],
  "objects": [
    {"bbox_2d": [10,10,50,50], "desc": "对象A"},
    {"bbox_2d": [400,300,450,350], "desc": "对象B"},  // center
    {"bbox_2d": [750,550,790,590], "desc": "对象C"}   // edge
  ],
  "width": 800, "height": 600
}

After scale(1.3x) - zoom into center:
  Images: 800×600 (center region enlarged, edges cropped)
  Objects: [
    {"bbox_2d": [0,0,0,0], "desc": "对象A"},        // degenerate!
    {"bbox_2d": [150,100,250,200], "desc": "对象B"},  // visible
    {"bbox_2d": [799,599,799,599], "desc": "对象C"}   // degenerate!
  ]

Generated JSON (from builder):
{
  "图片_1": {
    "object_1": {"bbox_2d": [0,0,0,0], "desc": "对象A"},  // ← NOT VISIBLE!
    "object_2": {"bbox_2d": [150,100,250,200], "desc": "对象B"},
    "object_3": {"bbox_2d": [799,599,799,599], "desc": "对象C"}  // ← NOT VISIBLE!
  }
}
❌ Misalignment: Model told to see 3 objects, only 1 visible
```

### After (With RandomCrop + Filtering)

```
Input JSONL: (same as above)

After random_crop(scale=[0.6,0.8], min_coverage=0.5):
  Crop region: [200, 150, 600, 450] (center-right, 50% of image)
  
  Coverage computation:
    - 对象A [10,10,50,50]: 0% overlap → DROPPED
    - 对象B [400,300,450,350]: 90% overlap → KEPT
    - 对象C [750,550,790,590]: 0% overlap → DROPPED
  
  Images: 400×300 (cropped region)
  Objects: [
    {"bbox_2d": [200,150,250,200], "desc": "对象B"}  // translated to crop coords
  ]

Generated JSON (from builder):
{
  "图片_1": {
    "object_1": {"bbox_2d": [200,150,250,200], "desc": "对象B"}
  }
}
✅ Perfect alignment: Model sees 1 object, told about 1 object
```

---

## Risks / Trade-offs

### Risk 1: Crop Skip Rate Too High
**Scenario**: Many samples skip crop due to min_objects=4 or line objects

**Mitigation**:
- Log skip rate per epoch ("Crop skipped: 45% (180/400 samples)")
- Provide tuning guide (adjust min_coverage, min_objects, or disable crop for line-heavy datasets)
- Make crop probability configurable (start conservative with prob=0.3)

### Risk 2: Config Complexity
**Mitigation**:
- Provide sensible defaults (scale=[0.6,1.0], min_coverage=0.5)
- Document with examples in AUGMENTATION.md
- Add validation (warn if scale range too small, etc.)

### Risk 3: Interaction with Other Augmentations
**Example**: rotate → crop vs crop → rotate produce different results

**Mitigation**:
- Document ordering recommendations
- Crop should typically come AFTER affine transforms
- Can disable expand_to_fit_affine when using crops (config option)

### Risk 4: Coverage Computation Accuracy
**Limitation**: AABB coverage is approximate for complex geometries
- **Quads**: Rotated quads have larger AABB than actual area
- **Lines**: AABB doesn't detect segments that exit and re-enter crop region

**Mitigation**:
- Good enough for filtering decisions (conservative estimates)
- Lines skip crop entirely (avoid truncation complexity)
- Document limitation in user guide
- Future enhancement: segment-length-based coverage for lines (if needed)

---

## Migration Plan

### Phase 1: Core Implementation (This Change)
1. Add coverage utilities to geometry.py
2. Implement RandomCrop and CenterCrop operators
3. Update validation in augment.py
4. Add tests

### Phase 2: Configuration Tuning (User-Driven)
1. Add crop config to stage_3 experiments
2. Visualize cropped samples
3. Tune coverage/object thresholds
4. Monitor dropped sample rate

### Phase 3: Future Enhancements (Optional)
1. Object-aware crop positioning (bias towards dense regions)
2. Multi-scale cropping (different zoom levels per epoch)
3. Pixel-perfect IoU for complex geometries
4. Crop ensemble (multiple crops per sample)

---

## Open Questions

1. **Default coverage threshold**: 0.5 (50%) or 0.3 (30%)?
   - Recommendation: Start with 0.5, expose as config
   
2. **Min objects policy**: Skip sample or allow 0 objects?
   - Recommendation: Skip by default (min_objects=1), allow override

3. **Crop scale range**: Conservative [0.7, 1.0] or aggressive [0.5, 1.0]?
   - Recommendation: Start conservative, users can increase

4. **Aspect ratio**: Fixed 1.0 or variable [0.8, 1.2]?
   - Recommendation: Variable (more realistic)

5. **Crop position**: Uniform random or object-aware?
   - Recommendation: Uniform for now, object-aware as enhancement

6. **Integration with expand_to_fit_affine**: Auto-disable or independent?
   - Recommendation: Independent (user can choose order)

---

## References

- Object detection crop augmentations: YOLO, Faster R-CNN
- Coverage computation: Similar to IoU but faster for AABBs
- Label filtering: Standard practice in detection/segmentation tasks
- Current geometry system: `src/datasets/geometry.py`
- Builder system: `src/datasets/builders/jsonlines.py`


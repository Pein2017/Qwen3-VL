# Degenerate Quad Issue - Sample 1676

## Problem Summary
Sample index 1676 in `data/bbu_full_768/all_samples.jsonl` contains a **degenerate quad** (polygon with duplicate vertices) that caused a validation error during training.

## Sample Details

**Location:** `data/bbu_full_768/all_samples.jsonl`, line 1676

**Image:** `images/QC-20231218-0025165_4127784.jpeg`
- Width: 1248
- Height: 608

**Problematic Object (object 0):**
```json
{
  "quad": [44, 0, 44, 0, 98, 0, 93, 0],
  "desc": "螺丝、光纤插头/BBU端光纤插头,显示完整,符合要求"
}
```

## Root Cause

The original quad annotation has **duplicate vertices**:
- Points: `[(44, 0), (44, 0), (98, 0), (93, 0)]`
- Unique points: `[(44, 0), (98, 0), (93, 0)]` - only 3 unique points (triangle)

This is a **degenerate quad** - a 3-point polygon (triangle) stored as a 4-point quad with a duplicate vertex.

## Conversion Issue

When converting from `quad` to `poly`:
1. The conversion code removed the duplicate closing point
2. This left only 3 points = 6 coordinates: `[44, 0, 93, 0, 98, 0]`
3. The validation expects at least 8 coordinates (4 points) for a poly
4. **Error:** `ValueError: poly must contain >=8 floats with even length, got 6`

## Fixes Applied

### 1. Degenerate Polygon Detection

**File:** `data_conversion/pipeline/coordinate_manager.py`
**Method:** `_extract_poly_coordinates()`

- Added validation after removing closing point to ensure at least 8 coordinates (4 points)
- Added duplicate vertex detection: checks if polygon has fewer unique points than total points
- Rejects degenerate polygons (triangles stored as quads with duplicate vertices) by returning empty list

### 2. Area Validation

**File:** `data_conversion/pipeline/flexible_taxonomy_processor.py`
**Method:** `_process_geometry()`

- Added area validation using `properties.area` and `properties.selfArea` from raw annotations
- Rejects objects with zero or negative area
- Rejects objects with extremely small area (< 1 pixel²)
- Degenerate polygons that fail poly extraction fall back to `bbox_2d` if bbox is valid, otherwise rejected

**Code change:**
```python
# After removing closing point, validate we still have at least 8 coordinates (4 points)
# If not, this is likely a triangle or invalid polygon - fall back to bbox_2d
if len(raw_coords) < 8:
    logger.warning(
        f"Polygon has less than 4 points after removing closing point: {len(raw_coords)} coordinates. "
        f"Falling back to bbox_2d. Original had {len(raw_coords) + 2} coordinates."
    )
    return []
```

## Impact

- **Before fix:** Degenerate quads (triangles with duplicate vertices) were converted to invalid 6-point polys, causing training errors
- **After fix:** Degenerate quads fall back to `bbox_2d`, preventing validation errors

## Related Files

- Original sample: `data/bbu_full_768/all_samples.jsonl` (line 1676)
- Converted sample: `data/bbu_full_768_poly_test1/train.jsonl` (index 1676)
- Fixed dataset: `data/bbu_full_768_poly_test1/train_fixed.jsonl`

## Recommendation

1. **Immediate:** Use `train_fixed.jsonl` which converts the problematic sample to `bbox_2d`
2. **Long-term:** Re-run data conversion with the fixed code - degenerate quads will automatically fall back to `bbox_2d`
3. **Data quality:** Consider validating source annotations to prevent degenerate quads in the future

# Object Ordering Consistency

## Summary

✅ **FIXED**: Preprocessing ordering now matches prompt specification exactly.

ℹ️ **Legacy compatibility**: pass `--preserve_annotation_order` (or set
`preserve_annotation_order=True` in `DataConversionConfig`) to skip the TLBR
resort when you need byte-identical diffs against historic exports. The default
remains TLBR sorting so prompts and preprocessing stay aligned.

## Ordering Rule

**Both preprocessing and prompts use the same rule:**
- Sort by **Y coordinate first** (top-to-bottom, smallest to largest)
- Then by **X coordinate** (left-to-right, smallest to largest)

## Reference Points by Geometry Type

### 1. `bbox_2d`
- **Prompt**: "使用左上角坐标 (x1, y1)"
- **Preprocessing**: `obj["bbox_2d"][0], obj["bbox_2d"][1]` ✅

### 2. `poly`
- **Prompt**: "使用第一个顶点 (x1, y1)"
- **Preprocessing**: `obj["poly"][0], obj["poly"][1]` ✅

### 3. `line`
- **Prompt**: "使用最左端点（X 坐标最小的点）作为排序参考；若多个点的 X 坐标相同，则取其中 Y 坐标最小的点"
- **Preprocessing**: **FIXED** - Now finds leftmost point (min X, then min Y if tie) ✅
  - **Before**: Used first point `[0], [1]` ❌
  - **After**: Finds leftmost point using `min(points, key=lambda p: (p[0], p[1]))` ✅

## Implementation

**File**: `data_conversion/utils/sorting.py`

```python
def _first_xy(obj: Dict) -> Tuple[int, int]:
    """Get sorting reference point according to prompt specification."""
    if "bbox_2d" in obj:
        return obj["bbox_2d"][0], obj["bbox_2d"][1]  # top-left
    if "poly" in obj:
        return obj["poly"][0], obj["poly"][1]  # first vertex
    if "line" in obj:
        # Find leftmost point (min X, then min Y if tie)
        coords = obj["line"]
        points = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
        leftmost = min(points, key=lambda p: (p[0], p[1]))
        return leftmost[0], leftmost[1]
    return 0, 0

def sort_objects_tlbr(objects: List[Dict]) -> List[Dict]:
    """Sort top-to-bottom, then left-to-right."""
    return sorted(objects, key=lambda o: (_first_xy(o)[1], _first_xy(o)[0]))
```

## Prompt Reference

**File**: `src/config/prompts.py` (lines 41-45)

```
- 对象按"自上到下 → 左到右"排序（线以最左端点为起点），编号从 1 递增。
  * 排序规则详解：首先按 Y 坐标（纵向）从小到大排列（图像上方优先），Y 坐标相同时按 X 坐标（横向）从小到大排列（图像左方优先）。
  * bbox_2d 排序参考点：使用左上角坐标 (x1, y1) 作为该对象的排序位置。
  * poly 排序参考点：使用第一个顶点 (x1, y1) 作为该对象的排序位置；当前样本以 4 个顶点为主，后续可扩展。
  * line 排序参考点：使用最左端点（X 坐标最小的点）作为排序参考；若多个点的 X 坐标相同，则取其中 Y 坐标最小的点。
```

## Where It's Applied

**File**: `data_conversion/pipeline/unified_processor.py` (line 416)

```python
# Sort objects by position using first coordinate pair
objects = sort_objects_tlbr(objects)
```

## Impact

- **Before fix**: Line objects were sorted using first point, causing inconsistent ordering
- **After fix**: Line objects are sorted using leftmost point, matching prompt specification
- **Result**: Preprocessing and prompts now have 100% consistent ordering

## Verification

Run `verify_ordering_consistency.py` to check consistency:
```bash
conda run -n ms python verify_ordering_consistency.py
```

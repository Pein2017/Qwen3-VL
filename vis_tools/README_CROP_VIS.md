# Crop Augmentation Visualization Guide

## Quick Start

The `vis_augment_compare.py` script now supports visualizing crop operations with automatic label filtering.

### Option 1: Visualize Training Config (Recommended)

If your training config includes crop operators, the script will automatically use them:

```bash
# From repo root
conda run -n ms python vis_tools/vis_augment_compare.py
```

Make sure your `config_yaml` path points to a config with crop enabled (e.g., `configs/stage_3_vision_last6_lora.yaml` with crop uncommented).

### Option 2: Test Crop Operators Standalone

Edit `vis_augment_compare.py` main section:

```python
cfg = VisConfig(
    jsonl_path='data/bbu_full_768/train.jsonl',
    out_dir='vis_out/crop_test',
    num_samples=8,
    variants=3,
    seed=2025,
    
    # Disable YAML to use manual config
    config_yaml=None,  # or point to non-existent file
    
    # Enable crop augmentation
    random_crop_p=0.8,           # 80% chance of random crop
    random_crop_scale_lo=0.7,    # Crop 70-100% of image
    random_crop_scale_hi=1.0,
    center_crop_p=0.0,           # Disable center crop for now
    crop_min_coverage=0.3,       # Drop objects <30% visible
    crop_min_objects=4,          # Skip crop if <4 objects remain
    crop_skip_if_line=True,      # Skip crop if line objects present
)
```

Then run:
```bash
conda run -n ms python vis_tools/vis_augment_compare.py
```

## What to Look For

### Output Visualization

Each output image shows:
- **Left panel**: Original image with all objects
- **Right panels**: Augmented variants with filtered objects

**Title format**: `Aug 1: ops|list\n5 objects (-3)`
- Operation list shows applied augmentations
- Object count shows filtered objects
- `(-3)` means 3 objects were dropped by crop

### Console Output

When crop filters objects, you'll see:
```
[CROP] Sample 0, variant 1: 8 → 5 objects
       Kept indices: [0, 2, 3, 5, 7]
       Avg coverage of kept objects: 68.45%
```

This tells you:
- Original object count: 8
- After crop: 5 objects
- Which objects were kept (by index)
- Average visibility of kept objects

### Visual Indicators

- **Object count change**: Title shows `(+/-)` if objects were filtered
- **Cropped region**: You can see smaller image dimensions
- **Missing objects**: Objects near edges are removed
- **Truncated objects**: Partially visible objects have clipped boundaries

## Testing Scenarios

### 1. Aggressive Cropping (More Filtering)

```python
random_crop_p=1.0,
random_crop_scale_lo=0.5,    # Very small crops (50-100%)
random_crop_scale_hi=0.8,
crop_min_coverage=0.5,       # Stricter threshold (50%)
crop_min_objects=2,          # Allow more crops
```

Expect: High object drop rate, many filtered objects

### 2. Conservative Cropping (Less Filtering)

```python
random_crop_p=1.0,
random_crop_scale_lo=0.8,    # Larger crops (80-100%)
random_crop_scale_hi=1.0,
crop_min_coverage=0.2,       # Lenient threshold (20%)
crop_min_objects=4,          # Skip if too few objects
```

Expect: Lower drop rate, most objects retained

### 3. Center Crop (Scale Zoom-In Replacement)

```python
random_crop_p=0.0,           # Disable random crop
center_crop_p=1.0,           # Always apply center crop
center_crop_scale=0.75,      # 75% → 1.33x zoom
crop_min_coverage=0.3,
crop_min_objects=4,
```

Expect: Fixed center region, consistent filtering

### 4. Skip Conditions Testing

Use a dataset with line objects (cables/fibers):

```python
random_crop_p=1.0,
crop_skip_if_line=True,      # Skip if line objects present
crop_min_objects=10,         # Skip if <10 objects (high threshold)
```

Expect: Many crops skipped, console shows skip reasons

## Output Location

Default: `vis_out/augment_stage3_exact/`

Files: `vis_00000.jpg`, `vis_00001.jpg`, ...

## Troubleshooting

### No crops applied
- Check `random_crop_p` or `center_crop_p` > 0
- Check console for skip messages
- Try lowering `crop_min_objects` threshold

### All crops skipped
- Dataset may have too few objects (< min_objects)
- Dataset may have line objects (with skip_if_line=True)
- Lower `crop_min_objects` or set `crop_skip_if_line=False`

### Too many objects dropped
- Increase `crop_scale_lo` (larger crops)
- Lower `crop_min_coverage` (more lenient)
- Check if threshold too strict for your data

### Objects not truncated at boundaries
- This is expected: crop operators clip geometries to crop boundaries
- Check console output for coverage values
- Partially visible objects should have <100% coverage

## Advanced: Comparing Crop Strategies

Create multiple configs to compare:

```bash
# Test 1: Random crop
# (Edit vis_augment_compare.py with random_crop_p=1.0)
conda run -n ms python vis_tools/vis_augment_compare.py
mv vis_out/augment_stage3_exact vis_out/test_random_crop

# Test 2: Center crop
# (Edit with center_crop_p=1.0, random_crop_p=0.0)
conda run -n ms python vis_tools/vis_augment_compare.py
mv vis_out/augment_stage3_exact vis_out/test_center_crop

# Compare side by side
ls -lh vis_out/test_*/
```

## Geometry Evaluation (gt_vs_pred.jsonl)

For objective, offline evaluation of dense captioning/detection dumps (`gt_vs_pred.jsonl`, `norm1000`), use:

```bash
conda run -n ms python vis_tools/eval_dump.py path/to/gt_vs_pred.jsonl \
  --primary-threshold 0.50 \
  --line-tol 8 \
  --output-json output/geometry_eval.json
```

Notes:
- `--output-json` is required (the command fails fast if omitted).
- Only a single input JSONL file is supported for now; evaluate multiple dumps by running the command separately per file.

This reports metrics under three modes:
- `localization`: geometry-only matching
- `phase`: require phase/head label match derived from `desc`
- `category`: require fine category match derived from `desc` (supports legacy `螺丝、光纤插头/...` parsing)

## Example Output Analysis

**Good filtering** (balanced):
```
[CROP] Sample 0, variant 1: 12 → 8 objects
       Avg coverage of kept objects: 75.23%
```
- Kept 67% of objects
- Good average coverage (75%)

**Too aggressive** (losing too much):
```
[CROP] Sample 0, variant 1: 12 → 3 objects
       Avg coverage of kept objects: 45.67%
```
- Kept only 25% of objects
- Low coverage → increase crop size or lower threshold

**Too conservative** (not much filtering):
```
[CROP] Sample 0, variant 1: 12 → 11 objects
       Avg coverage of kept objects: 95.34%
```
- Almost all objects kept
- Very high coverage → consider more aggressive cropping

---

## Integration with Training

Once you've tuned parameters visually, add them to your training config:

```yaml
# configs/stage_3_vision_last6_lora.yaml
custom:
  augmentation:
    ops:
      # ... other ops ...
      - name: random_crop
        params:
          scale: [0.7, 1.0]              # From vis testing
          aspect_ratio: [0.9, 1.1]
          min_coverage: 0.3              # From vis testing
          completeness_threshold: 0.95
          min_objects: 4                 # From vis testing
          skip_if_line: true
          prob: 0.3
```

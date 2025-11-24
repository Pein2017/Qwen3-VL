# LVIS Dataset - Current Status

**Last Updated**: 2025-10-27

## 📥 Download Progress

| Component | Size | Status | Location |
|-----------|------|--------|----------|
| LVIS Train Annotations | 1.1 GB | ✅ **Complete** | `./lvis/raw/annotations/lvis_v1_train.json` |
| LVIS Val Annotations | 192 MB | ✅ **Complete** | `./lvis/raw/annotations/lvis_v1_val.json` |
| COCO Train Images | 18 GB | 🔄 **In Progress** (9.4/18 GB) | `./lvis/raw/images/train2017.zip` |
| COCO Val Images | 1 GB | ⏳ **Pending** | - |

**Total Progress**: ~55% (11.6 / 20.3 GB)

---

## ✅ Completed Tests

All tests pass using annotations only (no images required yet):

```bash
cd /data/public_data
bash tests/run_tests.sh
```

### Test Results

| Test | Status | Description |
|------|--------|-------------|
| Annotation Loading | ✅ PASS | 1203 categories, 1.27M annotations |
| BBox Conversion | ✅ PASS | COCO → Qwen3-VL format |
| Polygon Conversion | ✅ PASS | N-point polygons → quad |
| Format Compliance | ✅ PASS | Qwen3-VL schema validation |

---

## 🎯 Key Findings

### Multi-Polygon Support Validated

**Geometry Types**:
- `bbox_2d`: 2-point implicit rectangle ✅
- `quad`: **N-point closed polygon** (N ≥ 3) ✅

**LVIS Polygon Distribution** (sample of 203 polygons):
- 5-10 points: 11.8%
- 11-20 points: 25.1%
- 21-40 points: 34.0% ⬅️ **Most common**
- 41-70 points: 17.2%
- 71+ points: 11.8%

**Extreme cases**:
- Minimum: 5 points (simple shapes)
- Maximum: 314 points (highly detailed contours)
- Average: ~30 points

### Format Design

**Unified Representation**:
```json
{
  "images": ["train2017/000000391895.jpg"],
  "objects": [
    {"bbox_2d": [x1, y1, x2, y2], "desc": "person"},
    {"quad": [x1,y1,...,xn,yn], "quad_points": n, "desc": "car"}
  ],
  "width": 640,
  "height": 360
}
```

**Key Points**:
- ✅ `quad` = **generic polygon**, not limited to 4 points
- ✅ `quad_points` tracks actual point count
- ✅ Backward compatible with 4-point quads
- ✅ Consistent with `line` + `line_points` pattern

---

## 🚀 Next Steps

### When Download Completes (~50% remaining)

1. **Extract Images** (~10 min)
   ```bash
   cd ./lvis/raw/images
   unzip train2017.zip
   ```

2. **Test Conversion with Images** (~1 min)
   ```bash
   conda run -n ms python scripts/convert_lvis.py --test --split train
   ```

3. **Full Train Conversion** (~10 min for 100K images)
   ```bash
   conda run -n ms python scripts/convert_lvis.py --split train
   ```

4. **Create Training Sample** (~1 min)
   ```bash
   conda run -n ms python scripts/sample_dataset.py \
     --input lvis/processed/train.jsonl \
     --output lvis/processed/samples/train_5k_stratified.jsonl \
     --num_samples 5000 \
     --strategy stratified
   ```

5. **Integrate with Qwen3-VL Training**
   - Update `DATA_AND_DATASETS.md` to document polygon support
   - Relax `quad` validation (accept N ≥ 3 points)
   - Test template serialization

---

## 🔧 Qwen3-VL Integration TODO

### Code Changes Required

**File**: `/data/Qwen3-VL/src/datasets/preprocessors/dense_caption.py`

```python
# Current (4-point only):
assert len(obj["quad"]) == 8, "quad must have 8 values"

# New (N-point polygons):
assert len(obj["quad"]) >= 6 and len(obj["quad"]) % 2 == 0, \
    "quad must have at least 6 values (3 points) and even number"
assert "quad_points" in obj, "quad requires quad_points field"
assert len(obj["quad"]) == obj["quad_points"] * 2, \
    "quad length must match quad_points * 2"
```

**File**: `/data/Qwen3-VL/docs/DATA_AND_DATASETS.md`

Update schema documentation:
```markdown
| Type | Format | Use Case |
|------|--------|----------|
| bbox_2d | [x1,y1,x2,y2] | Axis-aligned boxes |
| quad | [x1,y1,...,xn,yn] + quad_points: n | N-point closed polygons (N≥3) |
| line | [x1,y1,...,xn,yn] + line_points: n | Open polylines |
```

---

## 📊 Dataset Statistics

**LVIS v1.0 Train Split**:
- Images: 100,170
- Annotations: 1,270,141
- Categories: 1,203
  - Frequent (>100 instances): 405 categories
  - Common (10-100 instances): 461 categories
  - Rare (<10 instances): 337 categories
- Avg annotations/image: 12.7
- Long-tail distribution: ✅ Ideal for few-shot learning

---

## 📁 Directory Structure

```
./
├── lvis/
│   ├── raw/
│   │   ├── annotations/          ✅ Complete
│   │   │   ├── lvis_v1_train.json
│   │   │   └── lvis_v1_val.json
│   │   └── images/               🔄 Downloading (52%)
│   │       └── train2017.zip
│   └── processed/                ⏳ Pending conversion
│       ├── train.jsonl
│       ├── val.jsonl
│       └── samples/
├── converters/                   ✅ Ready
│   ├── base.py
│   ├── geometry.py
│   └── lvis_converter.py         (polygon support enabled)
├── scripts/                      ✅ Ready
│   ├── download_lvis.py
│   ├── convert_lvis.py           (--use-polygon flag)
│   ├── sample_dataset.py
│   └── validate_jsonl.py
└── tests/                        ✅ All passing
    ├── README.md
    ├── run_tests.sh
    └── test_lvis_converter.py
```

---

## 🎓 Key Learnings

1. **LVIS is polygon-rich**: Not just bboxes, but detailed vector segmentations
2. **Polygon diversity**: From simple 5-point shapes to complex 300+ point contours
3. **Vector format**: Perfect for V-LLM (no pixel-level masks)
4. **Unified representation**: `quad` as generic polygon simplifies architecture
5. **Backward compatible**: Existing 4-point quads still work

---

## ⏰ Estimated Timeline

- **Now**: Download in progress (~50% complete)
- **+30 min**: Download completes
- **+40 min**: Images extracted
- **+50 min**: Full dataset converted
- **+55 min**: Training samples created
- **+1 hour**: Ready for Qwen3-VL integration

---

## 💡 Quick Commands

```bash
# Check download progress
du -sh ./lvis/raw/images/train2017.zip

# Run tests (no images needed)
cd /data/public_data && bash tests/run_tests.sh

# When ready: convert with polygons
conda run -n ms python scripts/convert_lvis.py \
  --split train --use-polygon

# Validate output
conda run -n ms python scripts/validate_jsonl.py \
  lvis/processed/train.jsonl
```


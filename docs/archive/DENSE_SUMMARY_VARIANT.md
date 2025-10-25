# Dense-Summary Variant Feature (Dynamic Per-Group Mode Selection)

Status: Archived — Superseded by docs/DATA_FORMATS.md and docs/TRAINING_GUIDE.md

## Overview

This feature enables **flexible per-pairing-group prompt selection** for dense captioning training:

1. **Dense-only (default)**: All groups use dense mode (grouped JSON with geometry)
2. **Summary-only**: All groups use summary mode (one-line summaries per image)
3. **Mixed (dynamic)**: Each pairing group randomly selects either dense or summary mode

The key insight: **All samples within a single pairing group share the same mode and system prompt**, enabling clean training with dynamic prompt injection at the dataset level.

## Summary Field Format (Standardized)

The `summary` field in your JSONL follows a **unified, all-slash format**:

```json
{
  "images": ["path/to/img1.jpg"],
  "objects": [
    {"bbox_2d": [x1,y1,x2,y2], "desc": "BBU设备/华为,显示完整,无需安装"},
    {"quad": [x1,y1,...,x4,y4], "desc": "标签/5G-AAU1光纤"}
  ],
  "summary": "BBU设备/华为/显示完整/无需安装×1，螺丝、光纤插头×2，标签/可以识别×1",
  "width": 1920,
  "height": 1080
}
```

### Transformation Rules

The summary is generated from object descriptions by:

1. **Replace commas with slashes** (flatten hierarchical levels)
   - Input desc: `"BBU设备/华为,显示完整,无需安装"`
   - Output: `"BBU设备/华为/显示完整/无需安装"`

2. **Categorize labels as '可以识别' or '无法识别'**
   - `"标签/5G-AAU1光纤"` → `"标签/可以识别"`
   - `"标签/无法识别"` → `"标签/无法识别"` (unchanged)

3. **Group identical items and count** (×N notation)
   - Same captions grouped in object order
   - Count appended: `"item×3"` for 3 occurrences

4. **Preserve object list order** (no reordering)
   - Items appear in the order they were listed in `objects`

5. **Preserve attribute order** from descriptions
   - All hierarchical levels converted to slash-separated
   - Order of attributes within each level maintained

### Examples

**Example 1: Basic Dense Summary**
```json
Objects:
  1. "BBU设备/华为,显示完整,无需安装"
  2. "螺丝、光纤插头/BBU安装螺丝,只显示部分,符合要求"
  3. "标签/5G-BBU光纤"

Summary:
"BBU设备/华为/显示完整/无需安装×1，螺丝、光纤插头/BBU安装螺丝/只显示部分/符合要求×1，标签/可以识别×1"
```

**Example 2: With Counts and Mixed Labels**
```json
Objects:
  1. "光纤/有保护措施,弯曲半径合理/蛇形管"
  2. "光纤/有保护措施,弯曲半径合理/蛇形管"
  3. "标签/无法识别"
  4. "标签/无法识别"

Summary:
"光纤/有保护措施/弯曲半径合理/蛇形管×2，标签/无法识别×2"
```

**Example 3: With Remarks**
```json
Objects:
  1. "BBU设备/华为,只显示部分,无需安装,备注:疑似设备"
  2. "螺丝、光纤插头/BBU安装螺丝,显示完整,符合要求"

Summary:
"BBU设备/华为/只显示部分/无需安装×1，螺丝、光纤插头/BBU安装螺丝/显示完整/符合要求×1，备注:疑似设备"
```

## Backward Compatibility

✅ **Existing configs are fully compatible**. All configs without `summary_ratio` work exactly as before:
- `prompts.scheme: A` or `B` → uses `SYSTEM_PROMPT_A` / `SYSTEM_PROMPT_B` (dense mode)
- No `summary_ratio` in `custom` → always dense-only (existing behavior)
- All training pipeline, trainer, and collator logic **unchanged**

Example: `configs/stage_3_vision_lora.yaml` with `scheme: B` runs dense-only, no modifications needed.

## Core Design

### Per-Group Mode Selection

Rather than per-sample mode mixing (which leads to inconsistent JSON shapes in one batch), mode selection happens at the **pairing group level**:

- If `images_per_user_turn=2`, each group contains 2 images
- For that group, the dataset randomly picks a mode (dense or summary) based on `summary_ratio`
- **All 2 images in that group use the same mode and see the same system prompt**

This ensures all records in a group produce coherent JSON (either all objects with geometry or all summaries).

### System Prompts

Two system prompts are supported:

- **Dense**: `SYSTEM_PROMPT_A` or `SYSTEM_PROMPT_B` (selected via `prompts.scheme`)
- **Summary**: `SYSTEM_PROMPT_SUMMARY` (used when `custom.summary_ratio > 0`; loader forces this when `summary_ratio >= 1.0`)

### Dynamic Prompt Injection

At dataset `__getitem__`:
1. Randomly select mode for the group based on `summary_ratio`
2. Create builder with selected mode
3. **Temporarily inject appropriate system prompt** into template before encoding
4. Build and encode messages
5. **Restore original system prompt** after encoding

This ensures the right instructions are given without modifying YAML or trainer logic.

### Output Shapes

**Dense group**:
```json
{
  "图片_1": {
    "object_1": {"bbox_2d": [x1, y1, x2, y2], "desc": "..."},
    "object_2": {"quad": [x1, y1, ..., x4, y4], "desc": "..."}
  },
  "图片_2": {...}
}
```

**Summary group** (uses summary field):
```json
{
  "图片_1": "BBU设备/华为/显示完整/无需安装×1，螺丝、光纤插头×2，标签/可以识别×1",
  "图片_2": "光纤/有保护措施/蛇形管×3，标签/无法识别×1"
}
```

## Configuration

### Dense-Only (Default) — No Changes Needed

```yaml
prompts:
  scheme: B  # or A; both use dense system prompt

custom:
  train_jsonl: /path/to/train.jsonl
  val_jsonl: /path/to/val.jsonl
  emit_norm: norm1000
  images_per_user_turn: 2
  # No summary_ratio → always dense mode (existing behavior)
```

This is the recommended configuration for backward compatibility. No changes to existing configs required.

### Summary-Only

```yaml
prompts:
  scheme: B  # Use A or B; loader will force summary when ratio=1.0

custom:
  train_jsonl: /path/to/train.jsonl
  val_jsonl: /path/to/val.jsonl
  emit_norm: norm1000
  images_per_user_turn: 2
  summary_ratio: 1.0  # All groups use summary mode
  # Optional override
  # system_prompt_summary: |
  #   你是图像摘要助手。...
```

Behavior:
- Loader injects `SYSTEM_PROMPT_SUMMARY` when `summary_ratio >= 1.0` (regardless of scheme A/B)
- Dataset selects builder `mode="summary"` per group

**Data requirement**: All records **must** have a non-empty `summary` field in JSONL.

### Mixed (Dynamic Per-Group)

```yaml
prompts:
  scheme: B  # Dense system prompt (default for dense groups)

custom:
  train_jsonl: /path/to/train.jsonl
  val_jsonl: /path/to/val.jsonl
  emit_norm: norm1000
  images_per_user_turn: 2
  summary_ratio: 0.5  # Each group: 50% chance of summary, 50% chance of dense
```

When `summary_ratio=0.5`, roughly half the groups will use summary mode, half will use dense mode. The selection is deterministic per epoch (seeded RNG).

**Data requirement**: All records **must** have a non-empty `summary` field (needed for summary-selected groups).

## Summary Field Generation

### Standardized Format Rules

Your `summary` fields follow these rules (implemented in `scripts/regenerate_summaries.py`):

1. **All-slash format**: No mixed comma/slash separators
   - Commas in descriptions are replaced with slashes
   - Creates a unified hierarchical structure: `Type/attr1/attr2/attr3`

2. **Label categorization**:
   - Specific label text (e.g., "5G-AAU1光纤") → `"标签/可以识别"`
   - "无法识别" → `"标签/无法识别"` (unchanged)

3. **Object order preservation**:
   - Items appear in the order they were in the objects list
   - No reordering or alphabetization

4. **Attribute order preservation**:
   - Order of attributes from original descriptions maintained
   - All levels converted consistently

5. **Grouping with counts**:
   - Identical items grouped and counted: `item×N`
   - Items separated by Chinese comma `，`

### Regenerating Summaries

Use the helper when summaries are missing or malformed:

```bash
conda run -n ms python scripts/regenerate_summaries.py --input /abs/path/to/train.jsonl
```

### Verification

Quickly validate formatting:

```bash
conda run -n ms python scripts/verify_summary.py --jsonl /abs/path/to/train.jsonl
```

Expected output for properly formatted summaries:
```
Total samples:            2196
Correct summaries:        2196 ✅
Accuracy:                 100.0%
```

## Training & Inference

### Training Pipeline

No changes to trainer, collator, or optimization. Only JSON assistant target shape varies per group:

- Dense groups: `{"图片_1": {"object_1": {...}, ...}, ...}`
- Summary groups: `{"图片_1": "summary text", "图片_2": "summary text", ...}`

The template encodes both formats correctly; tokenizer handles either JSON structure.

### Inference

Load checkpoint and run inference normally. Chat template applies to both output formats:

- Model trained on dense learns to output geometry + desc
- Model trained on summary learns to output summary strings
- Model trained on mixed learns both

## Error Handling

**Missing `summary` when selected for summary mode**:
```
ValueError: Missing or invalid 'summary' for record index 5; expected non-empty string. 
Please ensure all records in JSONL have a 'summary' field when using summary mode.
```

**summary_ratio > 0 but system_prompt_summary not found**:
```
ValueError: summary_ratio > 0 but system_prompt_summary not found. 
Please set custom.system_prompt_summary in YAML or ensure SYSTEM_PROMPT_SUMMARY is defined.
```

## Example Workflows

### 1. Dense-only (default, unchanged) ✅
```bash
conda run -n ms python -m src.sft --config configs/stage_3_vision_lora.yaml
```

- Uses `scheme: B` (from config)
- No `summary_ratio` → always dense
- **Backward compatible** — no YAML changes needed

### 2. Summary-only
```bash
conda run -n ms python -m src.sft --config configs/example_summary.yaml
```

- Use scheme A or B; set `custom.summary_ratio: 1.0`
- Loader injects `SYSTEM_PROMPT_SUMMARY`; builder uses `mode="summary"`
- Requires all records to have valid `summary`

### 3. Mixed (50/50 per-group)
```bash
conda run -n ms python -m src.sft --config configs/example_mixed.yaml
```

- Uses `scheme: B` (dense system prompt as default)
- `summary_ratio: 0.5` → each group: 50% summary, 50% dense
- Requires all records to have valid `summary`

## Files & Tools

- `scripts/regenerate_summaries.py`: Generate standardized summaries from object descriptions
- `scripts/verify_summary.py`: Validate summary field accuracy (target: 100%)
- `docs/DENSE_SUMMARY_VARIANT.md`: This document

## Modules and Boundaries (Where logic lives)

- `src/config/loader.py`:
  - Resolves `prompts.scheme` (A/B)
  - Forces `SYSTEM_PROMPT_SUMMARY` when `custom.summary_ratio >= 1.0`
  - Exposes `custom.user_prompt`; sets `template.system`

- `src/datasets/dense_caption.py` (DenseCaptionDataset):
  - Per-group mode selection via `summary_ratio`
  - Temporarily injects system prompt per group
  - Builds via `JSONLinesBuilder(mode)` and calls `template.encode`

- `src/datasets/builders/jsonlines.py` (JSONLinesBuilder):
  - `mode="dense"`: grouped JSON with geometry + desc
  - `mode="summary"`: grouped JSON with per-image summary strings
  - Emits user `{"type":"image","image":...}`; maintains `objects` for template normalization

- `src/datasets/preprocessors/dense_caption.py`:
  - Validates records; optional `require_summary` to filter when summary is mandatory

- `src/datasets/dynamic_pair.py`:
  - Epoch-seeded pairing and fixed-size grouping (`images_per_user_turn`)
  - Supports `build_many(records)` builder API

- `src/sft.py`:
  - Loads YAML via `ConfigLoader`, builds dataset, optional augmentation
  - Applies tuner via `sft.prepare_model(...)` before trainer creation

## Notes

- **Standardized Format**: All summaries now use unified all-slash format for consistency
- **Backward Compatible**: Dense-only mode (default) is unchanged; existing configs work as-is
- **Fail-Fast**: Missing/invalid `summary` errors at dataset load time, not training
- **Per-Group Cohesion**: All samples in a group share the same prompt and mode; no mixed shapes per batch
- **Deterministic Ratio**: RNG seeded per epoch; ratio statistically maintained across batches
- **No Template Changes**: Chat template sees valid JSON either way; no modifications needed
- **Dynamic Injection**: System prompt temporarily set per group; original restored after encoding

## Troubleshooting

**"Missing or invalid 'summary'"**
- Check all JSONL records have `summary` field (for mixed or summary-only modes)
- Ensure `summary` is a non-empty string (not null, empty, or number)
- Verify UTF-8 encoding
- Not required for dense-only mode

**summary_ratio > 0 but training errors early**
- Check `SYSTEM_PROMPT_SUMMARY` is available in `src/config/prompts.py`
- Verify `summary_ratio` value is between 0 and 1
- Ensure all records have valid `summary` field

**Summary field looks malformed**
- Run `scripts/verify_summary.py --jsonl your_file.jsonl` to check format
- Use `scripts/regenerate_summaries.py --input your_file.jsonl` to fix
- Expected accuracy: 100% after regeneration

**Existing configs won't run**
- This should not happen — backward compatibility is maintained
- Verify `prompts.scheme` is `A` or `B` (not custom values)
- Check no new required fields added to `custom.*`

## Files Modified

- `src/config/prompts.py`: Centralized prompt templates
- `src/config/loader.py`: Config loading and scheme resolution
- `src/datasets/builders/jsonlines.py`: Message builders (dense/summary modes)
- `src/datasets/dense_caption.py`: Dataset with mode selection
- `src/sft.py`: Runner with prompt/ratio extraction
- `scripts/regenerate_summaries.py`: **NEW** — Regenerate summaries from descriptions
- `scripts/verify_summary.py`: **NEW** — Validate summary field accuracy
- `docs/DENSE_SUMMARY_VARIANT.md`: This document (updated with standardized format)

---

**Last Updated**: October 24, 2025
**Format Version**: Standardized All-Slash (100% accuracy)
**Dataset**: Qwen3-VL BBU Installation QC

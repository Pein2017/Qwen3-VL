# Data & Datasets

Comprehensive guide to data format, schema, dataset builders, and preprocessing pipeline.

**Source of truth**: `src/datasets/`, `src/datasets/data_details.md`, `src/datasets/geometry.py`

**Raw annotation intake** is covered in `DATA_PREPROCESSING_PIPELINE.md` (how `data_conversion/` produces the train/val JSONL that feed this pipeline).

---

## Table of Contents
- [Data Format](#data-format)
- [Dataset Pipeline](#dataset-pipeline)
- [Builders](#builders)
- [Preprocessors](#preprocessors)
- [Best Practices](#best-practices)

---

## Data Format

### JSONL Schema

Each record in your training data follows this structure:

```json
{
  "images": ["path/to/img1.jpg", "path/to/img2.jpg"],
  "objects": [
    {"bbox_2d": [x1, y1, x2, y2], "desc": "物体描述"},
    {"poly": [x1, y1, x2, y2, x3, y3, ...], "poly_points": M, "desc": "多边形描述"},
    {"line": [x1, y1, ..., xn, yn], "line_points": N, "desc": "线段描述"}
  ],
  "width": 1920,
  "height": 1080,
  "summary": "可选: 单行中文汇总"
}
```

**Key Rules**:
- Image paths resolve relative to JSONL file directory (absolute paths also allowed)
- Exactly ONE geometry field per object (`bbox_2d`, `poly`, or `line`)
- For polygons: `poly` is a flat, even-length list (≥6 values / ≥3 points). `poly_points` is optional metadata but should match `len(poly) / 2` when present.
- For lines: `line_points` should equal number of coords ÷ 2 (optional but recommended; validation falls back to the coord count when absent)
- Coordinates are in pixel space with original `width`/`height`

### Geometry Types

| Type | Format | Use Case |
|------|--------|----------|
| **bbox_2d** | `[x1, y1, x2, y2]` | Axis-aligned boxes |
| **poly** | `[x1,y1, x2,y2, x3,y3, ...]` | Arbitrary polygons (even-length list, ≥3 points). Use `poly_points` to record vertex count. |
| **line** | `[x1,y1, ..., xn,yn]` + `line_points: N` | Polylines (cables, fibers) |

### Coordinate Normalization

**On Disk** (source data):
- Pixel coordinates with `width`/`height` metadata
- Keep original resolution for maximum precision

**During Training**:
- `custom.emit_norm` controls assistant text format:
  - `none`: Pixel coordinates in JSON
  - `norm100`: Normalized to [0, 100]
  - `norm1000`: Normalized to [0, 1000]
- Template automatically normalizes top-level `objects` to `norm1000` for grounding

### Modes: Dense vs Summary

**Dense Mode** (default):
```json
{
  "object_1": {"bbox_2d": [100, 200, 300, 400], "desc": "..."},
  "object_2": {"line_points": 4, "line": [50, 60, 80, 120, 130, 180, 180, 220], "desc": "..."}
}
```

**Summary Mode**:
```
"单行汇总文本"
```
Requires `summary` field in every record.
Enable by setting `custom.use_summary: true` in the training config. Mixed dense/summary sampling is **not** implemented; the dataset runs in whichever mode `use_summary` selects.

### Summary Field Standard

Format requirements (aligned with training/inference prompts):
- Use Chinese comma '，' between items (object entries)
- Group strictly by the raw `desc` text; do not rewrite/canonicalize
- Merge identical desc into `desc×N` (full-width ×, no space)
- Ordering: sort by `len(desc)` ascending; ties keep first appearance order
- Keep `desc` exactly as annotated（含备注、组前缀等），不把备注另行拆出
- Single sentence per image: no newlines, no trailing '。', no extra spaces
- No geometry or coordinate arrays in summary strings
- Optional training toggle: set `custom.summary_label_grouping: true` to collapse all 标签/* entries that are not `标签/无法识别` into `标签/可以识别×N` while preserving the `无法识别` count separately.
- Conversion is fail-fast: if a sample has no objects or all `desc` are空/缺失，`build_summary_from_objects` raises `ValueError` and the sample is rejected.

**Example**: `光模块×3，线缆×2，BBU设备/需复核,备注:无法判断品牌×1`

---

## Dataset Pipeline

### Critical Configuration Requirement

**REQUIRED in all config files**:
```yaml
data:
  dataset: ["dummy"]  # NEVER REMOVE - required for ms-swift TrainArguments validation
```

**Why this is needed**:
- ms-swift's `TrainArguments.__post_init__()` validates that `dataset` or `cached_dataset` is non-empty
- This check happens during config initialization, before custom dataset loading
- Even though we load actual datasets via `custom.train_jsonl` and pass them directly to the trainer, the validation must pass first
- The `["dummy"]` placeholder satisfies the validation but is never actually used
- Removing this will cause: `ValueError: self.dataset: [], self.cached_dataset: []. Please input the training dataset.`

**Source**: `/data/ms-swift/swift/llm/argument/train_args.py:162-164`

### Architecture Overview

```
JSONL → DenseCaptionDataset → Collator → Trainer
```

**Key Components**:
1. **DenseCaptionDataset**: Mode selection (dense/summary), augmentation config, per-item orchestration
2. **Preprocessors**: Validation, augmentation (plugged into the dataset)
3. **Builder**: Message formatting (JSONLinesBuilder)
4. **Collator**: Tensor preparation with standard padding (packing removed)

### Visual Feature Distillation (optional)

- Enable via `custom.visual_kd` when you want to lock the vision/aligner stack to a teacher while giving the language tower more room.
- The dataset already supplies `pixel_values` and `image_grid_thw`; as long as a record contains images, the trainer captures and distills the corresponding activations automatically.
- Batches without images (e.g., summary-only validation groups) skip the extra loss—no action required.

### DenseCaptionDataset

**Role**:
- Selects dense vs summary mode per sample
- Applies augmentation/preprocessing
- Attaches metadata for downstream processing and template encoding

**Configuration**:
```yaml
custom:
  train_jsonl: /path/to/train.jsonl
  val_jsonl: /path/to/val.jsonl
  use_summary: false                # true → summary-only mode
  emit_norm: norm1000              # Coordinate format in text
```

## Conversion & QA Tooling

If your source is a human-annotation export, start with the intake guide (`docs/DATA_PREPROCESSING_PIPELINE.md`) and run `data_conversion/convert_dataset.sh` to produce train/val/tiny JSONL that already satisfy this contract.

- **BBU conversion (`data_conversion/`)**:
  - `convert_dataset.sh` wraps `data_conversion/pipeline/unified_processor.py` with environment + parameter guardrails (max pixels, resize factor, validation toggles).
  - After `train.jsonl`/`val.jsonl` are built, the script also writes `train_tiny.jsonl` (20 samples) and `val_tiny.jsonl` (8 samples) in the same output directory using the same `SEED` for deterministic debug runs.
  - Taxonomies live in `attribute_taxonomy.json` + `hierarchical_attribute_mapping.json`; update both when new object types/attributes ship. `pipeline/summary_builder.py` and `pipeline/flexible_taxonomy_processor.py` consume these definitions.
  - Validation artifacts (`invalid_objects.jsonl`, `validation_results.json`) allow offline QA before a dataset ever reaches `src/datasets/`.
  - Coordinate sanity is centralized in `pipeline/coordinate_manager.py` (EXIF + smart-resize + clamp) and `pipeline/vision_process.py`.
  - RRU now reuses the same pipeline: new classes/attributes (`ground_screw`, 尾纤/接地线标签与套管保护等) are defined in the taxonomy JSONs; station + distance are merged to `站点距离/<text>`; group info is encoded in `desc` via `组<id>:` prefix (no top-level `groups`). Records fail fast if any group has only one member.
  - RRU summaries preserve the full `desc` (including组前缀/备注) and aggregate identical entries with `×N`, matching BBU dense/summary behavior while avoiding geometry text.
  - Polygon vertices are canonicalized offline (clockwise, top-most vertex first, closing-duplicate removed) to prevent self-crossing; `vis_tools/` applies the same ordering when overlaying samples.
- **Public datasets (`public_data/`)**:
  - See `PUBLIC_DATA.md` + `public_data/README.md` for LVIS download, conversion, sampling, visualization, and pytest coverage.
  - Each converter produces JSONL that matches this document’s schema; polygons include `poly_points`. Cap polygon complexity during conversion (e.g., `--poly-max-points 12`) if you want oversized shapes turned into `bbox_2d`.
- **Fusion tooling**:
  - `scripts/fuse_datasets.py` plus `src/datasets/fusion.py` can pre-build fused JSONL based on a YAML config (target dataset + auxiliary sources). Useful when you want deterministic sampling instead of streaming fusion.
- **Visualization**:
  - `vis_tools/vis_augment_compare.py` and friends overlay objects/summaries to validate augmentation and JSONL integrity. See `vis_tools/README_CROP_VIS.md`.

## Multi-Dataset Fusion

When you want BBU/RRU multi-target dense-caption training to consume auxiliary detection datasets (LVIS, COCO, etc.), provide a `custom.fusion_config` (YAML/JSON). The unified fusion loader mixes:

- **Targets (one or more)**: declare under `targets:`. Optional per-target `ratio` is self-scaled: `quota_i = round(len_i * ratio_i)` with `ratio_i` defaulting to `1.0` (ratio < 1 downsamples, ratio > 1 upsamples with replacement; ratio = 1 or unset keeps full coverage). Target indices are shuffled deterministically per epoch. Evaluation concatenates all target `val_jsonl` splits (no sources).
- **Auxiliary sources**: each entry declares the dataset wrapper (e.g., `coco`, `lvis`, `objects365`) plus a `ratio`. Each epoch samples `round(ratio * N_target_total)` records **with replacement**, where `N_target_total` is the sum of target quotas for that epoch. Errors if the source pool is empty; shuffles deterministically using the fusion seed and optional per-dataset seed.
- **Text-only sources**: you can add a chat-only auxiliary (`dataset: chat`, `template: chatml`) that points to a JSONL with pre-authored `messages` only (e.g., `public_data/coig_cqia/coig_cqia_merged.jsonl`). Chat sources skip augmentation/curriculum, reuse their own prompts, and are mixed by ratio like any other source.
- **Per-dataset fields** (target and sources): `name`, `train_jsonl`, optional `val_jsonl`, `template`, optional `user_prompt`/`system_prompt` override, `augmentation_enabled`, `curriculum_enabled`, `max_objects_per_image`, optional `seed`. Sources default to **no augmentation/curriculum** and a **64 object cap**; targets inherit global augmentation/curriculum and can opt into a cap.
- **Prompt priority**: `default < domain (wrapper template) < dataset-specific override`, applied to both system and user prompts per sample while keeping a single shared template instance.
- **Object caps**: applied deterministically after augmentation and before encoding. Sources cap by default; targets may opt in via `max_objects_per_image`.
- **Telemetry**: `last_sample_debug` exposes `dataset`, `prompt_source`, augmentation on/off, cap applied/limit, and input length for every sample; per-epoch `epoch_plan` reports counts and policy flags.
- **No online smart-resize**: inputs are assumed pre-filtered/resized offline; resizing occurs only through augmentation ops when configured. If you need smart-resize, run it during conversion and provide the resized `train/val` JSONLs explicitly.

Example fusion config:

```yaml
target:
  dataset: bbu
  params:
    train_jsonl: /data/bbu/train.jsonl
    val_jsonl: /data/bbu/val.jsonl
sources:
  - dataset: coco
    ratio: 0.1
    params:
      train_jsonl: /data/coco/train.jsonl
      user_prompt: "List objects in JSON."        # optional override
      max_objects_per_image: 48                  # optional cap override
      seed: 123                                  # optional per-source seed
  - dataset: objects365
    ratio: 0.05
    params:
      train_jsonl: /data/objects365/train.jsonl
```

Runtime loader: `custom.fusion_config` always uses `FusionCaptionDataset` (alias `UnifiedFusionDataset`) with a single shared template. For deterministic static mixes, you can still precompute fused JSONLs with `scripts/fuse_datasets.py --config <path>`.

For the universal JSONL record contract shared by all domains, see `docs/DATA_JSONL_CONTRACT.md`.

---

## Builders

### JSONLinesBuilder

**Purpose**: Formats single-image records into single-turn conversation messages

**Dense Mode**:
```python
# User message: embed the image followed by the prompt
[
  {"type": "image", "image": "path"},
  {"type": "text", "text": prompt}
]

# Assistant message: minimal object hierarchy (no per-image wrapper)
{
  "object_1": {"bbox_2d": [...], "desc": "类型/属性/..."},
  "object_2": {"line_points": 4, "line": [...], "desc": "..."}
}
```

**Summary Mode**:
```python
# Assistant message: single summary string
"标签×3，BBU设备×1，挡风板×1"
```

**Key Behavior**:
- Attaches top-level `objects` with pixel coords (for template normalization)
- Geometries normalized based on `emit_norm` setting
- Deterministic ordering of object indices (`object_1`, `object_2`, ...)
- Consumes validated `ConversationRecord` objects and exposes augmentation telemetry (`pipeline.last_summary`) for downstream health checks.

---

## Preprocessors

### DenseCaptionPreprocessor

**Purpose**: Validation and light filtering

**Checks**:
- Schema validity (required fields present)
- Geometry field uniqueness (exactly one per object)
- Line point count matches `line_points`
- Image paths resolve correctly

**Action**: Raises `ValueError` on invalid records (fail-fast)

### AugmentationPreprocessor

**Purpose**: Apply geometry-aware augmentations

**Features**:
- Atomic updates (image + geometries transformed together)
- Preserves coordinate alignment
- See [DATA_AUGMENTATION.md](DATA_AUGMENTATION.md) for details
- Reads standardized telemetry (`AugmentationTelemetry`) with crop coverage, kept indices, and skip reasons to audit augmentation pipelines.

**Example**:
```yaml
custom:
  augmentation:
    enabled: true
    bypass_prob: 0.1              # 10% clean samples
    ops:
      - name: hflip
        params: { prob: 0.5 }
      - name: rotate
        params: { max_deg: 25.0, prob: 0.4 }
      - name: random_crop
        params: { scale: [0.7, 1.0], prob: 0.3 }
      - name: resize_by_scale
        params: { lo: 0.9, hi: 1.1, prob: 0.5 }
      - name: color_jitter
        params: { brightness: [0.75, 1.25], prob: 0.5 }
      # ✅ MUST be last: ensures final padding to multiple of 32
      - name: expand_to_fit_affine
        params: { multiple: 32 }
```

### Domain Context: BBU Installation Inspection

This corpus covers telecom cabinet inspections, focused on **BBU (Baseband Unit) installation quality**. Understanding the domain helps keep dense annotations, summaries, and production flows aligned.

- **Dense captioning** (training) enumerates every inspected item. Object types map to the attribute taxonomy in `data_conversion/hierarchical_attribute_mapping.json` and `data_conversion/attribute_taxonomy.json` (ignore the `occlusion` block). Core categories include:
  - `bbu` (BBU设备) — expects attributes such as brand (`bbu_brand`), completeness (`bbu_stituation`), and windshield requirement (`bbu_equipment`).
  - `bbu_shield` (挡风板) — brand, completeness, obstruction status, and installation direction.
  - `connect_point` (接地/接线端子), `label` (标签), plus `fiber` / `wire` lines for cabling.
  Geometry constraints from the mapping JSON enforce polys/bboxes for hardware and polylines for cabling.
- **Summary / Stage-A / Stage-B** (production) reuse the same records:
  - Stage-A generates per-image summaries (requires every record to carry a `summary`).
  - Stage-B consumes multiple images per site to emit a final deployment verdict.
  - Keep taxonomies in sync so Stage pipelines and dense training stay compatible.
- **Hierarchy semantics**: Attribute templates are slash-delimited (`brand/completeness/...`). Conditional levels (e.g., windshield conformity) only appear when parent attributes require them. Free-text `备注` fields append at the end.

When adding new inspection criteria, update both conversion JSON files and regenerate summaries before training to avoid drift between dense captions and production outputs.

---

## Best Practices

### Data Preparation

✅ **Do**:
- Keep pixel coords on disk (template normalizes during training)
- Use relative image paths when possible
- Validate schema before training
- Include `width`/`height` metadata
- Test with small dataset first

❌ **Don't**:
- Mix coordinate systems in same file
- Omit required fields (`width`, `height`)
- Use absolute paths unnecessarily
- Skip validation (fail early is better)

### Schema Validation

```bash
# Recommended: validate before training
python -m src.datasets.validate_jsonl --input train.jsonl --verbose
```

**Common Issues**:
- Missing `line_points` for line geometries
- Multiple geometry fields per object
- Path resolution failures
- Width/height mismatch

### Performance Tips

1. **Image Loading**: Use relative paths from JSONL directory for portability
2. **Augmentation**: Enable only needed ops (each adds overhead)
3. **Packing**: Removed. Training always uses padded batches; packing knobs are rejected.

### Debugging

**Enable debug mode**:
```bash
python -m src.sft --config config.yaml --debug
```

**Check first batch**:
```python
from src.datasets import DenseCaptionDataset
ds = DenseCaptionDataset(config)
item = ds[0]
print(item.keys())  # input_ids, labels, pixel_values, ...
```

---

## Collation (padding-only)

Training and evaluation now always use padded batches:
- `input_ids`, `labels`, `pixel_values`, `image_grid_thw`, `objects` are produced by the template collator.
- Packing and its knobs (`training.packing`, `custom.packing_group_key`, cached length overrides) are removed; configs containing them fail fast.
- Per-dataset telemetry is still available in the padded path using dataset labels from metadata.

---

## Verification Checklist

Before training:

- [ ] JSONL schema valid (all required fields present)
- [ ] Geometry fields correct (one per object)
- [ ] Line objects have `line_points` matching coord count
- [ ] Image paths resolve correctly
- [ ] Width/height metadata present
- [ ] Summary field present (if using summary mode)
- [ ] Coordinates in pixel space (not normalized)
- [ ] No duplicate objects or malformed geometries
- [ ] Test with `--debug` flag first

---

## See Also

- **Augmentation**: [DATA_AUGMENTATION.md](DATA_AUGMENTATION.md) - Geometry-aware transforms
- **Training**: [REFERENCE.md](REFERENCE.md#training) - Full training guide
- **Architecture**: [README.md](README.md#architecture) - End-to-end pipeline
- **Upstream Models**: [UPSTREAM_DEPENDENCIES.md](UPSTREAM_DEPENDENCIES.md) - HF Qwen3-VL + ms-swift background

---

**Last Updated**: 2025-11-24 (geometry schema + links)

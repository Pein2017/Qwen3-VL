# Data & Datasets

Status: Active
Scope: Dataset schema, builders, preprocessing, and conversion/fusion integration.
Owners: Data Pipeline + Training
Last updated: 2026-01-12
Related: [DATA_JSONL_CONTRACT.md](DATA_JSONL_CONTRACT.md), [DATA_PREPROCESSING_PIPELINE.md](DATA_PREPROCESSING_PIPELINE.md), [UNIFIED_FUSION_DATASET.md](UNIFIED_FUSION_DATASET.md)

Comprehensive guide to data format, schema, dataset builders, and preprocessing pipeline.

**Source of truth**: `src/datasets/`, `src/datasets/changelog.md`, `src/datasets/geometry.py`

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
- For polygons: `poly` is a flat, even-length list (≥8 values / ≥4 points; current runtime validation). `poly_points` is optional metadata but should match `len(poly) / 2` when present.
- For lines: `line_points` should equal number of coords ÷ 2 (optional but recommended; validation falls back to the coord count when absent)
- Coordinates are in pixel space with original `width`/`height`

### Geometry Types

| Type | Format | Use Case |
|------|--------|----------|
| **bbox_2d** | `[x1, y1, x2, y2]` | Axis-aligned boxes |
| **poly** | `[x1,y1, x2,y2, x3,y3, ...]` | Polygons (even-length list, ≥4 points). Use `poly_points` to record vertex count. |
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
"{...JSON summary string...}"
```
Requires `summary` field in every record (JSON string), except irrelevant-image samples which remain `summary: 无关图片`.
For fusion runs, set `mode: summary` (or `use_summary: true`) per dataset inside the fusion config to mix summary targets with dense sources. When `mode` is omitted, datasets fall back to `custom.use_summary` (default = dense) for backward compatibility.
Dense mode validation requires at least one object with geometry; summary mode validation requires a non-empty `summary` string.

### Summary Field Standard

Format requirements (aligned with training/inference prompts):
- Summary is a **single-line JSON string** (no newlines, no trailing punctuation), except irrelevant-image samples which keep `summary: 无关图片`.
- Required keys: `统计`.
- `统计` is a list; each item contains `类别` plus any observed attribute counts (`{value: count}`).
- BBU summaries include a top-level `备注` list when non-empty; RRU summaries omit `备注` and may include `分组统计` when present.
- Training corpora must not emit a `dataset` key in summary JSON.
- Only observed values are counted; do not emit missing/review/遮挡 placeholders.
- OCR/备注 are free text: remove whitespace only (preserve `,|=` and other symbols); unreadable → `可读性=不可读` (no “可以识别/无法识别”). Any stray comma tokens without `key=` are folded into `备注`, and `这里已经帮助修改,请注意参考学习` is stripped if present.
- Conversion is fail-fast: if a sample has no objects or all `desc` are空/缺失，`build_summary_from_objects` raises `ValueError` and the sample is rejected.
- Conversion also raises when invalid/unknown/conflict markers are detected in desc; fix raw annotations rather than emitting placeholder fields.

#### Fixed Value Compression (BBU/RRU)

BBU/RRU converters normalize fixed (non‑free‑text) values to compact forms. OCR/备注 are **not** compressed.

- **可见性**: `完整` / `部分` (from `显示完整` / `只显示部分`)
- **挡风板需求**: `免装` / `空间充足需安装`
- **挡风板符合性**: `按要求配备` / `未按要求配备`
- **安装方向**: `方向正确` / `方向错误`
- **符合性**: `符合` / `不符合`
- **保护措施**: `有保护` / `无保护`
- **保护细节**: `蛇形管` / `铠装` / `蛇形管+铠装`
- **弯曲半径**: `半径合理` / `半径不合理<4cm或成环`
- **捆扎**: `整齐` / `散乱`
- **安装状态**（RRU紧固类）: `合格` / `不合格`
- **标签**（RRU尾纤/接地线）: `有标签` / `无标签`
- **套管保护**（RRU尾纤）: `有套管` / `无套管`

#### RRU Summary Guidance

- RRU summaries are built from key=value descs; `站点距离` appears as `类别=站点距离,站点距离=<int>` in desc (current exports yield digits) and becomes a `统计` entry with key `站点距离`.
- Train summary-mode runs via `configs/train/sft/summary_1024.yaml` (dataset mix in `configs/fusion/variants/bbu_rru_summary_1024.yaml`). To focus on RRU only, edit the mix to keep just the RRU summary stream.

**Example**: `{"统计": [{"类别": "BBU设备", "品牌": {"华为": 1}}, {"类别": "标签", "文本": {"NR900-BBU": 1}}]}`

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

**Source**: ms-swift TrainArguments validation (see [docs/ops/UPSTREAM_DEPENDENCIES.md](../ops/UPSTREAM_DEPENDENCIES.md)).

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
- Selects dense vs summary mode per sample (per-dataset when using fusion configs)
- Applies augmentation/preprocessing
- Attaches metadata for downstream processing and template encoding

**Configuration**:
```yaml
custom:
  train_jsonl: path/to/train.jsonl
  val_jsonl: path/to/val.jsonl
  use_summary: false                # true → summary-only mode
  emit_norm: norm1000              # Coordinate format in text
  # fusion configs: each target/source can set mode: dense|summary (alias use_summary) to override this default
```

## Conversion & QA Tooling

If your source is a human-annotation export, start with the intake guide (`./DATA_PREPROCESSING_PIPELINE.md`) and run `data_conversion/convert_dataset.sh` to produce train/val/tiny JSONL that already satisfy this contract.

- **BBU conversion (`data_conversion/`)**:
  - `convert_dataset.sh` wraps `data_conversion/pipeline/unified_processor.py` with environment + parameter guardrails (max pixels, resize factor, validation toggles).
  - After `train.jsonl`/`val.jsonl` are built, the script also writes `train_tiny.jsonl` (20 samples) and `val_tiny.jsonl` (8 samples) in the same output directory using the same `SEED` for deterministic debug runs.
  - Taxonomies live in `attribute_taxonomy.json` + `hierarchical_attribute_mapping.json`; update both when new object types/attributes ship. `pipeline/summary_builder.py` and `pipeline/flexible_taxonomy_processor.py` consume these definitions.
  - Validation artifacts (`invalid_objects.jsonl`, `validation_results.json`) allow offline QA before a dataset ever reaches `src/datasets/`.
  - Coordinate sanity is centralized in `pipeline/coordinate_manager.py` (EXIF + smart-resize + clamp) and `pipeline/vision_process.py`.
  - RRU now reuses the same pipeline: new classes/attributes (`ground_screw`, 尾纤/接地线标签与套管保护等) are defined in the taxonomy JSONs; station distance is represented as `类别=站点距离,站点距离=<int>` (digits extracted from raw text); group info is encoded in `desc` via `组=<id>` (no top-level `groups`). Records fail fast if any group has only one member.
  - RRU summaries are JSON strings with per-category stats; `备注` is omitted and `分组统计` may appear when group counts exist.
  - Polygon vertices are canonicalized offline (clockwise, top-most vertex first, closing-duplicate removed) to prevent self-crossing; `vis_tools/` applies the same ordering when overlaying samples.
- **Public datasets (`public_data/`)**:
  - See `PUBLIC_DATA.md` + `public_data/README.md` for LVIS download, conversion, sampling, visualization, and pytest coverage.
  - Each converter produces JSONL that matches this document’s schema; polygons include `poly_points`. Cap polygon complexity during conversion (e.g., `--poly-max-points 12`) if you want oversized shapes turned into `bbox_2d`.
- **Fusion tooling**:
  - `scripts/fuse_datasets.py` plus `src/datasets/fusion.py` can pre-build fused JSONL based on a YAML config (target dataset + auxiliary sources). Useful when you want deterministic sampling instead of streaming fusion.
- **Visualization**:
- `vis_tools/vis_augment_compare.py` and friends overlay objects/summaries to validate augmentation and JSONL integrity. See `../../vis_tools/README_CROP_VIS.md`.

## Multi-Dataset Fusion

When you want BBU/RRU multi-target dense-caption training to consume auxiliary detection datasets (LVIS, COCO, etc.), provide a `custom.fusion_config` (YAML/JSON). The unified fusion loader mixes:

### Supported Templates & Description Styles
Fusion can mix multiple record styles as long as the base template is compatible:

- **ChatML text-only**: JSONL with `messages` only (no `images`/`objects`/`summary`). Use `dataset: chat` with `template: chatml` (or another chatml-compatible template). These samples bypass dense-caption construction.
- **Generic detection (LVIS/COCO, etc.)**: JSONL with `images` + `objects`, `desc` is typically a single category token or short phrase (no key/value hierarchy), and `summary` is usually absent.
- **Target domain (BBU/RRU)**: JSONL with `images` + `objects`, `desc` uses comma-separated `key=value` pairs (`类别` first); `summary` is a JSON string (or the literal `无关图片` for irrelevant streams).

**Template note**: `FusionCaptionDataset` uses a **single template instance**; per-dataset `template` selects prompt presets, not a different template class. When mixing chat-only + dense-caption sources, choose a base template that can encode both formats (ChatML family recommended).

- **Targets (one or more)**: declare under `targets:`. Optional per-target `ratio` is self-scaled: `quota_i = round(len_i * ratio_i)` with `ratio_i` defaulting to `1.0` (ratio < 1 downsamples, ratio > 1 upsamples with replacement; ratio = 1 or unset keeps full coverage). Target indices are shuffled deterministically per epoch. Evaluation concatenates all target `val_jsonl` splits (no sources).
- **Auxiliary sources**: each entry declares the dataset wrapper (e.g., `coco`, `lvis`, `objects365`) plus a `ratio`. Each epoch samples `round(ratio * N_target_total)` records **with replacement**, where `N_target_total` is the sum of target quotas for that epoch. Errors if the source pool is empty; shuffles deterministically using the fusion seed and optional per-dataset seed.
- **Text-only sources**: you can add a chat-only auxiliary (`dataset: chat`, `template: chatml`) that points to a JSONL with pre-authored `messages` only (e.g., `public_data/coig_cqia/coig_cqia_merged.jsonl`). Chat sources skip augmentation/curriculum, reuse their own prompts, and are mixed by ratio like any other source.
- **Per-dataset fields** (target and sources): `name`, `train_jsonl`, optional `val_jsonl`, `template`, optional `user_prompt`/`system_prompt` override, `augmentation_enabled`, `curriculum_enabled`, `max_objects_per_image`, optional `seed`. Sources default to **no augmentation/curriculum** and a **64 object cap**; targets inherit global augmentation/curriculum and can opt into a cap.
- **Prompt priority**: `default < domain (wrapper template) < dataset-specific override`, applied to both system and user prompts per sample while keeping a single shared template instance.
- **Object caps**: applied deterministically after augmentation and before encoding. Sources cap by default; targets may opt in via `max_objects_per_image`.
- **Telemetry**: `last_sample_debug` exposes `dataset`, `prompt_source`, augmentation on/off, cap applied/limit, and input length for every sample; per-epoch `epoch_plan` reports counts and policy flags.
- **No online smart-resize**: inputs are assumed pre-filtered/resized offline; resizing occurs only through augmentation ops when configured. If you need smart-resize, run it during conversion and provide the resized `train/val` JSONLs explicitly.

### Irrelevant Streams (Negative Pools)

For SFT regularization (reduce hallucinations on out-of-domain images), you can add small **irrelevant** streams (e.g., `irrelevant_summary`, `irrelevant_dense`). Irrelevant streams are identified by `_fusion_source` starting with `irrelevant` and have special handling:
- **Assistant text is always exactly** `无关图片` (**single line**, **no prefix**), even if the stream is declared as `mode: dense`.
- Prompts reuse summary templates and alternate per epoch between `summary_bbu` and `summary_rru` (~50/50 within an epoch) without changing `_fusion_source` (eval uses a deterministic mapping).
- Use `data/irrelevant_summary/train.jsonl` as the shared backing pool; records keep `summary: 无关图片` and include a dummy full-frame bbox to satisfy the global JSONL contract.

See `configs/fusion/variants/bbu_rru_dense_plus_summary_1024.yaml` for the current dense+summary SFT mix (targets + irrelevant streams + public sources).

- Generate the JSONL from a folder of JPEGs (EXIF-aware width/height) and keep the global contract by emitting a single dummy full-frame bbox per image:
  - `conda run -n ms python scripts/build_irrelevant_summary_jsonl.py --images-dir data/irrelevant_summary/images --output-jsonl data/irrelevant_summary/train.jsonl`
- Reference it as an additional **target** (target ratios scale by the dataset's own pool size; `ratio: 1` means each image appears once per epoch). See `configs/fusion/variants/bbu_rru_summary_1024.yaml` for a concrete example.

Example fusion config:

```yaml
targets:
  - name: bbu_dense
    dataset: bbu
    train_jsonl: data/bbu/train.jsonl
    val_jsonl: data/bbu/val.jsonl
    template: target_dense_bbu
    mode: dense
    ratio: 1.0
sources:
  - name: coco
    dataset: coco
    ratio: 0.1
    train_jsonl: data/coco/train.jsonl
    template: source_dense
    mode: dense
    user_prompt: "List objects in JSON."          # optional override
    max_objects_per_image: 48                     # optional cap override
    seed: 123                                     # optional per-source seed
    augmentation_enabled: false
    curriculum_enabled: false
  - name: objects365
    dataset: objects365
    ratio: 0.05
    train_jsonl: data/objects365/train.jsonl
    template: source_dense
    mode: dense
    augmentation_enabled: false
    curriculum_enabled: false
```

Runtime loader: `custom.fusion_config` always uses `FusionCaptionDataset` (alias `UnifiedFusionDataset`) with a single shared template. For deterministic static mixes, you can still precompute fused JSONLs with `scripts/fuse_datasets.py --config <path>`.

For the universal JSONL record contract shared by all domains, see `./DATA_JSONL_CONTRACT.md`.

---

## Builders

### JSONLinesBuilder

**Purpose**: Formats single-image records into single-turn conversation messages

**Dense Mode**:
```text
# User message: embed the image followed by the prompt
[
  {"type": "image", "image": "path"},
  {"type": "text", "text": prompt}
]

# Assistant message: minimal object hierarchy (no per-image wrapper)
# (When assistant_prefix_format is enabled, prepend the prefix line + newline.)
<TASK=DETECTION>, <DATASET=bbu>
{
  "object_1": {"bbox_2d": [...], "desc": "类别=BBU设备,品牌=华为,可见性=部分"},
  "object_2": {"line_points": 4, "line": [...], "desc": "类别=站点距离,站点距离=51"}
}
# For generic detection sources (LVIS/COCO), desc can be a short class token (e.g., "person", "car").
```

**Summary Mode**:
```text
# Assistant message: single summary string (JSON), with optional prefix line.
<TASK=SUMMARY>, <DATASET=bbu>
{"统计": [{"类别": "BBU设备", "品牌": {"华为": 1}}]}
```

**Key Behavior**:
- Attaches top-level `objects` with pixel coords (for template normalization)
- Geometries normalized based on `emit_norm` setting
- Deterministic ordering of object indices (`object_1`, `object_2`, ...)
- Assistant JSON/summary serialization uses separators `", "` and `": "` to preserve spaces in coordinate lists (tokenizer stability).
- When `custom.assistant_prefix_format` is set (e.g., `<TASK={task}>, <DATASET={dataset}>`), assistant text is prefixed with that line plus a newline for **target** BBU/RRU samples (non-irrelevant only; sources are unchanged).
- For `irrelevant*` streams (e.g., `irrelevant_summary`, `irrelevant_dense`), the assistant payload is always the single line `无关图片` and the prefix is suppressed.
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
	      - name: roi_crop
	        params:
	          anchor_classes: ["BBU设备", "RRU设备", "紧固件"]
	          scale_range: [1.25, 2.3]
	          min_crop_size: 384
	          min_coverage: 0.4
	          completeness_threshold: 0.95
	          prob: 0.3
	      - name: rotate
	        params: { max_deg: 25.0, prob: 0.4 }
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

- **Detailed glossary / quick index**: `BBU_RRU_BUSINESS_KNOWLEDGE.md` (BBU/RRU 类别、属性、分组规则与高频标签模式)
- **Dense captioning** (training) enumerates every inspected item. Object types map to the attribute taxonomy in `data_conversion/hierarchical_attribute_mapping.json` and `data_conversion/attribute_taxonomy.json` (ignore the `occlusion` block). Core categories include:
  - `bbu` (BBU设备) — expects attributes such as brand (`bbu_brand`), completeness (`bbu_stituation`), and windshield requirement (`bbu_equipment`).
  - `bbu_shield` (挡风板) — brand, completeness, obstruction status, and installation direction.
  - `connect_point` (接地/接线端子), `label` (标签), plus `fiber` / `wire` lines for cabling.
  Geometry constraints from the mapping JSON enforce polys/bboxes for hardware and polylines for cabling.
- **Summary / Stage-A / Stage-B** (production) reuse the same records:
  - Stage-A generates per-image summaries (requires every record to carry a `summary`).
  - Stage-B consumes multiple images per site to emit a final deployment verdict.
  - Keep taxonomies in sync so Stage pipelines and dense training stay compatible.
- **Hierarchy semantics**: Attribute templates are comma‑separated `key=value` pairs (`类别` first, no spaces). Conditional attributes only appear when parent values require them. Free-text `备注`/OCR append at the end (whitespace stripped, punctuation preserved). RRU may include `组=<id>`; BBU never includes `组`.

When adding new inspection criteria, update both conversion JSON files and regenerate summaries before training to avoid drift between dense captions and production outputs.

### Domain Context: RRU Installation Inspection

RRU corpus covers remote radio unit inspection images (现场/弱电井/抱杆等)，强调“线缆有标签/有保护、紧固件合格、接地合格、站点距离数字化”等规则。Key points:

- **Core categories**: `RRU设备`, `站点距离`, `紧固件`, `RRU接地端`, `尾纤`, `接地线`, `标签` (see `BBU_RRU_BUSINESS_KNOWLEDGE.md` for a category→geometry→attribute index).
- **Grouping semantics (RRU-only)**: label-to-cable pairing is encoded via `desc` (`组=<id>`), not a top-level `groups` field; `summary` may include `分组统计` when groups exist. Conversion rejects groups with only one member.
- **Station distance**: represented as `类别=站点距离,站点距离=<int>` (digits), typically extracted/normalized upstream during conversion.
- **Audit pass/fail**: some RRU exports carry `审核通过/审核不通过` in the image path. If pass/fail becomes a first-class target, inject it into `metadata` during conversion/fusion rather than relying on path parsing.

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

- **Data index**: [README.md](README.md) - Quick map of `docs/data/`
- **Schema contract**: [DATA_JSONL_CONTRACT.md](DATA_JSONL_CONTRACT.md) - Global JSONL record contract
- **BBU/RRU glossary**: [BBU_RRU_BUSINESS_KNOWLEDGE.md](BBU_RRU_BUSINESS_KNOWLEDGE.md) - Domain categories, attributes, grouping rules
- **Augmentation**: [DATA_AUGMENTATION.md](DATA_AUGMENTATION.md) - Geometry-aware transforms
- **Fusion**: [UNIFIED_FUSION_DATASET.md](UNIFIED_FUSION_DATASET.md) - Multi-source mixing and policies
- **Training**: [REFERENCE.md](../training/REFERENCE.md#training) - Full training guide
- **Architecture**: [Architecture overview](../overview/ARCHITECTURE.md) - End-to-end pipeline
- **Upstream Models**: [UPSTREAM_DEPENDENCIES.md](../ops/UPSTREAM_DEPENDENCIES.md) - HF Qwen3-VL + ms-swift background

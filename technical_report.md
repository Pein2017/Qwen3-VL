# Technical Report: AI Group Quality Inspection Pipeline (Qwen3‑VL)

This report documents the end-to-end “group quality inspection” system in this repository: offline data preprocessing → supervised fine-tuning (SFT) training → two-stage inference (Stage‑A/Stage‑B) for mission-scoped group-level pass/fail verdicts. It focuses on **processes, stages, and workflows**; configuration syntax and CLI flags are intentionally de-emphasized.

Primary references:
- Documentation index and mapping: `docs/README.md`
- Training architecture: `docs/training/REFERENCE.md`
- Data preprocessing: `docs/data/DATA_PREPROCESSING_PIPELINE.md`
- JSONL contract: `docs/data/DATA_JSONL_CONTRACT.md`
- Stage‑A runtime: `docs/runtime/STAGE_A_RUNTIME.md`
- Stage‑B runtime: `docs/runtime/STAGE_B_RUNTIME.md`
- Two-stage business context: `docs/runtime/STAGE_A_STAGE_B.md`, `docs/stage-B-knowledge-Chinese.md`

---

## 1. Executive Summary

### 1.1 What the system does

The system reviews **missions** (inspection programs) for telecom equipment installation quality (BBU/RRU). Each **ticket/group** is a set of images for a single mission, labeled as pass/fail by human operations. The pipeline produces:

- **Training-ready JSONL** for dense-caption and summary-mode SFT (optional offline preprocessing).
- **Stage‑A evidence JSONL**: per-image structured summaries designed to cover rare/long-tail objects.
- **Stage‑B verdicts**: group-level binary decisions; the underlying *model output* is enforced to follow a strict two-line protocol:
  - `Verdict: 通过 / 不通过`
  - `Reason: ...` (single-line Chinese; no third-state wording)
  - Stage‑B records rule-search artifacts (`rule_candidates.jsonl`, `benchmarks.jsonl`, `rule_search_*`) and optional `distill_chatml.jsonl` after convergence.

Stage‑B is **training-free**: instead of fine-tuning per mission, it iteratively refines **mission guidance** via prompt-only rollouts and a rule-search gating loop (`src/stage_b/`).

> Update (2025‑12): Stage‑B runs rule_search only. Legacy selection/need-review artifacts and selection/signals modules are removed; any remaining legacy references below are historical.

Canonical mission names and focus definitions (used across Stage‑A and Stage‑B) live in `src/config/missions.py`:
- `BBU安装方式检查（正装）`
- `BBU接地线检查`
- `BBU线缆布放要求`
- `挡风板安装检查`

### 1.2 Architecture overview

```mermaid
flowchart LR
  A[Raw annotations + images] --> B[data_conversion/ (optional offline conversion)]
  B --> C[Training JSONL (docs/data/DATA_JSONL_CONTRACT.md)]
  C --> D[SFT training (src/sft.py)]
  D --> E[Qwen3‑VL checkpoint (+ optional LoRA)]
  E --> F[Stage‑A per-image summarization (src/stage_a/)]
  F --> G[Stage‑A evidence JSONL]
  G --> H[Stage‑B rule-search loop (src/stage_b/)]
  H --> I[rule_candidates.jsonl / benchmarks.jsonl]
  H --> J[guidance.json + snapshots (auditable)]
```

Key entrypoints (orchestration wrappers):
- Training: `scripts/train.sh` → `src/sft.py` (`docs/training/TRAINING_PLAYBOOK.md`, `docs/training/REFERENCE.md`)
- Stage‑A: `scripts/stage_a.sh` → `src/stage_a/cli.py` (`docs/runtime/STAGE_A_RUNTIME.md`)
- Stage‑B: `scripts/stage_b.sh` → `src/stage_b/runner.py` (`docs/runtime/STAGE_B_RUNTIME.md`)
- Dataset fusion (optional): `scripts/fuse_datasets.py` and runtime fusion loader (`docs/data/UNIFIED_FUSION_DATASET.md`)

---

## 2. Data Preprocessing Pipeline (Annotation → Training JSONL)

This stage is **optional** but recommended when upstream annotations change, new domains are added, or you need offline QA artifacts before training. The offline converter produces JSONL that matches the shared training contract consumed by `src/datasets/`.

References:
- Workflow: `docs/data/DATA_PREPROCESSING_PIPELINE.md`
- Contract: `docs/data/DATA_JSONL_CONTRACT.md`
- Orchestrator: `data_conversion/convert_dataset.sh`
- Main pipeline: `data_conversion/pipeline/unified_processor.py`

### 2.1 Inputs

Inputs are labeling-platform exports (JSON + images). The converter maintains taxonomy and formatting consistency via:
- Attribute taxonomy: `data_conversion/attribute_taxonomy.json`
- Hierarchical attribute mapping: `data_conversion/hierarchical_attribute_mapping.json`

The output `desc` strings are **hierarchical** and mission/domain-specific (Chinese for BBU/RRU). Stage‑A summaries and Stage‑B reasoning depend on this consistency.

### 2.2 Processing stages (module responsibilities)

The pipeline is organized around `UnifiedProcessor` (`data_conversion/pipeline/unified_processor.py`) and its components:

1. **Format normalization**
   - `data_conversion/pipeline/format_converter.py` converts upstream variants into a unified internal representation.

2. **Image & coordinate management (EXIF → resize → clamp)**
   - `data_conversion/pipeline/coordinate_manager.py` maintains pixel↔geometry alignment through:
     - EXIF orientation corrections
     - deterministic rescaling / “smart-resize” constraints
     - coordinate clamping and degenerate-geometry rejection

3. **Taxonomy normalization and hierarchical description building**
   - `data_conversion/pipeline/flexible_taxonomy_processor.py` builds the hierarchical `desc` strings and standardizes rare/long-tail labels.
   - `data_conversion/utils/sanitizer_pipeline.py` and `data_conversion/utils/sanitizers.py` implement text cleaning steps (sanitization, note stripping, standardized label description, etc.).

4. **Deterministic object ordering**
   - Objects are sorted top-left to bottom-right using `data_conversion/utils/sorting.py` (`sort_objects_tlbr`). This ordering is reused in training message construction (`src/datasets/builders/jsonlines.py`) to keep indices stable.

5. **Validation and QA artifacts**
   - `data_conversion/pipeline/validation_manager.py` enforces core invariants (bounds, size thresholds, required `desc` when strict, etc.) and emits reports. The pipeline is designed to **report** issues explicitly rather than silently re-writing data.

6. **Summary generation**
   - `data_conversion/pipeline/summary_builder.py` produces the per-image `summary` field used in summary-mode SFT. The active implementation is **raw-desc grouping**:
     - group by *raw* `desc` (no canonicalization)
     - merge identical entries into `desc×N`
     - sort by `len(desc)` ascending with first-appearance tie-break
     - fail-fast if objects are missing or `desc` is empty

7. **Deterministic splitting and tiny subsets**
   - `data_conversion/pipeline/data_splitter.py` produces train/val splits using a fixed seed and can also emit `*_tiny.jsonl` for smoke testing.

### 2.3 Geometry canonicalization (why it matters)

This project treats geometry as a **first-class, preserved signal** (for training-time grounding and augmentation). Canonicalization ensures consistent polygon interpretation across converters, visualization tools, and augmentation.

Core behaviors:
- Canonical geometry keys are `bbox_2d`, `poly`, `line` (one per object), as specified in `docs/data/DATA_JSONL_CONTRACT.md`.
- Polygon vertex ordering is canonicalized offline in `data_conversion/pipeline/coordinate_manager.py` via `CoordinateManager.canonical_poly_ordering(...)`:
  - remove duplicate closing points
  - sort vertices clockwise around the centroid
  - rotate so the first vertex is the top-most (then left-most)
  - reject degenerate polygons (e.g., duplicate vertices) rather than silently accepting them

This canonicalization prevents downstream issues like self-crossing polygons and inconsistent visualization/truncation during augmentation.

### 2.4 Output format: the shared JSONL contract

The converter emits JSONL records that follow `docs/data/DATA_JSONL_CONTRACT.md` and are validated by `scripts/validate_dense_jsonl_contract.py`.

Key invariants (contract-level):
- Top-level keys: `images`, `objects`, `width`, `height`, optional `summary`, optional `metadata`.
- Each object has **exactly one** geometry key among `bbox_2d`, `poly`, `line` and must have a non-empty `desc`.
- Group membership for RRU/BBU is encoded in `desc` using a prefix like `组1:`. There is **no top-level** `groups` field (`docs/data/DATA_JSONL_CONTRACT.md`).

#### Example: minimal (demo) record

From `demo/data/train_tiny.jsonl`:

```json
{"images":["../images/QC-20230106-0000211_16517.jpeg"],"objects":[{"bbox_2d":[48,76,312,428],"desc":"设备/示例"},{"poly":[360,120,480,120,480,260,360,260],"poly_points":4,"desc":"标签/示例"}],"summary":"设备×1，标签×1","width":532,"height":728}
```

#### Example: BBU record with mixed geometry + summary

From `data/bbu_full_768_poly/train_tiny.jsonl` (first line):

```json
{
  "images": [
    "images/QC-20230222-0000297_272976.jpeg"
  ],
  "objects": [
    {
      "poly": [0, 190, 438, 230, 401, 622, 0, 614],
      "desc": "BBU设备/华为,只显示部分,机柜空间充足需要安装/这个BBU设备按要求配备了挡风板,备注:无法判断品牌"
    },
    {
      "bbox_2d": [443, 226, 491, 286],
      "desc": "螺丝、光纤插头/BBU安装螺丝,符合要求"
    },
    {
      "bbox_2d": [429, 426, 460, 458],
      "desc": "螺丝、光纤插头/机柜处接地螺丝,符合要求"
    },
    {
      "line": [453, 469, 470, 572, 501, 630, 529, 670, 611, 726],
      "desc": "电线/捆扎整齐"
    },
    {
      "bbox_2d": [411, 544, 450, 587],
      "desc": "螺丝、光纤插头/BBU安装螺丝,符合要求"
    },
    {
      "poly": [481, 553, 661, 544, 644, 618, 507, 624],
      "desc": "标签/5GBBU接地线"
    },
    {
      "poly": [0, 615, 385, 630, 377, 793, 0, 801],
      "desc": "挡风板/华为,只显示部分,安装方向正确,备注:无法判断品牌和安装方向是否正确"
    },
    {
      "poly": [157, 898, 252, 895, 260, 1023, 163, 1023],
      "desc": "标签/无法识别"
    },
    {
      "poly": [307, 1021, 354, 956, 414, 1023, 309, 1023],
      "desc": "标签/无法识别"
    }
  ],
  "summary": "BBU设备/华为/只显示部分×1，螺丝、光纤插头/BBU安装螺丝/符合要求×2，螺丝、光纤插头/机柜处接地螺丝/符合要求×1，电线/捆扎整齐×1，标签/可以识别×1，标签/无法识别×2，挡风板/只显示部分/安装方向正确×1，备注: 无法判断品牌；无法判断品牌和安装方向是否正确",
  "width": 768,
  "height": 1024
}
```

#### Example: RRU record with group-encoded membership (`组<id>:`)

From `data/rru_full_1024_poly/train_tiny.jsonl` (second line):

```json
{
  "images": [
    "images/审核通过/QC-20240424-0028974/QC-20240424-0028974_3119298.jpeg"
  ],
  "objects": [
    { "bbox_2d": [37, 269, 79, 298], "desc": "站点距离/98" },
    { "line": [122, 1173, 105, 799, 245, 703, 235, 269], "desc": "组2:接地线/有标签" },
    { "line": [40, 1111, 16, 833, 165, 540, 138, 118], "desc": "组1:尾纤/有标签,有套管保护" },
    { "poly": [128, 840, 86, 852, 104, 954, 149, 941], "desc": "组2:标签/900M-RRU2-接地" },
    { "poly": [53, 843, 5, 862, 53, 966, 94, 945], "desc": "组1:标签/900M-RRU2-光纤" }
  ],
  "summary": "站点距离/98，组2:接地线/有标签，组1:尾纤/有标签,有套管保护，组2:标签/900M-RRU2-接地，组1:标签/900M-RRU2-光纤",
  "width": 672,
  "height": 1504
}
```

---

## 3. Training Pipeline (SFT)

Training is a **config-first** SFT pipeline designed to learn:
- **Dense captioning**: geometry-grounded object descriptions (bbox/poly/line + hierarchical `desc`).
- **Summary mode**: single-line per-image summaries consistent with production formatting rules.

References:
- Implementation entrypoint: `src/sft.py`
- Config loading and validation: `src/config/loader.py`, `src/config/schema.py`
- Dataset builders: `src/datasets/`
- Training documentation: `docs/training/REFERENCE.md`, `docs/training/TRAINING_PLAYBOOK.md`

### 3.1 Config-first training lifecycle

The training runner (`src/sft.py`) intentionally minimizes runtime CLI surface and treats YAML as the source of truth:

1. **Load + validate configuration**
   - `src/config/loader.py::ConfigLoader.load_training_config(...)` loads YAML (with inheritance), resolves prompts, and materializes typed dataclasses (`TrainingConfig`) via `src/config/schema.py`.
   - Training fails fast if the configuration requests deprecated features (e.g., packing is explicitly rejected in `src/sft.py`).

2. **Build datasets (direct or fused)**
   - Direct JSONL: `src/datasets/dense_caption.py::BaseCaptionDataset.from_jsonl(...)`
   - Fusion (target + auxiliary sources): `src/datasets/unified_fusion_dataset.py::FusionCaptionDataset` with config types in `src/datasets/fusion.py` and design notes in `docs/data/UNIFIED_FUSION_DATASET.md`.

   #### Fusion dataset: target vs source domains (and why it exists)

   Fusion is the training-time mechanism for balancing two competing goals:
   - **Specialize on the domain we care about** (the *target* domain: mission-focused BBU/RRU inspection data).
   - **Avoid catastrophic forgetting / capability drift** while specializing (the *source* domain: auxiliary vision-language and language-only data mixed at controlled ratios).

   Concretely in this repo:
   - **Target-domain datasets** (domain = `target`): the inspection datasets we optimize for (e.g., BBU/RRU dense captioning and summary-mode SFT). In the runtime fusion loader, targets are eligible for geometry-aware augmentation and curriculum when configured, are typically *uncapped* in object count, and are the only datasets evaluated in fusion-mode validation (target-only eval policy in `src/datasets/unified_fusion_dataset.py`).
   - **Source-domain datasets** (domain = `source`): auxiliary datasets mixed to preserve general grounding/language competence and stabilize training (e.g., LVIS/COCO-style detection captioning, or chat-style JSONL with pre-authored `messages`). In the runtime fusion loader, sources default to **no augmentation/curriculum** and typically have an object cap (default 64 via `src/datasets/wrappers/__init__.py`) to keep sequence lengths bounded.

   How it is implemented (workflow level):
   - **Domain assignment**: fusion entries are converted into `DatasetSpec` via wrappers in `src/datasets/wrappers/`, which set `domain: target|source` and default policies (e.g., source caps, source no-augmentation).
   - **Deterministic per-epoch mixing**: `src/datasets/unified_fusion_dataset.py::FusionCaptionDataset` builds an epoch schedule where:
     - targets contribute “self-scaled” quotas based on their pool size (and optional per-target ratio; implemented in `src/datasets/fusion.py::_compute_target_quotas`), and
     - each source contributes a quota proportional to the total target quota (`round(source.ratio * N_target)`), sampled deterministically with optional “without replacement when possible, otherwise fall back to replacement”.
   - **Per-sample routing**: each record is tagged with `_fusion_source`, `_fusion_domain`, `_fusion_template`, and `_fusion_mode` in `metadata`, enabling per-dataset prompt selection and per-dataset metrics.
   - **Prompt priority**: fusion resolves prompts with a clear precedence (global defaults → template/domain prompts → per-dataset overrides) via `src/config/prompts.py` and `FusionCaptionDataset._resolve_prompts(...)`.
   - **Telemetry and metrics**: the training collator attaches `dataset_labels` derived from `_fusion_source` (`src/data_collators/dataset_metrics.py`), enabling per-dataset/per-domain metrics and debugging (useful to confirm sources are doing their “anti-forgetting” job without dominating training).

3. **Prepare the model (LoRA/adapters) before trainer construction**
   - `src/sft.py` calls `SwiftSft.prepare_model(...)` before creating the trainer, ensuring adapter state is correct for optimization and checkpointing (`docs/training/REFERENCE.md`).

4. **Train with padded batches**
   - Packing is removed from this runtime; the system uses padded batches only and enforces this by validation in `src/sft.py` (and in docs).

### 3.2 Dataset builders and message formatting (dense vs summary)

All training datasets ultimately produce **chat-style messages** that the Qwen3‑VL chat template encodes.

Core components:
- Record validation: `src/datasets/contracts.py::validate_conversation_record(...)`
- Conversation builder: `src/datasets/builders/jsonlines.py::JSONLinesBuilder`
- Dataset wrapper: `src/datasets/dense_caption.py::BaseCaptionDataset` (aliased as `DenseCaptionDataset`)

**Dense mode workflow**
1. Load a JSONL record (`images`, `objects`, `width`, `height`).
2. Sort objects deterministically (`data_conversion/utils/sorting.py`, `sort_objects_tlbr`).
3. Build a single-turn conversation:
   - user: `[image, prompt]`
   - assistant: JSON mapping `object_1`, `object_2`, … to `{desc, geometry...}` (`JSONLinesBuilder._build_group_entry`).
4. Attach top-level `objects` metadata (pixel-space points) for template-side normalization and downstream tooling.

**Summary mode workflow**
1. Load a JSONL record with a non-empty `summary`.
2. Build a single-turn conversation:
   - user: `[image, summary_prompt]`
   - assistant: the summary string (single line; no coordinates)

### 3.3 Template encoding and token flow (vision placeholders)

This project relies on the model’s native chat template (ms-swift + HF tokenizer) for vision token insertion and masking. Key behaviors (see `docs/training/REFERENCE.md` and `docs/README.md`):
- Templates automatically insert image placeholders; do not hand-craft `<|image_pad|>` tokens.
- Geometry stays in pixel space on disk; templates and builders handle normalization (e.g., norm1000) at encoding time.

### 3.4 Augmentation pipeline (geometry-preserving)

Augmentation is applied as a **preprocessor** that transforms both the image and its geometry in sync:
- Augmentation ops and Compose pipeline: `src/datasets/augmentation/` (`builder.py`, `base.py`, `ops.py`)
- Core geometry transforms and clipping utilities: `src/datasets/geometry.py`
- Augmentation guide: `docs/data/DATA_AUGMENTATION.md`

Design goals:
- Preserve pixel↔geometry alignment through affine transforms (rotate/flip/scale) and size-changing ops (crop/resize).
- Robust polygon/line handling (clipping and coverage thresholds) to avoid training on misaligned labels.
- Support curriculum scheduling of augmentation intensity via `src/datasets/augmentation/curriculum.py` (and its integration in `src/sft.py`).

### 3.5 Fail-fast behavior and reproducibility

The training pipeline enforces a number of correctness and reproducibility guardrails:
- Typed config validation (`src/config/schema.py`) and early runtime checks (`src/sft.py`).
- Dataset-level validation:
  - summary mode requires a non-empty `summary`
  - dense mode requires at least one object and at least one geometry field per object (`BaseCaptionDataset.__getitem__` in `src/datasets/dense_caption.py`)
- Over-length safety:
  - If the template is configured to raise on overflow, `BaseCaptionDataset` catches `MaxLengthError` and **drops** the sample instead of truncating (`src/datasets/dense_caption.py`), preserving detail rather than silently clipping.
- Deterministic sampling:
  - `BaseCaptionDataset` uses epoch-seeded RNG with worker-aware mixing (`src/datasets/dense_caption.py`).

### 3.6 Logging and telemetry

Rank-aware logging is standardized via `src/utils/logger.py`:
- `get_logger(...)` emits only from rank 0 by default, with opt-in verbose mode for all ranks.
- This is used throughout training and inference entrypoints (`src/sft.py`, `src/stage_a/`, `src/stage_b/`).

---

## 4. Stage‑A Inference (Per-image Object Recognition & Summarization)

Stage‑A generates **per-image evidence summaries** from raw images, producing an evidence JSONL that Stage‑B consumes.

References:
- Code: `src/stage_a/inference.py`, `src/stage_a/prompts.py`, `src/stage_a/cli.py`
- Runtime doc: `docs/runtime/STAGE_A_RUNTIME.md`
- Mission definitions: `src/config/missions.py`

### 4.1 Input discovery and ticket semantics

Stage‑A discovers groups from a mission-based directory structure (see `discover_groups` in `src/stage_a/inference.py`):

```
<root>/<mission>/{审核通过|审核不通过}/<group_id>/*.{jpg,jpeg,png}
```

Key semantics:
- Labels are inferred from the folder name: `审核通过 → pass`, `审核不通过 → fail` (`LABEL_DIR_MAP` in `src/stage_a/inference.py`).
- Resubmissions are allowed: the same `group_id` can exist under both label folders; Stage‑A emits one record per occurrence (matching the provided label).

### 4.2 Mission-aware prompting (runtime vs training)

Stage‑A uses a **runtime** summary prompt that is richer than the training-minimal summary prompt:
- System prompt: `src/stage_a/prompts.py::SUMMARY_SYSTEM_PROMPT` (derived from `src/config/prompts.py::SYSTEM_PROMPT_SUMMARY_RUNTIME`)
- User prompt builder: `src/stage_a/prompts.py::build_user_prompt(...)`
  - appends a “drawing/document image” guardrail (force `无关图片` for blueprint/CAD-like images)
  - optionally appends a mission focus hint from `src/config/missions.py::STAGE_A_MISSION_FOCUS`

This separation keeps SFT from overfitting to business priors while Stage‑A runtime still gets operational guardrails.

### 4.3 Output schema and validation

Stage‑A writes streaming JSONL records (one per group) with strict coverage validation (`process_group` in `src/stage_a/inference.py`):
- `group_id` (string)
- `mission` (string)
- `label` (`pass|fail`)
- `images` (list of filenames)
- `per_image` (object mapping `image_1`, `image_2`, … to summary strings)

Example record from `output_post/stage_a/挡风板安装检查_stage_a.jsonl`:

```json
{
  "group_id": "QC-20231218-0025165",
  "mission": "挡风板安装检查",
  "label": "pass",
  "images": [
    "QC-20231218-0025165_4127773.jpeg",
    "QC-20231218-0025165_4127774.jpeg",
    "QC-20231218-0025165_4127784.jpeg"
  ],
  "per_image": {
    "image_1": "BBU设备/需复核,备注:无法判断品牌,从空间角度看,无法判断是否足够空间安装挡风板×1，螺丝、光纤插头/BBU安装螺丝,符合要求×1，BBU设备/华为,备注:只显示部分,拍摄角度原因无法框选完整×1",
    "image_2": "电线/捆扎整齐×1，标签/5G-BBU-接地线×1，螺丝、光纤插头/机柜处接地螺丝,符合要求×1",
    "image_3": "标签/无法识别×3，BBU设备/华为,只显示部分,无需安装×1，螺丝、光纤插头/BBU安装螺丝,符合要求×2，螺丝、光纤插头/BBU端光纤插头,符合要求×3，挡风板/华为,显示完整,安装方向正确×1，BBU设备/华为,显示完整,机柜空间充足需要安装/这个BBU设备按要求配备了挡风板×1"
  }
}
```

Notes:
- Stage‑A attempts to sanitize model outputs that accidentally return JSON objects (it extracts `image_1`/`图片_1` or joins string values) while otherwise preserving raw content (`sanitize_single_image_summary` in `src/stage_a/inference.py`).
- Stage‑A applies EXIF orientation fixes (`data_conversion/utils/exif_utils.py`, `apply_exif_orientation`) before encoding images.

### 4.4 Operational verification hooks

Stage‑A supports optional verification logging to debug token/grid alignment and image hashing (see `infer_one_image` / `infer_batch` in `src/stage_a/inference.py`). This is intended to catch:
- broken image decoding
- incorrect processor pixel budgets
- mismatched image token counts vs `image_grid_thw`

---

## 5. Stage‑B Inference (Rule-Search, Training-Free)

Stage‑B consumes Stage‑A evidence summaries and generates mission-scoped group verdicts. It is “training-free”: learning happens by **updating mission guidance** through a rule-search loop instead of weight updates.

References:
- Orchestrator: `src/stage_b/runner.py`
- Config schema: `src/stage_b/config.py`
- Ingestion: `src/stage_b/ingest/stage_a.py`
- Prompt construction: `src/stage_b/sampling/prompts.py`
- Rollout sampler + parser: `src/stage_b/rollout.py`
- Rule-search metrics + gates: `src/stage_b/rule_search.py`
- Guidance repository + snapshots: `src/stage_b/io/guidance.py`
- Runtime doc: `docs/runtime/STAGE_B_RUNTIME.md`

### 5.1 Ingest: Stage‑A JSONL → tickets

Stage‑B loads Stage‑A JSONL files and normalizes per-image keys (`image_1`, `image_2`, …) to build `GroupTicket` objects:
- `src/stage_b/ingest/stage_a.py::ingest_stage_a(...)`
- Tickets are uniquely keyed by `uid = "{group_id}::{label}"` to allow resubmissions under different labels.

### 5.2 Prompting: guidance + summaries + strict output protocol

Stage‑B prompts are assembled as a system+user message pair:
- `src/stage_b/sampling/prompts.py::build_system_prompt(...)`
  - embeds the mission’s `G0` focus (the *only* “task points” the model should use)
  - enforces the strict two-line output protocol
- `src/stage_b/sampling/prompts.py::build_user_prompt(...)`
  - includes per-image summaries and a derived `ImageN(obj=...)` statistic (computed from `×N` counts or fallback heuristics)
  - includes additional experiences (`G1`, `G2`, …) as “补充提示”

Important compatibility detail:
- Stage‑A summaries may contain a soft marker like `需复核,备注:`. Stage‑B sanitizes this marker out of the prompt to avoid forbidden third-state tokens appearing in model outputs (`_sanitize_stage_a_summary_for_prompt` in `src/stage_b/sampling/prompts.py`), while preserving the remark content (`备注:`).

### 5.3 Rollout: prompt-only candidate generation + parsing

Stage‑B generates multiple candidate responses per ticket:
- `src/stage_b/rollout.py::RolloutSampler.generate_for_batch(...)`
- It renders prompts with the tokenizer’s chat template and explicitly disables “thinking” blocks (`enable_thinking=False`) to keep outputs stable.
- Each candidate is parsed by `src/stage_b/rollout.py::_parse_two_line_response(...)`:
  - must be exactly two non-empty lines: `Verdict: ...` and `Reason: ...`
  - verdict must be binary (`通过`/`不通过` → `pass|fail`)
  - reason must be non-empty and **must not** contain third-state phrases (e.g., `需复核`, `证据不足`, `待定`, etc.)
  - parsing failures are recorded as `format_ok = false` and are excluded from rule-search scoring

### 5.4 Rule-search loop: propose → gate → apply

Rule-search grows guidance using metric-gated candidates:
- Baseline rollouts on the **train pool** establish ticket-level majority stats and metrics.
- A proposer emits 1–N rule candidates (prompted by hard cases and coverage gaps).
- Each candidate is gated with deterministic metrics and bootstrap probability (relative error reduction, changed-fraction, etc.; see `src/stage_b/rule_search.py`).
- Optional eval-pool auditing verifies the gated candidate before application.
- Accepted candidates are merged into mission guidance and checkpointed via snapshots.

### 5.5 Artifacts emitted by Stage‑B (rule-search)

Per mission, Stage‑B writes a run directory layout (see `docs/runtime/STAGE_B_RUNTIME.md` and `src/stage_b/runner.py`):
- `rule_candidates.jsonl`: candidate rules with gating metrics and decisions
- `benchmarks.jsonl`: per-epoch train/eval metrics and summaries
- `rule_search_hard_cases.jsonl`: hard tickets mined during rule-search
- `rule_search_candidate_regressions.jsonl`: regressions discovered during candidate evaluation
- `guidance.json` + `snapshots/`: evolving, auditable mission guidance state
- `distill_chatml.jsonl` (optional): low‑temperature ChatML distillation samples emitted after early stop

### 5.6 Distributed execution model (single node)

Stage‑B supports “ticket-parallel rollout” using distributed execution (see `docs/runtime/STAGE_B_RUNTIME.md` and `src/stage_b/runner.py`):
- Each rank runs rollouts on a shard of tickets; the model is replicated per rank.
- Rule-search aggregation is centralized on rank 0; only rank 0 writes the final artifacts.
- In distributed mode, device placement is forced to per-rank single-GPU to avoid accidental model-parallel sharding (`_load_model` in `src/stage_b/runner.py`).

## 6. End-to-End Workflow (Raw Data → Final Verdict)

This section ties the stages together as an operational workflow.

### 6.1 Pipeline walk-through

1. **Ingest raw annotations (optional offline conversion)**
   - Run the offline pipeline under `data_conversion/` to generate train/val JSONL and QA reports.
   - Validate outputs against `docs/data/DATA_JSONL_CONTRACT.md` (use `scripts/validate_dense_jsonl_contract.py`).

2. **Train / update the shared Qwen3‑VL checkpoint**
   - Run SFT training via `src/sft.py` (wrapped by `scripts/train.sh`).
   - Training may be direct JSONL or fusion-based; augmentation is applied only where configured and is geometry-aware (`docs/data/DATA_AUGMENTATION.md`).

3. **Run Stage‑A to generate evidence JSONL**
   - Organize incoming ticket images in mission folders (pass/fail splits).
   - Run Stage‑A (`src/stage_a/`) to emit per-image summaries as JSONL.

4. **Run Stage‑B to produce group verdicts and refine guidance**
   - Ingest Stage‑A JSONL into Stage‑B.
   - For each ticket:
     - rollout N candidates under guidance
     - parse strict two-line outputs
     - score and gate rule candidates via rule-search metrics
   - Apply accepted candidates to guidance and snapshot changes per epoch.

5. **Review rule-search outputs and governance**
   - Review `rule_search_hard_cases.jsonl` and `rule_search_candidate_regressions.jsonl` for problematic tickets.
   - Audit `benchmarks.jsonl` for train/eval trend drift and gated candidate stats.
   - Promote guidance changes from run-local `guidance.json` back to the shared guidance seed only after approval (auditable via snapshots).

### 6.2 Artifact handoffs (what feeds what)

```mermaid
flowchart TD
  subgraph Offline["Offline (optional)"]
    A1[Raw exports + images] --> A2[data_conversion/pipeline]
    A2 --> A3[train/val JSONL + QA reports]
  end
  subgraph Training["Training"]
    A3 --> B1[src/datasets/ + template encode]
    B1 --> B2[src/sft.py (SwiftSft)]
    B2 --> B3[checkpoint (+LoRA)]
  end
  subgraph Runtime["Runtime"]
    B3 --> C1[Stage‑A: src/stage_a]
    C1 --> C2[Stage‑A evidence JSONL]
    C2 --> D1[Stage‑B: src/stage_b]
    D1 --> D2[rule_candidates.jsonl + benchmarks.jsonl]
    D1 --> D3[guidance.json + snapshots]
    D1 --> D4[rule_search_hard_cases.jsonl + candidate_regressions]
  end
```

---

## 7. Key Design Principles (and where they are enforced)

### 7.1 Config-first architecture + typed validation
- Training configs are loaded and validated early as frozen dataclasses (`src/config/schema.py`, `src/config/loader.py`).
- Stage‑B configs are similarly loaded into frozen dataclasses (`src/stage_b/config.py`).
- Scripts in `scripts/` exist to standardize environment/logging and reduce ad-hoc divergence (`scripts/README.md`).

### 7.2 Determinism and reproducibility
- Offline conversion uses deterministic splitting seeds (`data_conversion/pipeline/data_splitter.py`).
- Training datasets use epoch-seeded shuffling and worker-aware seeding (`src/datasets/dense_caption.py`).
- Stage‑B seeds Python/NumPy/Torch RNGs (`src/stage_b/utils/seed.py`) and shuffles tickets per epoch with `seed + epoch` (`src/stage_b/runner.py`).
- Rollout decode seeds can be set per decode-grid entry; repeated runs remain reproducible when seeds are fixed (`src/stage_b/rollout.py`).

### 7.3 Fail-fast validation and safe fallbacks
- Data conversion validates structure, bounds, and required fields and emits explicit invalid-object/sample reports (`data_conversion/pipeline/validation_manager.py`).
- Training fails fast on unsupported runtime features (e.g., packing) and on schema violations (missing summary in summary mode, missing geometry in dense mode).
- Stage‑B enforces strict output formatting; malformed candidates are excluded from rule-search metrics (`src/stage_b/rollout.py`, `docs/runtime/STAGE_B_RUNTIME.md`).

### 7.4 Geometry-preserving augmentation (no silent geometry loss)
- Augmentation is designed to transform geometry and pixels together (`src/datasets/geometry.py`, `src/datasets/augmentation/ops.py`).
- Crop/rotate/truncation logic is built to maintain correctness of polygons and lines (see `docs/data/DATA_AUGMENTATION.md`).
- Offline polygon ordering canonicalization prevents downstream ambiguity (`data_conversion/pipeline/coordinate_manager.py`).

### 7.5 Training-free Stage‑B refinement with governance hooks
- Stage‑B’s “learning” happens by updating a guidance file (experiences), not weights (`src/stage_b/io/guidance.py`).
- Rule-search candidates are gated by deterministic metrics and bootstrap probability before application (`src/stage_b/rule_search.py`).
- Changes are auditable via step counters and snapshot retention; hard cases and regressions are recorded for review (`rule_search_hard_cases.jsonl`, `rule_search_candidate_regressions.jsonl`).

### 7.6 Rank-aware logging (distributed-safe)
- All core pipelines use `src/utils/logger.py` (`get_logger(...)`) to keep logs readable in distributed runs and to avoid global logging side effects.

### 7.7 Change management (OpenSpec, optional but recommended)

For major behavior/contract changes (schema shifts, Stage‑B policy changes, augmentation refactors), the repository uses OpenSpec governance under `openspec/`:
- Instructions and conventions: `openspec/AGENTS.md`
- Historical changes (including Stage‑B evolution) are tracked under `openspec/changes/` and `openspec/changes/archive/`.

---

## Appendix: “Where to look” quick map

- Data conversion: `data_conversion/pipeline/unified_processor.py`, `docs/data/DATA_PREPROCESSING_PIPELINE.md`
- JSONL contract and validators: `docs/data/DATA_JSONL_CONTRACT.md`, `scripts/validate_dense_jsonl_contract.py`
- Training pipeline: `src/sft.py`, `src/config/`, `src/datasets/`, `docs/training/REFERENCE.md`
- Augmentation and geometry: `src/datasets/augmentation/`, `src/datasets/geometry.py`, `docs/data/DATA_AUGMENTATION.md`
- Stage‑A: `src/stage_a/`, `docs/runtime/STAGE_A_RUNTIME.md`
- Stage‑B: `src/stage_b/`, `docs/runtime/STAGE_B_RUNTIME.md`, `docs/runtime/STAGE_A_STAGE_B.md`, `docs/stage-B-knowledge-Chinese.md`

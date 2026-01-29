# Technical Report: AI Group Quality Inspection Pipeline (Qwen3‑VL)

This report documents the end-to-end “group quality inspection” system in this repository: offline data preprocessing → **two-stage training (SFT → GRPO post-training)** → two-stage inference (Stage‑A/Stage‑B) for mission-scoped group-level pass/fail verdicts. It focuses on **processes, stages, and workflows**; configuration syntax and CLI flags are intentionally de-emphasized.

Scope emphasis:
- Stage‑B rule-search (tree‑growth / decision‑tree‑like exploration) as the primary training‑free optimization path.
- Data contracts that bind the pipeline end‑to‑end: dense-caption JSONL, per-image summaries, Stage‑A evidence JSONL, and Stage‑B guidance/artifacts.

This report is **self‑contained**: assumptions, interfaces, and behaviors are described in place without relying on external documents.

---

## 1. Executive Summary

### 1.1 What the system does

The system reviews **missions** (inspection programs) for telecom equipment installation quality (BBU/RRU). Each **ticket/group** is a set of images for a single mission, labeled as pass/fail by human operations. The pipeline produces:

- **Training-ready JSONL** for dense-caption and summary-mode training (consumed by both SFT and GRPO post-training).
- **A shared Qwen3‑VL checkpoint** produced by SFT → GRPO post-training (typically adapter-based and then merged for inference).
- **Stage‑A evidence JSONL**: per-image structured summaries designed to cover rare/long-tail objects.
- **Stage‑B verdicts**: group-level binary decisions; the underlying *model output* is enforced to follow a strict two-line protocol:
  - `Verdict: 通过 / 不通过`
  - `Reason: ...` (single-line Chinese; no third-state wording)
  - Stage‑B records rule‑search artifacts (`rule_candidates.jsonl`, `benchmarks.jsonl`, `rule_search_*`), run‑local `guidance.json` + snapshots, and optional `distill_chatml.jsonl`.

Stage‑B is **training‑free**: instead of fine‑tuning per mission, it refines **mission guidance** via rule‑search (proposer‑generated candidates + metric gates) rather than unconstrained self‑reflection.

> Update (2026‑01): Stage‑B executes `rule_search` only. Legacy selection/need‑review loops are inactive; compatibility types (e.g., deterministic signals) remain but are not used for selection.

Canonical mission names and focus definitions (used across Stage‑A and Stage‑B) live in `src/config/missions.py`:
- `BBU安装方式检查（正装）`
- `BBU接地线检查`
- `BBU线缆布放要求`
- `挡风板安装检查`

### 1.2 Architecture overview

```mermaid
flowchart LR
  A[Raw annotations + images] --> B[data_conversion/ (optional offline conversion)]
  B --> C[Training JSONL (dense + summary contract)]
  C --> D[SFT (src/sft.py / SwiftSft)]
  D --> E[SFT checkpoint (+LoRA/DoRA)]
  E --> F[GRPO post-training (src/sft.py / SwiftRLHF, rlhf_type=grpo)]
  F --> G[Qwen3‑VL checkpoint (GRPO, merged or adapter)]
  G --> H[Stage‑A per-image summarization (src/stage_a/)]
  H --> I[Stage‑A evidence JSONL]
  I --> J[Stage‑B rule-search loop (src/stage_b/)]
  J --> K[rule_candidates.jsonl / benchmarks.jsonl]
  J --> L[guidance.json + snapshots (auditable)]
```

Key entrypoints (orchestration wrappers):
- Training (SFT/GRPO): `scripts/train.sh` → `src/sft.py` (SwiftSft or SwiftRLHF)
- GRPO server-rollout (server mode): `scripts/grpo_server_mode.sh` (unified; server background + learner foreground) → `swift rollout` + `src/sft.py`
- Stage‑A: `scripts/stage_a.sh` → `src/stage_a/cli.py`
- Stage‑B: `scripts/stage_b.sh` → `src/stage_b/runner.py`
- Dataset fusion (optional): `scripts/fuse_datasets.py` → `src/datasets/unified_fusion_dataset.py`

### 1.3 Data contracts at a glance

- **Dense‑caption training JSONL**: `images`, `width`, `height`, and `objects[]` with `desc` + one geometry (`bbox_2d|poly|line`). Used for geometry‑grounded captioning.
- **Summary‑mode training JSONL**: requires a non‑empty `summary` JSON string (single‑line). Irrelevant-image pools use the literal `无关图片`. Objects may be present but are not required for summary‑only training.
- **Stage‑A evidence JSONL**: `mission`, `group_id`, `label` (`pass|fail`), `images[]`, `per_image{image_i: summary}`. This is the sole input to Stage‑B.
- **Stage‑B guidance + artifacts**: `guidance.json` (per‑mission experiences + metadata) plus rule‑search artifacts (`rule_candidates.jsonl`, `benchmarks.jsonl`, `rule_search_*`, optional `distill_chatml.jsonl`).

---

## 2. Data Preprocessing Pipeline (Annotation → Training JSONL)

This stage is **optional** but recommended when upstream annotations change, new domains are added, or offline QA artifacts are needed before training. The converter produces JSONL that matches the shared training contract consumed by `src/datasets/`.

Core components:
- Orchestrator: `data_conversion/convert_dataset.sh`
- Main pipeline: `data_conversion/pipeline/unified_processor.py`

### 2.1 Inputs

Inputs are labeling-platform exports (JSON + images). The converter maintains taxonomy and formatting consistency via:
- Attribute taxonomy: `data_conversion/attribute_taxonomy.json`
- Hierarchical attribute mapping: `data_conversion/hierarchical_attribute_mapping.json`

The output `desc` strings are **hierarchical key=value pairs** and mission/domain-specific (Chinese for BBU/RRU). Stage‑A summaries and Stage‑B reasoning depend on this consistency.

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
   - `data_conversion/pipeline/summary_builder.py` produces a **JSON-string** `summary` with per-category stats:
     - required key: `统计`
     - BBU includes `备注` only when non-empty; RRU may include `分组统计`
     - `dataset` and `异常` keys are **not emitted** (and are treated as contract violations in training corpora)
   - Only observed values are counted (no missing/third-state/遮挡 placeholders).
   - OCR/备注 are free text: whitespace removed only; punctuation preserved; unreadable → `可读性=不可读`.
   - Fail-fast if objects are missing or any `desc` is empty; irrelevant-image samples keep `summary: 无关图片` and bypass the builder.

7. **Deterministic splitting and tiny subsets**
   - `data_conversion/pipeline/data_splitter.py` produces train/val splits using a fixed seed and can also emit `*_tiny.jsonl` for smoke testing.

### 2.3 Geometry canonicalization (why it matters)

This project treats geometry as a **first-class, preserved signal** (for training-time grounding and augmentation). Canonicalization ensures consistent polygon interpretation across converters, visualization tools, and augmentation.

Core behaviors:
- Canonical geometry keys are `bbox_2d`, `poly`, `line` (one per object).
- Polygon vertex ordering is canonicalized offline in `data_conversion/pipeline/coordinate_manager.py` via `CoordinateManager.canonical_poly_ordering(...)`:
  - remove duplicate closing points
  - sort vertices clockwise around the centroid
  - rotate so the first vertex is the top-most (then left-most)
  - reject degenerate polygons (e.g., duplicate vertices) rather than silently accepting them

This canonicalization prevents downstream issues like self-crossing polygons and inconsistent visualization/truncation during augmentation.

### 2.4 Output format: the shared JSONL contract

The converter emits JSONL records that follow the shared training contract described in this report and can be validated by `scripts/validate_dense_jsonl_contract.py`.

Key invariants (contract-level):
- Top-level keys: `images`, `objects`, `width`, `height`, optional `summary`, optional `metadata`.
- Dense mode requires a non-empty `objects` list; each object must include `desc` and **exactly one** geometry key among `bbox_2d`, `poly`, `line` (flat even-length list of x/y values). `quad` is rejected in favor of `poly`.
- Summary mode requires a non-empty `summary` JSON string (single line) or the literal `无关图片`; `objects` may be present but is not required for summary-only training.
- Summary JSON structure (BBU/RRU): top-level `统计` is required; `备注` (BBU only, non-empty) and `分组统计` (RRU only) are optional. `dataset` / `异常` keys are forbidden to keep the contract stable.
- `desc` must be a non-empty string with no control newlines/tabs. For BBU/RRU it uses comma‑separated `key=value` pairs with **no spaces** and `类别` first. `文本`/`备注` are free text: whitespace removed only; punctuation (including `,|=`) is preserved; unreadable → `可读性=不可读`. Any third-state placeholders are not emitted.
- **Groups (RRU only)**: group membership is encoded directly in `desc` as `组=<id>` (multiple groups joined with `|` if present). BBU MUST NOT include `组`. There is **no top‑level** `groups` field.
- **Station distance (RRU only)**: represented as `类别=站点距离,站点距离=<int>` and carried through to summary stats as `站点距离`.
- Assistant outputs are serialized with separators `", "` and `": "` to retain spaces in coordinate lists and JSON summaries (tokenizer stability).

#### Example: minimal (demo) record

```json
{"images":["../images/QC-20230106-0000211_16517.jpeg"],"objects":[{"bbox_2d":[48, 76, 312, 428],"desc":"类别=BBU设备,品牌=示例,可见性=部分,挡风板需求=免装"},{"poly":[360, 120, 480, 120, 480, 260, 360, 260],"poly_points":4,"desc":"类别=标签,文本=NR900-BBU"}],"summary":"{\"统计\": [{\"类别\": \"BBU设备\", \"品牌\": {\"示例\": 1}, \"可见性\": {\"部分\": 1}, \"挡风板需求\": {\"免装\": 1}}, {\"类别\": \"标签\", \"文本\": {\"NR900-BBU\": 1}}]}","width":532,"height":728}
```

#### Example: BBU record with mixed geometry + summary

```json
{
  "images": [
    "images/QC-20230222-0000297_272976.jpeg"
  ],
  "objects": [
    {
      "poly": [0, 190, 438, 230, 401, 622, 0, 614],
      "desc": "类别=BBU设备,品牌=华为,可见性=部分,挡风板需求=免装,备注=无法判断品牌"
    },
    {
      "bbox_2d": [443, 226, 491, 286],
      "desc": "类别=BBU安装螺丝,符合性=符合"
    },
    {
      "line": [453, 469, 470, 572, 501, 630, 529, 670, 611, 726],
      "desc": "类别=电线,捆扎=整齐"
    },
    {
      "poly": [481, 553, 661, 544, 644, 618, 507, 624],
      "desc": "类别=标签,文本=5GBBU接地线"
    }
  ],
  "summary": "{\"统计\": [{\"类别\": \"BBU设备\", \"品牌\": {\"华为\": 1}, \"可见性\": {\"部分\": 1}, \"挡风板需求\": {\"免装\": 1}}, {\"类别\": \"BBU安装螺丝\", \"符合性\": {\"符合\": 1}}, {\"类别\": \"电线\", \"捆扎\": {\"整齐\": 1}}, {\"类别\": \"标签\", \"文本\": {\"5GBBU接地线\": 1}}], \"备注\": [\"无法判断品牌\"]}",
  "width": 768,
  "height": 1024
}
```

#### Example: RRU record with group-encoded membership (`组=<id>`)

```json
{
  "images": [
    "images/审核通过/QC-20240424-0028974/QC-20240424-0028974_3119298.jpeg"
  ],
  "objects": [
    { "bbox_2d": [37, 269, 79, 298], "desc": "类别=站点距离,站点距离=98" },
    { "line": [122, 1173, 105, 799, 245, 703, 235, 269], "desc": "类别=接地线,标签=有标签,组=2" },
    { "line": [40, 1111, 16, 833, 165, 540, 138, 118], "desc": "类别=尾纤,标签=有标签,套管保护=有套管,组=1" },
    { "poly": [128, 840, 86, 852, 104, 954, 149, 941], "desc": "类别=标签,文本=900M-RRU2-接地,组=2" }
  ],
  "summary": "{\"统计\": [{\"类别\": \"站点距离\", \"站点距离\": {\"98\": 1}}, {\"类别\": \"接地线\", \"标签\": {\"有标签\": 1}}, {\"类别\": \"尾纤\", \"标签\": {\"有标签\": 1}, \"套管保护\": {\"有套管\": 1}}, {\"类别\": \"标签\", \"文本\": {\"900M-RRU2-接地\": 1}}], \"分组统计\": {\"1\": 1, \"2\": 2}}",
  "width": 672,
  "height": 1504
}
```

---

## 3. Training Pipeline (SFT → GRPO)

Training is a **config-first, two-stage** pipeline:
- **Stage 1 (SFT)** learns dense captioning and summary-mode formatting.
- **Stage 2 (GRPO post-training)** refines the same model checkpoint with reward functions (format stability for summary, localization-first scoring for dense).

Both stages use the same YAML-driven entrypoint (`scripts/train.sh` → `src/sft.py`): the presence of `rlhf.rlhf_type` in the YAML switches the runner from `SwiftSft` (SFT) to `SwiftRLHF` (GRPO).

Implementation anchors: `src/sft.py`, `src/config/loader.py`, `src/config/schema.py`, `src/config/grpo.py`, `src/rlhf/grpo/`, `src/datasets/`, `configs/train/`

### 3.1 Config-first training lifecycle

The training runner (`src/sft.py`) intentionally minimizes runtime CLI surface and treats YAML as the source of truth:

1. **Load + validate configuration**
   - `src/config/loader.py::ConfigLoader.load_training_config(...)` loads YAML (with inheritance), resolves prompts, and materializes typed dataclasses (`TrainingConfig`) via `src/config/schema.py`.
   - GRPO-only knobs are validated by `src/config/grpo.py::validate_grpo_config(...)` (called from both `scripts/validate_sft_config.py` and `src/sft.py`).
   - Training fails fast if the configuration requests deprecated features (e.g., packing is explicitly rejected in `src/sft.py`).

2. **Select the training pipeline (SFT vs GRPO)**
   - If `rlhf.rlhf_type` is set (e.g., `grpo`), `src/sft.py` runs a post-training pipeline (`SwiftRLHF`) and registers custom GRPO rewards (`src/rlhf/grpo/`).
   - Otherwise, `src/sft.py` runs a supervised pipeline (`SwiftSft`).

3. **Build datasets (direct or fused)**
   - Direct JSONL: `src/datasets/dense_caption.py::BaseCaptionDataset.from_jsonl(...)`
   - Fusion (target + auxiliary sources): `src/datasets/unified_fusion_dataset.py::FusionCaptionDataset` with config types in `src/datasets/fusion.py`.

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

4. **Prepare the model (LoRA/adapters) before trainer construction**
   - `src/sft.py` calls `prepare_model(...)` on the underlying ms-swift pipeline (SwiftSft for SFT, SwiftRLHF for GRPO) before creating the trainer, ensuring adapter state is correct for optimization and checkpointing.

5. **Train with padded batches**
   - Packing is removed from this runtime; the system uses padded batches only and enforces this by validation in `src/sft.py`.

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
   - assistant: JSON mapping `object_1`, `object_2`, … to `{desc, geometry...}` (`JSONLinesBuilder._build_group_entry`), optionally prefixed with `<DOMAIN=...>, <TASK=DETECTION>` + newline when `assistant_prefix_format` is enabled (required for BBU/RRU).
4. Attach top-level `objects` metadata (pixel-space points) for template-side normalization and downstream tooling.

**Summary mode workflow**
1. Load a JSONL record with a non-empty `summary`.
2. Build a single-turn conversation:
   - user: `[image, summary_prompt]`
   - assistant: the summary JSON string (single line; no coordinates), optionally prefixed with `<DOMAIN=...>, <TASK=SUMMARY>` + newline when `assistant_prefix_format` is enabled (required for BBU/RRU). Irrelevant-image samples use the literal `无关图片` as the summary line.

### 3.3 Template encoding and token flow (vision placeholders)

This project relies on the model’s native chat template (ms-swift + HF tokenizer) for vision token insertion and masking. Key behaviors:
- Templates automatically insert image placeholders; do not hand-craft `<|image_pad|>` tokens.
- Geometry stays in pixel space on disk; templates and builders handle normalization (e.g., norm1000) at encoding time.

### 3.4 Augmentation pipeline (geometry-preserving)

Augmentation is applied as a **preprocessor** that transforms both the image and its geometry in sync:
- Augmentation ops and Compose pipeline: `src/datasets/augmentation/` (`builder.py`, `base.py`, `ops.py`)
- Core geometry transforms and clipping utilities: `src/datasets/geometry.py`

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

### 3.7 GRPO post-training (reward-driven refinement)

GRPO (Group Relative Policy Optimization) is the **second stage after SFT**. Unlike Stage‑B rule‑search (which is training-free), GRPO is a **weight-updating** post-training stage (typically adapter-based) that uses reward functions to stabilize and improve outputs while keeping the same chat template and dataset contracts.

Implementation anchors: `src/rlhf/grpo/`, `src/rlhf/grpo/rollout_server_config.py`, `src/config/grpo.py`, `configs/train/grpo/`, `scripts/grpo_server_mode.sh`

**Enable and run**
- Enable GRPO by setting `rlhf.rlhf_type: grpo` in the training YAML (see `configs/train/grpo/`).
- Config examples: `configs/train/grpo/summary_2048.yaml`, `configs/train/grpo/dense_2048.yaml`, and server-mode rollouts like `configs/train/grpo/summary_1024.yaml` (uses `custom.extra.rollout_server`).
- Launch via `scripts/train.sh` (vLLM colocate rollouts) or the unified server-mode launcher: `scripts/grpo_server_mode.sh`.

**Reward interface**
- Rewards are selected by `rlhf.reward_funcs` using namespaced identifiers (e.g., `summary.format`, `dense.loc_mean_fbeta`); optional weights are provided via `rlhf.reward_weights`.
- Grouped sampling is controlled by `rlhf.num_generations` and `rlhf.generation_batch_size` (validated to be divisible in `src/config/grpo.py`).

**GRPO data contract (reward-time inputs)**
- Summary GRPO consumes `metadata.summary_ref` (ground-truth summary JSON from the dataset loader) and treats the literal `无关图片` as a special case.
- Dense GRPO consumes `assistant_payload` (ground-truth object payload emitted by the dataset builder) to score schema correctness, geometry, and attributes.

**Stability toggles**
- Optional CHORD-style supervised mixing is exposed as `custom.grpo.chord` to stabilize training when reward variance collapses.
- Optional periodic rollout dumps are exposed as `custom.grpo.dump` for lightweight debugging/telemetry.

---

## 4. Stage‑A Inference (Per-image Object Recognition & Summarization)

Stage‑A generates **per-image evidence summaries** from raw images, producing an evidence JSONL that Stage‑B consumes.

Implementation anchors: `src/stage_a/inference.py`, `src/stage_a/prompts.py`, `src/stage_a/cli.py`, `src/prompts/summary_core.py`, `src/prompts/stage_a_summary.py`, `src/prompts/summary_profiles.py`, `src/generation/`, `src/config/missions.py`

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
- System prompt: built by `src/stage_a/prompts.py::build_system_prompt(...)`, which delegates to `src/prompts/stage_a_summary.py` and the active summary profile (`src/prompts/summary_profiles.py`).
- User prompt: built by `src/stage_a/prompts.py::build_user_prompt(...)`, which includes the domain pack (BBU/RRU) when the selected profile enables it.
- The “non-site / blueprint / document” guardrail (force the literal `无关图片`) is a hard rule in the Stage‑A runtime prompt (`src/prompts/stage_a_summary.py`).
- Mission-specific prior rules (when present) are injected via `src/prompts/summary_core.py::MISSION_SPECIFIC_PRIOR_RULES` (profile-dependent), not via a separate mission-focus injection table under `src/config/missions.py`.

This separation keeps SFT from overfitting to business priors while Stage‑A runtime still gets operational guardrails.

### 4.3 Output schema and validation

Stage‑A writes streaming JSONL records (one per group) with strict coverage validation (`process_group` in `src/stage_a/inference.py`):
- `group_id` (string)
- `mission` (string)
- `label` (`pass|fail`)
- `images` (list of filenames; traceability only)
- `per_image` (object mapping `image_1`, `image_2`, … to summary strings)
- Optional postprocess may add annotation fields (e.g., `gt_fail_reason_text`) via scripts (see `scripts/stage_a.sh` and `scripts/add_gt_fail_reason_to_stage_a.py`), but Stage‑A itself emits only the fields above.

Example record:

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
    "image_1": "{\"统计\": [{\"类别\": \"BBU设备\", \"品牌\": {\"华为\": 1}, \"可见性\": {\"部分\": 1}}, {\"类别\": \"BBU安装螺丝\", \"符合性\": {\"符合\": 1}}, {\"类别\": \"标签\", \"文本\": {\"5GBBU接地线\": 1}}]}",
    "image_2": "{\"统计\": [{\"类别\": \"电线\", \"捆扎\": {\"整齐\": 1}}, {\"类别\": \"标签\", \"文本\": {\"NR900-BBU\": 1}}]}",
    "image_3": "{\"统计\": [{\"类别\": \"挡风板\", \"安装方向\": {\"方向正确\": 1}}]}"
  }
}
```

Notes:
- Stage‑A sanitizes model outputs into a stable single-line summary string: it extracts the JSON line from the typical two-line output (`<DOMAIN=...>, <TASK=SUMMARY>` + JSON), normalizes separators, and falls back to raw text only when JSON extraction fails (`sanitize_single_image_summary` in `src/stage_a/inference.py`).
- Stage‑A applies EXIF orientation fixes (`data_conversion/utils/exif_utils.py`, `apply_exif_orientation`) before encoding images.

### 4.4 Operational verification hooks

Stage‑A supports optional verification logging to debug token/grid alignment and image hashing (see `infer_one_image` / `infer_batch` in `src/stage_a/inference.py`). This is intended to catch:
- broken image decoding
- incorrect processor pixel budgets
- mismatched image token counts vs `image_grid_thw`
- Stage‑A uses the centralized `src/generation` engine; batched VLM generation defaults to **left padding** (`VlmPreprocessOptions.padding_side="left"`) for stability with variable-length prompts.

---

## 5. Stage‑B Inference (Rule‑Search, Training‑Free)

Stage‑B consumes Stage‑A evidence summaries and produces mission‑scoped group verdicts. It is **training‑free**: the model weights remain fixed, while **mission guidance** is evolved via **rule‑search** (tree‑growth / decision‑tree‑like exploration). Rule‑search is the only active Stage‑B mode in the current implementation (configs require `rule_search`, and `src/stage_b/runner.py` executes rule_search only); legacy selection/need‑review loops are not executed, though compatibility types remain.

### 5.1 Input contract: Stage‑A JSONL → `GroupTicket`

Stage‑B ingests Stage‑A JSONL and normalizes each record into a `GroupTicket`:
- Required fields: `mission` (string), `group_id` (string), `label` (`pass|fail`), `per_image` (object).
- Optional fields: `label_source` (string; default `human`), `label_timestamp` (ISO string; parsed if present).
- `per_image` must be a non‑empty mapping. Keys are normalized to `image_<index>` based on numeric suffix; ordering is normalized to ascending `image_1, image_2, ...`.
- Ticket identity is `ticket_key = "{group_id}::{label}"`, allowing the same `group_id` to appear under different labels.
- `images` is retained for traceability but is not used by Stage‑B inference.

Optional filtering is supported before rule‑search via `ticket_filter` (explicit ticket keys or a file listing ticket keys) to exclude known‑noisy labels.

### 5.2 Guidance contract (`guidance.json`)

Guidance is a mission‑scoped JSON file stored under the run directory and versioned with snapshots. The file is a mapping of `mission → section` where each section contains:
- `step` (int): monotonic guidance step counter.
- `updated_at` (ISO string): last update timestamp.
- `experiences` (mapping `key → text`):
  - `G0` is required and represents the mission focus (immutable anchor for prompting).
  - `S*` keys are scaffolds (immutable, structural invariants).
  - `G*` keys are mutable, learnable rules updated by rule‑search.
- `metadata` (optional mapping `key → object`), per‑experience provenance:
  - `updated_at`, `reflection_id`, `sources`, optional `rationale`.
  - lifecycle counters: `hit_count`, `miss_count`, `confidence` (float).

Guidance updates are written only to the run‑local `guidance.json` under `{output.root}/{mission}/{run_name}/` and are **not** merged back into the global seed automatically.

### 5.3 Prompting + strict output protocol

Each ticket is prompted with a system+user pair:
- **System prompt**: mission/domain policy scaffold (binary verdict, no third‑state language).
- **User prompt** includes:
  - mission name and focus (`G0`),
  - scaffold experiences (`S*`) and learnable rules (`G*`) rendered as numbered bullets,
  - per‑image summaries (sanitized), plus a derived `ImageN(obj=...)` statistic,
  - optional aggregation blocks for RRU missions (installation/position/cable summaries by station distance).

Stage‑B forbids “third-state / pending” language in its **own outputs** (strict two-line verdict parsing) and in candidate rule text/rationales used for guidance evolution. Stage‑A summaries are treated as evidence text and are passed through after sanitization.

**Output contract (strict two lines):**
- `Verdict: 通过|不通过` (binary only)
- `Reason: <single line>` (non‑empty, no forbidden “need‑review” phrasing)

### 5.4 Rollouts and parsing

Stage‑B generates multiple candidate responses per ticket using a decode grid (temperature/top‑p/seed). Prompts are rendered with the tokenizer chat template and “thinking” blocks are disabled for stability. Each response is parsed by a strict two‑line parser; failures are marked `format_ok = false` and excluded from rule‑search metrics.

### 5.5 Rule‑search loop (tree‑growth / decision‑tree‑like exploration)

Rule‑search grows guidance in iterations (one iteration per `runner.epochs`). Conceptually, each accepted candidate is a **branch** added to the guidance tree; metric gates act as pruning criteria.

Core loop:
1. **Pool setup**: tickets are split into train/eval pools (stratified by label). Over‑length prompts (exceeding `max_prompt_tokens`) are dropped per pool to keep comparisons consistent.
2. **Baseline rollout** on the train pool produces per‑ticket majority stats (`pass_count`, `fail_count`, `agreement`, `hard_wrong`). Hard cases are logged for analysis.
3. **Candidate proposal**:
   - A proposer LLM receives hard cases and current guidance and emits candidate operations.
   - Candidate operations include `upsert`, `update`, `merge`, `remove` with normalized rule signatures.
   - If the proposer yields too few candidates, ablation candidates are generated from low‑confidence existing rules.
4. **Candidate evaluation**:
   - Each candidate runs paired rollouts on the same train pool to compute ticket‑level stats.
   - **Gate metrics** are computed on the overlap between baseline and candidate tickets:
     - **Relative error reduction (RER)** on accuracy.
     - **Changed‑fraction**: fraction of tickets whose majority prediction flips.
     - **Bootstrap probability** that RER ≥ threshold.
   - For lifecycle ops (`update|merge|remove`), additional constraints apply: accuracy must improve, false‑release (FP) rate must improve, and FP rate increase must not exceed `max_fp_rate_increase`.
   - Optional **eval‑pool** rollouts verify that gated candidates do not regress on held‑out tickets.
5. **Selection + application**:
   - The best gated candidate (highest RER) is applied to guidance and recorded in `benchmarks.jsonl`.
   - If no candidate passes for `early_stop.patience` iterations, rule‑search stops early.

A baseline‑only mode (`jump_reflection`) runs the baseline rollout and exports diagnostics without proposing or gating rules.

### 5.6 Stage‑B artifacts and contracts

Run directory layout: `{output.root}/{mission}/{run_name}/`

Core artifacts (rule‑search runs):
- `guidance.json` + `snapshots/`: evolving guidance state with step counters and provenance.
- `rule_candidates.jsonl`: every candidate with decision and metrics. Key fields include:
  - `candidate_id`, `op`, `signature`, `target_signature(s)`, `text`, `rationale`, `source`.
  - train/eval metrics before/after (accuracy, false‑release rate, false‑block rate).
  - gate stats and final `decision` (`promoted|rejected`).
- `benchmarks.jsonl`: accepted candidates only, with before/after metrics and gate stats.
- `rule_search_hard_cases.jsonl`: hard or high‑confidence wrong tickets mined each iteration.
- `rule_search_candidate_regressions.jsonl`: tickets that regress under candidate guidance.

Baseline diagnostics (jump_reflection or baseline export):
- `baseline_metrics.json`: summary metrics (acc/fn/fp + derived np/ng).
- `baseline_ticket_stats.jsonl`: per‑ticket counts and agreement.
- `baseline_wrong_cases.jsonl`, `baseline_np_cases.jsonl`, `baseline_ng_cases.jsonl`: joined evidence for false‑block/false‑release analysis.
- `baseline_metrics_steps.jsonl`: progress snapshots during rollout.

Optional distillation:
- `distill_chatml.jsonl`: ChatML prompt/response pairs emitted under the current guidance for post‑training.

### 5.7 Distributed execution model

Stage‑B is ticket‑parallel: each rank processes a shard of tickets with a replicated model, while rank 0 aggregates stats, gates candidates, and writes artifacts. In distributed mode, device placement is forced to single‑GPU per rank to prevent unintended model‑parallel sharding.


## 6. End-to-End Workflow (Raw Data → Final Verdict)

This section ties the stages together as an operational workflow.

### 6.1 Pipeline walk-through

1. **Ingest raw annotations (optional offline conversion)**
   - The offline pipeline under `data_conversion/` produces train/val JSONL and QA reports.
   - Outputs can be validated with `scripts/validate_dense_jsonl_contract.py`.

2. **Train / update the shared Qwen3‑VL checkpoint**
   - Stage 1 SFT is executed via `src/sft.py` (wrapped by `scripts/train.sh`).
   - Stage 2 GRPO post-training is executed via the same entrypoint when `rlhf.rlhf_type=grpo` is set in the YAML (server-mode rollouts use `scripts/grpo_server_mode.sh`).
   - Training may be direct JSONL or fusion-based; augmentation is applied only when configured and preserves geometry.

3. **Stage‑A evidence generation**
   - Stage‑A reads mission folder splits (pass/fail) and emits per‑image summaries as JSONL.

4. **Stage‑B verdicts + guidance refinement**
   - Stage‑B ingests Stage‑A JSONL, performs rule‑search rollouts, and applies gated guidance updates.

5. **Audit and promote guidance**
   - Rule‑search artifacts (`rule_search_*`) and `benchmarks.jsonl` are inspected for regressions.
   - Run‑local guidance snapshots are reviewed before promotion to the shared guidance seed.

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
    B2 --> B3[SFT checkpoint (+LoRA)]
    B3 --> B4[src/sft.py (SwiftRLHF / GRPO)]
    B4 --> B5[GRPO checkpoint (merged or adapter)]
  end
  subgraph Runtime["Runtime"]
    B5 --> C1[Stage‑A: src/stage_a]
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
- Scripts in `scripts/` exist to standardize environment/logging and reduce ad-hoc divergence.

### 7.2 Determinism and reproducibility
- Offline conversion uses deterministic splitting seeds (`data_conversion/pipeline/data_splitter.py`).
- Training datasets use epoch-seeded shuffling and worker-aware seeding (`src/datasets/dense_caption.py`).
- Stage‑B seeds Python/NumPy/Torch RNGs (`src/stage_b/utils/seed.py`) and shuffles tickets per epoch with `seed + epoch` (`src/stage_b/runner.py`).
- Rollout decode seeds can be set per decode-grid entry; repeated runs remain reproducible when seeds are fixed (`src/stage_b/rollout.py`).

### 7.3 Fail-fast validation and safe fallbacks
- Data conversion validates structure, bounds, and required fields and emits explicit invalid-object/sample reports (`data_conversion/pipeline/validation_manager.py`).
- Training fails fast on unsupported runtime features (e.g., packing) and on schema violations (missing summary in summary mode, missing geometry in dense mode).
- Stage‑B enforces strict output formatting; malformed candidates are excluded from rule‑search metrics (`src/stage_b/rollout.py`).

### 7.4 Geometry-preserving augmentation (no silent geometry loss)
- Augmentation is designed to transform geometry and pixels together (`src/datasets/geometry.py`, `src/datasets/augmentation/ops.py`).
- Crop/rotate/truncation logic is built to maintain correctness of polygons and lines.
- Offline polygon ordering canonicalization prevents downstream ambiguity (`data_conversion/pipeline/coordinate_manager.py`).

### 7.5 Training-free Stage‑B refinement with governance hooks
- Stage‑B’s “learning” happens by updating a guidance file (experiences), not weights (`src/stage_b/io/guidance.py`).
- Rule-search candidates are gated by deterministic metrics and bootstrap probability before application (`src/stage_b/rule_search.py`).
- Changes are auditable via step counters and snapshot retention; hard cases and regressions are recorded for review (`rule_search_hard_cases.jsonl`, `rule_search_candidate_regressions.jsonl`).

### 7.6 Rank-aware logging (distributed-safe)
- All core pipelines use `src/utils/logger.py` (`get_logger(...)`) to keep logs readable in distributed runs and to avoid global logging side effects.

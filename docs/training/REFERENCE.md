# Training & Inference Reference

Comprehensive guide for training, inference, deployment, and advanced topics.

**Source**: `src/sft.py`, `src/stage_a/`, `src/stage_b/`, `scripts/`, `configs/`

---

## Table of Contents

- [Architecture & Implementation](#architecture--implementation)
- [Training](#training)
- [Inference](#inference)
- [Advanced Topics & FAQ](#advanced-topics--faq)
- [Additional Resources](#additional-resources)

---

## Architecture & Implementation

### Inspection Pipeline (Stage‑1 → Stage‑2)
- **Stage‑1 / Stage‑A (Basic Object Recognition)**: `src/stage_a/` emits per-image evidence/rare-object summaries used as inputs to Stage‑2. Runbook: `docs/runtime/STAGE_A_RUNTIME.md`.
- **Stage‑2 / Stage‑B (Group Ticket Verification)**: `src/stage_b/` ingests Stage‑A JSONL + labels and returns group verdicts with **prompt-only rollouts** under a strict two-line binary contract (`Verdict: 通过|不通过` + `Reason: ...`; no third-state wording). Inference is guarded by deterministic, **mission‑scoped fail‑first** signals (mission‑agnostic negative triggers + generalized patterns like `不符合要求/<issue>`, gated by the current mission `G0`). Reflection supports `add|update|delete|merge|none`. Human review is unified as `need_review_queue.jsonl` / `need_review.json` (**label-suspect only**: decision pass outputs `proposal.no_evidence_group_ids`, and bounded retry exhaustion routes with `reason_code=budget_exhausted`). Hard failures are logged to `failure_malformed.jsonl` and do not enter the review queue. No CriticEngine. Runbook: `docs/runtime/STAGE_B_RUNTIME.md` and business context in `docs/runtime/STAGE_A_STAGE_B.md`.
- **Offline preprocessing (optional)**: `data_conversion/` normalizes annotation exports into train/val/tiny JSONL and QA reports. Guide: `docs/data/DATA_PREPROCESSING_PIPELINE.md`.

### Source Code Layout

**Training Pipeline**:
- `src/sft.py` - Main entry point, `SwiftSft` integration
- `src/config/` - YAML loading, config merging, `TrainArguments` assembly
- `src/README.md` - Source-level architecture documentation

**Dataset Components**:
- `src/datasets/dense_caption.py` - `DenseCaptionDataset` (mode selection, augmentation config)
- `src/datasets/builders/jsonlines.py` - `JSONLinesBuilder` (message formatting)
- `src/datasets/preprocessors/` - Validation, augmentation preprocessing
- `src/datasets/collators.py` - Tensor preparation (padding)
- Canonical schema doc: `docs/data/DATA_JSONL_CONTRACT.md`

**Geometry & Augmentation**:
- `src/datasets/geometry.py` - Core geometry transforms (bbox, poly, line)
- `src/datasets/augmentation/base.py` - `Compose` pipeline, `ImageAugmenter` protocol
- `src/datasets/augmentation/ops.py` - All augmentation operators (Rotate, Flip, Crop, etc.)

**Utilities & Infrastructure**:
- `src/utils/logger.py` - Rank-aware logging (DDP-safe)
- `src/utils/README.md` - Utilities documentation
- `src/callbacks/save_delay_callback.py` - `SaveDelayCallback` (checkpoint throttling)

### Doc ↔ Code Cross-References
- **Stage‑1 inference**: `src/stage_a/` ↔ `docs/runtime/STAGE_A_RUNTIME.md`
- **Stage‑2 verdict loop**: `src/stage_b/` ↔ `docs/runtime/STAGE_B_RUNTIME.md`, `docs/runtime/STAGE_A_STAGE_B.md`
- **Data preprocessing**: `data_conversion/` ↔ `docs/data/DATA_PREPROCESSING_PIPELINE.md`, `docs/data/DATA_AND_DATASETS.md` (conversion section)
- **Fusion dataset**: `src/datasets/unified_fusion_dataset.py` ↔ `docs/data/UNIFIED_FUSION_DATASET.md`
- **Training & config**: `src/sft.py`, `src/config/` ↔ `docs/training/TRAINING_PLAYBOOK.md`

### Key Components Deep Dive

**ConfigLoader** (`src/config/loader.py`):
- Loads YAML and materializes frozen dataclasses (`TrainingConfig`, `CustomConfig`, `SaveDelayConfig`, `VisualKDConfig`)
- Resolves `global_max_length` → `model.max_model_len` + `template.max_length`
- Attaches typed runtime toggles to `TrainArguments` (e.g., `save_delay_config`, `visual_kd_config`)
- Fails fast with informative errors when schemas or inheritance are invalid
- Source of truth for all configuration behavior
- Over-length policy: when `template.truncation_strategy` is set to `raise`, downstream datasets catch `MaxLengthError` and drop the offending sample instead of truncating, then retry another record (see `DenseCaptionDataset`).

**DenseCaptionDataset** (`src/datasets/dense_caption.py`):
```python
# What it does:
# 1. Wraps JSONL data with preprocessors
# 2. Selects dense vs summary mode per sample (epoch-seeded)
# 3. Configures augmentation pipeline (bypass_prob, ops)
# 4. Handles both train and validation splits
```

**JSONLinesBuilder** (`src/datasets/builders/jsonlines.py`):
```python
# What it does:
# 1. Formats single-image records → single-turn messages
# 2. User message: [image, prompt]
# 3. Assistant message: {"object_1": {...}, "object_2": {...}}
# 4. Attaches top-level "objects" with pixel coords (for template normalization)
# 5. Handles dense/summary modes differently
```

**Geometry Transforms** (`src/datasets/geometry.py`):
```python
# Core function: transform_geometry(geom, M, width, height)
# - Applies affine matrix M to bbox/poly/line
# - Clips to image bounds (with polygon/line clipping algorithms)
# - Preserves degenerate geometries (fallback to clamping)
# - Handles rotation without clipping for fully-inside polygons
```

**Augmentation Pipeline** (`src/datasets/augmentation/`):
```python
# Compose pipeline:
# 1. Accumulates affine ops (rotate, flip, scale) → single matrix M
# 2. Flushes on barriers (resize, crop, expand) → applies M
# 3. Color ops deferred (applied after geometric ops)
# 4. Propagates crop metadata (kept_indices, coverages)
```

**Copy/Paste Patch Bank** (single-node):
- Enabled by copy/paste PatchOps (`small_object_zoom_paste`, `object_cluster_copy_paste`, `line_segment_copy_paste`) when `source_mode: mixed|bank` and `bank_add_prob > 0`.
- Directory is `QWEN3_VL_PATCH_BANK_DIR` (defaults to `<output_dir>/patch_bank` via `src/sft.py`).
- Shared across dataloader workers and distributed ranks **on a single node**; multi-node jobs are not supported (fail-fast).
- Sampling is deterministic when step context is available (selection derived from per-record `sample_id`, not by consuming RNG).

### Token Flow Details

**Vision Token Insertion**:
```
Image (PIL)
  → Processor (resizes to multiple of patch_size)
  → Vision Encoder (ViT) → [batch, num_patches, hidden_dim]
  → Aligner (MLP projector) → [batch, num_tokens, llm_dim]
  → Replace <|image_pad|> placeholders in LLM input
```

**Aligner Components** (in Qwen3-VL model):
- `model.visual.merger` - Main MLP projector
- `model.visual.deepstack_merger_list.{0,1,2}` - Additional projection layers
- Located in `model.visual.*` (HuggingFace model structure)

**Chat Template Mechanics**:
- Template automatically inserts `<|image_pad|>` tokens
- Placeholder count = `image_grid_thw.prod()` per image
- Do NOT manually insert placeholders in text
- Template handles vision token expansion automatically

### Logging & Callbacks

**Rank-Aware Logging** (`src/utils/logger.py`):
```python
from src.utils.logger import get_logger

logger = get_logger("my_module")
logger.info("Rank 0 only")           # Only logged on main process
logger.debug("Debug info")            # Controlled by --debug flag
logger.warning("Warning (all ranks)") # Logged on all ranks if severe
```

**SaveDelayCallback** (`src/callbacks/save_delay_callback.py`):
```python
# Prevents early checkpoints before model has learned anything
# Config: custom.save_delay_steps / custom.save_delay_epochs (via SaveDelayConfig)
# Example: save_delay_steps: 100 → no saves until step 100
```

### Health Check Implementation

**Validation Points** (enforced in code):

1. **Image Placeholder Count** (`src/datasets/builders/jsonlines.py`):
   - User message image count matches `len(images)`
   - Template inserts correct number of `<|image_pad|>` tokens

2. **Grid Alignment** (`src/datasets/collators.py`):
   - `image_grid_thw` shape matches `pixel_values` dimensions
   - Each image has valid T×H×W grid

3. **Label Masking** (template encoding):
   - Image tokens in `input_ids` have `labels = -100`
   - Assistant tokens have `labels = input_ids[pos]`
   - User tokens have `labels = -100`

4. **Geometry Normalization** (`JSONLinesBuilder`):
   - Top-level `objects` kept in pixel space
   - Template normalizes `bbox_2d` to norm1000 during encoding
   - Assistant text uses `emit_norm` setting

### Extension Points

**Add New Preprocessor**:
1. Create `src/datasets/preprocessors/my_preprocessor.py`
2. Implement `BasePreprocessor` protocol
3. Register in `DenseCaptionDataset.__init__`

**Add New Augmentation Op**:
1. Add class to `src/datasets/augmentation/ops.py`
2. Implement `ImageAugmenter` protocol (`affine()` or `apply()`)
3. Use in YAML: `- name: my_op`

**Add New Builder**:
1. Create `src/datasets/builders/my_builder.py`
2. Implement builder protocol (`build_messages()`)
3. Configure in dataset initialization

### Hard-Sample Mining
- Deprecated as of 2025-11-27. Any config containing `custom.hard_sample_mining` will fail validation with guidance to remove the block and train with standard padded settings.

### Critical Implementation Details

**Adapter Preparation** (`src/sft.py`):
```python
# MUST be called before creating trainer
sft = SwiftSft(args)
sft.prepare_model()  # Configures LoRA, freezes, modules_to_save
trainer = sft.create_trainer()  # Now trainer has correct config
```

**Packing**: Removed. Training now uses padded batches only; any config enabling `training.packing` or packing knobs fails fast. Legacy implementation is archived under `archive/packing/`.

**Dataset-specific metrics (fusion)**: Padded batches carry `dataset_labels` from fusion metadata. The trainer logs per-dataset `*_loss` / `*_token_acc` during training for all datasets and skips source domains during eval when `dataset_domains` marks them (e.g., targets `bbu/rru` vs sources `lvis/lang_chat`). No extra config is required beyond fusion dataset wiring.

**Freeze Logic** (`src/sft.py` + `SwiftSft`):
```python
# freeze_llm: true → model.model.layers[*].requires_grad = False
# freeze_vit: true → model.visual.*.requires_grad = False
# freeze_aligner: false → model.visual.merger*.requires_grad = True
```

---

## Training

Core SFT/LoRA recipes, KL anchoring overlays, augmentation telemetry, and troubleshooting now live in [TRAINING_PLAYBOOK.md](TRAINING_PLAYBOOK.md). Use that document for:
- YAML scaffolding for single- and multi-stage training
- LoRA/freezing setups and SaveDelay guidance (packing removed)
- Telemetry expectations plus health-check checklists
- Advanced topics (LR schedulers, DeepSpeed configs, augmentation FAQs, common issues, aligner tuning)

**Token-type telemetry (optional)**  
- Config: `custom.token_type_metrics.enabled` (default `false`), `include` (default `['target','lvis']`), `exclude` (default `['coig_lang_chat']`).  
- Behavior: collator reconstructs assistant JSON, tokenizes with the active template, aligns token types (1=desc, 2=coord numbers, 3=format), and pads/truncates to supervised positions; rows outside `include` get IGNORE.  
- Metrics: per dataset label, logs `{label}_token_acc` (all supervised tokens, naturally weighted), type-sliced accuracies `{label}_{desc|coord|format}_token_acc`.  
- Validation: smoke run on 2025-12-04 with `configs/smoke/group_metrics.yaml` (4B checkpoint, tiny fusion `configs/fusion/bbu_rru_lvis_coig_tiny.yaml`, `logging_steps=1`, `eval_steps=1`, `save_strategy=no`, `max_steps=20`) produced the expected token-type metrics; see `output/smoke/group_metrics/v0-20251204-062817/smoke_group_metrics_4b/logging.jsonl`.

Keep configs under `configs/` in sync with the playbook when making behavioral changes.

### Summary Prompt Profiles (Training vs Inference)
- **Training default**: `summary_train_min` (format + task criterion only, evidence-only/no-hallucination constraints).
- **Config knobs** (summary mode only):
  - `prompts.profile`: `summary_train_min` | `summary_runtime`
  - `prompts.domain`: `bbu` | `rru` (required only when using runtime profile)
  - `prompts.system` / `prompts.user` remain authoritative overrides and bypass profile composition.

## Inference

Runtime/deployment instructions for Stage-A summaries and the Stage-B verdict loop live in [STAGE_A_RUNTIME.md](../runtime/STAGE_A_RUNTIME.md) and [STAGE_B_RUNTIME.md](../runtime/STAGE_B_RUNTIME.md):

**Top urgency (Stage-A batching)**: Stage-A uses **batched generation** (not packing). For Qwen3-VL **decoder-only** models with variable-length VL prompts, callers MUST use **left padding** (`tokenizer.padding_side="left"`). Right padding can cause unstable/garbage prefixes (e.g., random English tokens) in batched inference; see the “Critical: Batched Generation Requires Left Padding” section in `docs/runtime/STAGE_A_RUNTIME.md`.
- Adapter vs merged checkpoints, export commands, and decoding tips
- Dense captioning usage examples
- Stage-A CLI guardrails and output schemas
- Stage-B sampler/selection/reflection flow (prompt-only, no CriticEngine); rollout 提示=guidance+Stage-A 摘要（不含 GT；S* 为结构不变量，G0+ 可学习），推理输出严格两行二分类（`Verdict: 通过|不通过` + `Reason: ...`）且禁止任何第三状态词面；若确定性护栏覆盖模型采样 verdict，则必须重写 `Reason` 以与最终 `Verdict` 一致。护栏包含 **mission‑scoped fail‑first**（仅当负项与当前 mission 的 `G0` 相关时触发，含 pattern-first `不符合要求/<issue>`）；Stage‑A summary 中的 `需复核,备注:` 视为软信号，需读取备注综合判断，不能把 `需复核` 硬等价为 fail。人工复核入口统一为 `need_review_queue.jsonl`：decision pass 在看到 `gt_label` 后仍明确标记“没有证据/想不明白错哪”（`proposal.no_evidence_group_ids` 命中该 ticket_key）时，才会把该工单写入 `need_review_queue.jsonl`，并在 run 结束生成 `need_review.json` 汇总。ops pass 输出严格 JSON ops + hypotheses，系统执行 learnability closure `L == E ∪ H` 并做 bounded retry；两行协议解析失败/无可用候选/selection 报错等硬故障仅写入 `failure_malformed.jsonl` 与日志。重跑同一 run_name 重建 per-run artifacts，指导沿用上次快照。
- Stage-B 生成 `metrics.jsonl`（step-wise `logging_steps` 窗口 + 每个 epoch 的汇总行），包含/排除人工复核两套 acc/fn/fp/n 计数，与 trajectories/selections 并列用于快速健康检查。
- Stage-B 可选生成 `group_report_delta.jsonl`（轻量、step-wise，仅包含最近一个 `logging_steps` 窗口内处理过的 group 的“增量快照”，便于监控 verdict/reason 变化）；仍会在 run 结束生成完整 `group_report.jsonl`（或通过 `scripts/stage_b_group_report.py` 离线重建）。
- 生产约束：Stage‑A 与 Stage‑B 在最终环境中共用同一个 Qwen3‑VL 模型（同一组权重 / LoRA 组合），通过不同 prompt 和 config 切换任务；训练 summary‑mode 或添加新 LoRA 时，需要显式评估对 Stage‑B rollout/verdict 行为的影响。

## Advanced Topics & FAQ

Operational FAQs (LR schedulers, DeepSpeed presets, augmentation pipelines, template mechanics, and troubleshooting) are consolidated into [TRAINING_PLAYBOOK.md](TRAINING_PLAYBOOK.md). Refer to that doc whenever you tweak trainers, infrastructure knobs, or template logic.

## Additional Resources

- **Training**: [TRAINING_PLAYBOOK.md](TRAINING_PLAYBOOK.md)
- **Stage-A & Stage-B runtime**: [STAGE_B_RUNTIME.md](../runtime/STAGE_B_RUNTIME.md)
- **Data preprocessing & contract**: [DATA_PREPROCESSING_PIPELINE.md](DATA_PREPROCESSING_PIPELINE.md), [DATA_JSONL_CONTRACT.md](DATA_JSONL_CONTRACT.md)
- **Data formats & augmentation**: [DATA_AND_DATASETS.md](DATA_AND_DATASETS.md), [DATA_AUGMENTATION.md](DATA_AUGMENTATION.md)
- **Archived docs**: `docs/archive/` (historical references, detailed technical guides)

---

**Last Updated**: 2025-12-04 (Token-type telemetry validated via smoke run)

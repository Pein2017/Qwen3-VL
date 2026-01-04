# Training & Inference Reference

Status: Active
Scope: Detailed reference for training, inference, configs, and troubleshooting.
Owners: Training + Runtime
Last updated: 2026-01-02
Related: [TRAINING_PLAYBOOK.md](TRAINING_PLAYBOOK.md), [runtime/STAGE_A_STAGE_B.md](../runtime/STAGE_A_STAGE_B.md), [reference/PROMPTS_REFERENCE.md](../reference/PROMPTS_REFERENCE.md)

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
- **Stage‑1 / Stage‑A (Basic Object Recognition)**: `src/stage_a/` emits per-image evidence/rare-object summaries used as inputs to Stage‑2. Runbook: [runtime/STAGE_A_RUNTIME.md](../runtime/STAGE_A_RUNTIME.md).
- **Stage‑2 / Stage‑B (Group Ticket Verification)**: `src/stage_b/` ingests Stage‑A JSONL + labels and runs **prompt-only rollouts** under a strict two-line binary contract (`Verdict: 通过|不通过` + `Reason: ...`; no third-state wording). Stage‑B only supports **rule_search** (reflection as rule proposer + large-scale rollout metric gating; produces `rule_candidates.jsonl`/`benchmarks.jsonl` + audit trails). No CriticEngine. Runbook: [runtime/STAGE_B_RUNTIME.md](../runtime/STAGE_B_RUNTIME.md) and business context in [runtime/STAGE_A_STAGE_B.md](../runtime/STAGE_A_STAGE_B.md).
- **Offline preprocessing (optional)**: `data_conversion/` normalizes annotation exports into train/val/tiny JSONL and QA reports. Guide: [data/DATA_PREPROCESSING_PIPELINE.md](../data/DATA_PREPROCESSING_PIPELINE.md).

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
- Canonical schema doc: [data/DATA_JSONL_CONTRACT.md](../data/DATA_JSONL_CONTRACT.md)

**Geometry & Augmentation**:
- `src/datasets/geometry.py` - Core geometry transforms (bbox, poly, line)
- `src/datasets/augmentation/base.py` - `Compose` pipeline, `ImageAugmenter` protocol
- `src/datasets/augmentation/ops.py` - All augmentation operators (Rotate, Flip, Crop, etc.)

**Utilities & Infrastructure**:
- `src/utils/logger.py` - Rank-aware logging (DDP-safe)
- `src/utils/README.md` - Utilities documentation
- `src/callbacks/save_delay_callback.py` - `SaveDelayCallback` (checkpoint throttling)

### Doc ↔ Code Cross-References
- **Stage‑1 inference**: `src/stage_a/` ↔ [runtime/STAGE_A_RUNTIME.md](../runtime/STAGE_A_RUNTIME.md)
- **Stage‑2 verdict loop**: `src/stage_b/` ↔ [runtime/STAGE_B_RUNTIME.md](../runtime/STAGE_B_RUNTIME.md), [runtime/STAGE_A_STAGE_B.md](../runtime/STAGE_A_STAGE_B.md)
- **Data preprocessing**: `data_conversion/` ↔ [data/DATA_PREPROCESSING_PIPELINE.md](../data/DATA_PREPROCESSING_PIPELINE.md), [data/DATA_AND_DATASETS.md](../data/DATA_AND_DATASETS.md) (conversion section)
- **Fusion dataset**: `src/datasets/unified_fusion_dataset.py` ↔ [data/UNIFIED_FUSION_DATASET.md](../data/UNIFIED_FUSION_DATASET.md)
- **Training & config**: `src/sft.py`, `src/config/` ↔ [training/TRAINING_PLAYBOOK.md](TRAINING_PLAYBOOK.md)

### Key Components Deep Dive

**ConfigLoader** (`src/config/loader.py`):
- Loads YAML and materializes frozen dataclasses (`TrainingConfig`, `CustomConfig`, `SaveDelayConfig`, `VisualKDConfig`)
- Resolves `global_max_length` → `model.max_model_len` + `template.max_length` and forces `template.truncation_strategy: raise`
- Attaches typed runtime toggles to `TrainArguments` (e.g., `save_delay_config`, `visual_kd_config`)
- Fails fast with informative errors when schemas or inheritance are invalid
- Source of truth for all configuration behavior
- Over-length policy: when `template.truncation_strategy` is `raise`, downstream datasets surface `MaxLengthError` as a hard failure so training stops rather than truncating or skipping samples (see `DenseCaptionDataset`).

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

**Template + Record Compatibility**:
- `scripts/train.sh` launches SFT with a **single base template** (from config), even when `custom.fusion_config` mixes datasets. Per-dataset `template` strings select **prompt presets** only.
- Supported record styles in fusion:
  - **ChatML text-only** (`messages` only; no `images/objects/summary`) for chat corpora.
  - **Generic detection** (LVIS/COCO, etc.) with simple `desc` tokens and no summary.
  - **Target domain** (BBU/RRU) with `key=value` `desc` and JSON-string `summary`.
- When mixing chat + dense-caption sources, choose a base template that can encode both formats (ChatML family recommended).

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
- Config: `custom.token_type_metrics.enabled` (default `true` in `configs/fusion_train/sft_base.yaml`), `include` (default `['bbu','rru','lvis']`), `exclude` (default `['lang_chat']`).  
- Behavior: collator reconstructs assistant JSON, tokenizes with the active template, aligns token types (1=desc, 2=coord numbers, 3=format), and pads/truncates to supervised positions; rows outside `include` get IGNORE.  
- Metrics: per dataset label, logs `{label}_token_acc` (all supervised tokens, naturally weighted), type-sliced accuracies `{label}_{desc|coord|format}_token_acc`.  
- Validation: smoke run on 2025-12-04 with `configs/smoke/group_metrics.yaml` (4B checkpoint, fusion `configs/dataset_mix/bbu_rru_dense_new_schema_1024.yaml`, `logging_steps=1`, `eval_steps=1`, `save_strategy=no`, `max_steps=20`) produced the expected token-type metrics; see `output/smoke/group_metrics/v0-20251204-062817/smoke_group_metrics_4b/logging.jsonl`.

Keep configs under `configs/` in sync with the playbook when making behavioral changes.

### Summary Prompt Profiles (Training vs Inference)
- **Training default**: `summary_runtime`.
- **Config knobs** (summary mode only):
  - `prompts.profile`: `summary_runtime`
  - `prompts.domain`: `bbu` | `rru` (required for runtime profile)
  - `prompts.system` / `prompts.user` remain authoritative overrides and bypass profile composition.
  - `custom.assistant_prefix_format`: required for BBU/RRU targets to prepend `<DOMAIN=...>, <TASK=...>` + newline before assistant payloads (dense + summary). Source datasets remain unchanged.
- **Stage-A runtime composition**: system prompt = summary task base + 全局“非现场/图纸”规则；user prompt = summary instruction + BBU/RRU 场景提示块 + 可选任务重点。

### Summary GRPO Post-Training (Format Stabilization)
- **Config example**: `configs/grpo/summary_grpo_base.yaml` (uses `configs/dataset_mix/bbu_rru_summary_grpo_new_schema_1024.yaml`; edit checkpoint + epochs/LRs).
- **Base template**: `configs/grpo/summary_grpo_base.yaml` (extends `configs/fusion_train/sft_base.yaml`).
  - **Required knobs**:
    - `rlhf.rlhf_type=grpo`
    - `rlhf.reward_funcs=[summary.format, summary.header, summary.strict, summary.parse, summary.no_dup_keys, summary.dataset, summary.category_recall, summary.content_structured_tversky, summary.text_bbu, summary.notes_bbu, summary.group_stats_presence]`
    - `rlhf.num_generations` (must divide `rlhf.generation_batch_size`)
    - `rlhf.max_completion_length=2048`
    - `training.effective_batch_size` (backward global batch), `rlhf.generation_batch_size` (rollout global trajectories)
    - `prompts.profile=summary_runtime`, `custom.assistant_prefix_format`, `custom.fusion_config`
    - Tune `rlhf.temperature` based on contract stability vs exploration.
- **Metadata contract**: summary-mode rows attach `metadata.summary_ref` (ground-truth JSON) and `metadata._fusion_domain_token` (BBU/RRU) for reward functions; irrelevant rows keep `_fusion_source=irrelevant_summary` and suppress assistant prefixes so labels remain single-line `无关图片`.
- **Dry-run recipe**: clone the example config, set `training.max_steps: 2`, `training.eval_strategy: "no"`, `training.save_strategy: "no"`, `custom.train_sample_limit: 2`, `custom.val_sample_limit: 2`, then launch via `scripts/train.sh config=<new-config.yaml> gpus=0 debug=true`.
  - **Success criteria**: job starts, reward metrics appear (format/header/strict/parse + dup-key penalty + structured content rewards), and 1–2 steps complete without dataset or format exceptions.

## Inference

Runtime/deployment instructions for Stage-A summaries and the Stage-B verdict loop live in [STAGE_A_RUNTIME.md](../runtime/STAGE_A_RUNTIME.md) and [STAGE_B_RUNTIME.md](../runtime/STAGE_B_RUNTIME.md):

**Top urgency (Stage-A batching)**: Stage-A uses **batched generation** (not packing). For Qwen3-VL **decoder-only** models with variable-length VL prompts, callers MUST use **left padding** (`tokenizer.padding_side="left"`). Right padding can cause unstable/garbage prefixes (e.g., random English tokens) in batched inference; see the “Critical: Batched Generation Requires Left Padding” section in [runtime/STAGE_A_RUNTIME.md](../runtime/STAGE_A_RUNTIME.md).
- Adapter vs merged checkpoints, export commands, and decoding tips
- Dense captioning usage examples
- Stage-A CLI guardrails and output schemas
- Stage-B 仅支持 `rule_search`（baseline rollout → proposer 产出 1–N 条候选操作 → **train pool** A/B gate → 通过者更新 guidance；eval pool 指标仅用于审计）。system prompt 固定为两行判决契约 + 规则/软硬信号；user prompt 仅包含 guidance（S*/G0/G*）+ Stage-A 摘要（不含 GT）。领域提示已上移到 Stage‑A user prompt。**规则默认 AND 关系**（未显式写“或/例外”的规则必须全部满足；缺证据即不通过），推理输出严格两行二分类（`Verdict: 通过|不通过` + `Reason: ...`）且禁止任何第三状态词面。rollout/proposer/reflection 超长提示不截断，直接 drop；全局配置建议：rollout `max_prompt_tokens=4096`，proposer/reflection `max_token_length=12000`。`rule_search` 会写入 `rule_candidates.jsonl`、`benchmarks.jsonl`、`rule_search_hard_cases.jsonl`、`rule_search_candidate_regressions.jsonl`。重跑同一 run_name 重建 per-run artifacts，指导沿用上次快照。另：可用 `jump_reflection=true`（或 `--jump-reflection` / YAML `jump_reflection: true`）跳过 proposer/reflection，仅跑 baseline rollout 并导出 `baseline_metrics.json` / `baseline_ticket_stats.jsonl` 以便人工分析后再手动修改 `initial_guidance.json`。
- RRU 任务口径：RRU安装/位置/线缆相互独立；仅依据本任务 G0/S* 要素。RRU位置/线缆的标签要求为“可读文本”，不强制包含特定字样。
- Stage-B 可选导出 `distill_chatml.jsonl` 供后训练使用（低温采样、随机抽样 `distill_size`）。
- 生产约束：Stage‑A 与 Stage‑B 在最终环境中共用同一个 Qwen3‑VL 模型（同一组权重 / LoRA 组合），通过不同 prompt 和 config 切换任务；训练 summary‑mode 或添加新 LoRA 时，需要显式评估对 Stage‑B rollout/verdict 行为的影响。

## Advanced Topics & FAQ

Operational FAQs (LR schedulers, DeepSpeed presets, augmentation pipelines, template mechanics, and troubleshooting) are consolidated into [TRAINING_PLAYBOOK.md](TRAINING_PLAYBOOK.md). Refer to that doc whenever you tweak trainers, infrastructure knobs, or template logic.

## Additional Resources

- **Training**: [TRAINING_PLAYBOOK.md](TRAINING_PLAYBOOK.md)
- **Stage-A & Stage-B runtime**: [STAGE_B_RUNTIME.md](../runtime/STAGE_B_RUNTIME.md)
- **Data preprocessing & contract**: [DATA_PREPROCESSING_PIPELINE.md](../data/DATA_PREPROCESSING_PIPELINE.md), [DATA_JSONL_CONTRACT.md](../data/DATA_JSONL_CONTRACT.md)
- **Data formats & augmentation**: [DATA_AND_DATASETS.md](../data/DATA_AND_DATASETS.md), [DATA_AUGMENTATION.md](../data/DATA_AUGMENTATION.md)

---

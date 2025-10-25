# Group-Level QC with GRPO (BBU)

Status: Archived — Feature-oriented; not part of core SFT/inference docs

## Overview: Two-Stage Pipeline

**Stage-A (Image-Level Text Summary)**
- Processes individual images to generate Chinese single-line summaries
- Input: Images → Output: Per-image text summaries grouped by scene

**Stage-B (Group-Level Judgment Inference)**
- Reads Stage-A summaries (text-only) and infers a two-line verdict
- Input: Stage-A text summaries → Output: Two lines (通过/不通过 + reasoning)

---

## Stage-A: Image-Level Text Summary Generation

**Purpose**: Generate a Chinese single-line summary for each image, organized by group.

**Input**: Mission-based directory with labeled groups
- Structure: `<root>/<mission>/{审核通过,审核不通过}/<group_id>/*.{jpg,jpeg,png}`
- Discovery: Natural sort of images within each group

**Processing**:
- Batch inference (one image at a time per forward pass, batched across images)
- Mission-dependent Chinese prompts (see `src/stage_a/prompts.py`)
- Generates single-line summary per image

**Grouping**:
- Derive `group_id` from filename prefix `QC-...-YYYYMMDD-NNNNNN` if present
- Fallback: use subdirectory name as group_id
- Map images to 图片_{i} keys (i = 1, 2, 3, ...)

**Validation**:
- 图片_{i} must align exactly to sorted order
- Empty summaries → raise error (no partial writes)
- Mismatch in count → abort group

**Output JSONL** (one file per mission; one line per group):
```json
{
  "group_id": "QC-TEMP-20250118-0015956",
  "mission": "挡风板安装检查",
  "label": "pass",
  "images": ["/abs/path/img-001.jpeg", "/abs/path/img-002.jpeg"],
  "per_image": {
    "图片_1": "BBU设备/华为/显示完整/安装牢固",
    "图片_2": "螺丝、光纤插头/BBU安装螺丝/符合要求"
  },
  "raw_texts": ["...", "..."],
  "clean_texts": ["...", "..."],
  "timestamp": "2025-10-24T12:34:56.000000Z"
}
```

**Key Characteristics**:
- ✅ Text-only summaries (no bounding boxes or coordinates)
- ✅ One summary per image
- ✅ Grouped by scene (group_id)
- ✅ Streaming output (resumable, monitorable)

---

## Stage-B: Group-Level Judgment Inference (GRPO Training)

**Purpose**: Given Stage-A text summaries, infer a group-level Pass/Fail verdict with reasoning.

**Input**: GRPO dataset built from Stage-A outputs
- Source: Stage-A JSONL files
- Processing: Text-only (no images re-fed)

**Dataset Schema** (one line per group):
```json
{
  "group_id": "QC-TEMP-20250118-0015956",
  "task_type": "挡风板安装检查",
  "group_label": "通过",
  "stage_a_summaries": {
    "图片_1": "BBU设备/华为/显示完整/安装牢固",
    "图片_2": "螺丝、光纤插头/BBU安装螺丝/符合要求"
  },
  "messages": [
    {"role": "system", "content": "你是BBU质检助手。请根据图片摘要判断是否通过检查，严格输出两行..."},
    {"role": "user", "content": "任务：挡风板安装检查\n图片_1: BBU设备/华为/...\n图片_2: 螺丝、光纤插头/..."}
  ]
}
```

**Model Inference Task**:
- Input: System prompt + user message with embedded Stage-A summaries (plain text)
- Output: **Exactly two lines**
  - **第一行**: 严格为 "通过" 或 "不通过" (verdict only, no extra text)
  - **第二行**: 理由: <中文自然语言reasoning, 可引用 图片_i, 不复述原文>

**Example Output**:
```
通过
理由: 图片_1和图片_2均显示BBU设备安装牢固，螺丝符合要求，无需安装挡风板。
```

**Key Characteristics**:
- ✅ Text-only input (Stage-A summaries as plain text)
- ✅ No images fed during Stage-B inference/training
- ✅ Two-line output format strictly enforced
- ✅ GRPO training with binary label reward + format reward

---

## Rewards (Stage-B GRPO Training, v1)

**Label Reward** (标签奖励):
- Extract verdict from first line of model output
- Match against `group_label` field (通过/不通过)
- Score: 1.0 if exact match, 0.0 otherwise
- If format invalid (can't parse first line) → 0.0

**Format Reward** (格式奖励):
- Validate exactly two lines (split by `\n`)
- First line must be exactly "通过" or "不通过" (no extra tokens)
- Second line must start with "理由:" and have content
- Score: 1.0 if valid, 0.0 otherwise

**Consistency Reward** (一致性奖励, deferred to v2):
- Check that reasoning in line 2 aligns with Stage-A summaries
- Requires passing `stage_a_summaries` to reward function
- Not implemented in initial version

---

## ms-swift GRPO Integration (Stage-B Training)

**Model Configuration**:
- **LoRA target**: Last-K LLM transformer blocks only (default K=4)
- **Frozen modules**: Vision encoder (ViT) + Aligner (MLP projector)
- **Rationale**: Stage-B is text-only reasoning; no image inputs → no vision tuning needed

**Training Parameters**:
- `num_generations >= 2` (required for GRPO on-policy sampling)
- `max_new_tokens`: 128 (short, two-line outputs)
- Reward weights: `label:format = 1.0:0.2` (emphasize correctness over format)

**Launcher Outline** (`scripts/run_grpo.py`):
```python
from swift.trainers.rlhf_trainer.grpo_trainer import GRPOTrainer
from src.stage_b.rewards import reward_label_cn, reward_format_cn

def run_grpo(cfg):
    # 1) Load tokenizer/model (base checkpoint + optional Stage-A adapters)
    # 2) Freeze vision encoder and aligner modules
    # 3) Apply LoRA to last-K LLM transformer blocks
    # 4) Load Stage-B dataset from JSONL (text-only messages)
    # 5) Register reward functions: [reward_label_cn, reward_format_cn]
    # 6) Configure reward weights: [1.0, 0.2]
    # 7) trainer = GRPOTrainer(model, tokenizer, train_dataset, cfg)
    # 8) trainer.train(max_steps=...)
    # 9) Save LoRA adapter checkpoint
    pass
```

---

## Prompting Strategy

**Stage-A (Image-Level Summary)**:
- System: 简洁中文提示，说明任务类型和输出要求（单行摘要）
- User: "请描述这张图片中的BBU设备和检查要点"
- Output: 单句中文摘要（避免计数、列表、模板化句式）
- Example: "BBU设备/华为/显示完整/安装牢固"

**Stage-B (Group-Level Judgment)**:
- System: 格式要求（严格两行输出）+ 任务上下文（检查标准）
- User: 任务类型 + 图片摘要汇总（图片_1: ..., 图片_2: ..., ...）
- Output: 两行（通过/不通过 + 理由）
- Example:
  ```
  通过
  理由: 图片_1至图片_3均显示BBU设备安装符合要求，无异常情况。
  ```

**Language Policy**:
- 全中文提示词与输出
- 允许英文专有名词（BBU, ODF, RRU等通信设备术语）
- 避免中英混杂（除专业术语外）

---

## Data Flow Summary

```
┌─────────────────────────────────────┐
│  Mission Images (labeled groups)    │
└──────────────┬──────────────────────┘
               │ Stage-A Inference
               │ (per-image summaries)
               ↓
┌─────────────────────────────────────┐
│  Stage-A JSONL Output                │
│  {group_id, mission, label,          │
│   per_image{图片_i: text}}           │
└──────────────┬──────────────────────┘
               │ GRPO Training
               │ (loads Stage-A directly)
               ↓
┌─────────────────────────────────────┐
│  GRPO Rollout (Dynamic Generation)   │
│  1. Build prompts from Stage-A       │
│  2. Generate verdicts on-the-fly     │
│  3. Compute rewards vs GT labels     │
│  4. Update LLM LoRA parameters       │
└──────────────┬──────────────────────┘
               │
               ↓
┌─────────────────────────────────────┐
│  Trained LoRA Checkpoint             │
│  (group-level judgment capability)   │
└─────────────────────────────────────┘
```

**Key Point**: Stage-B outputs are **NOT pre-built**. The model generates them dynamically during GRPO rollout.

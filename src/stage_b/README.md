# Stage-B: Group-Level Judgment Inference & GRPO Training

Stage-B processes Stage-A text summaries (no images) to infer a group-level Pass/Fail verdict with reasoning.

## Overview

**Input**: Stage-A per-image summaries (text-only)  
**Output**: Two-line verdict (通过/不通过 + reasoning) generated **dynamically during GRPO rollout**  
**Training**: GRPO with binary label reward + format reward

## Key Insight: Dynamic Generation

**Stage-B responses are NOT pre-built**. Instead:
1. Load Stage-A JSONL (summaries + GT labels)
2. Build prompts dynamically during GRPO training
3. Model generates verdicts on-the-fly during rollout
4. Rewards computed based on GT labels from Stage-A

## Components

### 1. Dataset Loader (`src/stage_b/dataset.py`)

Loads Stage-A JSONL outputs and prepares them for GRPO training.

**Key function**: `load_stage_a_for_grpo()`

```python
from src.stage_b.dataset import load_stage_a_for_grpo

# Load single or multiple Stage-A JSONL files
dataset = load_stage_a_for_grpo([
    "output_post/stage_a/挡风板安装检查_stage_a.jsonl",
    "output_post/stage_a/BBU安装方式检查（正装）_stage_a.jsonl",
])
```

**Dataset Schema** (prepared for GRPO):
```python
{
  "group_id": "QC-20250118-0001",
  "task_type": "挡风板安装检查",
  "group_label": "通过",  # GT label for reward computation
  "stage_a_summaries": {
    "图片_1": "BBU设备/华为/显示完整/安装牢固",
    "图片_2": "螺丝/符合要求"
  },
  "messages": [  # Prompt only (no assistant response pre-built)
    {"role": "system", "content": "你是通信机房质检助手..."},
    {"role": "user", "content": "任务: 挡风板安装检查\n..."}
  ]
}
```

### 2. Prompts (`src/stage_b/prompts.py`)

Mission-specific system and user message templates.

**Supported Missions**:
- `BBU安装方式检查（正装）`
- `BBU接地线检查`
- `BBU线缆布放`
- `挡风板安装检查`

**System Prompt**: Generic format rules + conflict handling  
**User Prompt**: Task-specific focus + Stage-A summaries embedded as text

### 3. Reward Functions (`src/stage_b/rewards.py`)

**Label Reward** (`label_reward`):
- Extracts verdict from first line
- Compares with `group_label` field
- Returns 1.0 for exact match, 0.0 otherwise

**Format Reward** (`format_reward`):
- Validates exactly two lines
- First line: exactly "通过" or "不通过"
- Second line: starts with "理由:" + content
- Returns 1.0 if valid, 0.0 otherwise

**Consistency Reward** (placeholder for v2):
- Will check reasoning aligns with Stage-A summaries
- Not implemented in v1

**Testing**:
```bash
cd /data/Qwen3-VL
python -m src.stage_b.test_rewards
```

### 4. GRPO Launcher (`scripts/run_grpo.py`)

Python launcher for GRPO training with ms-swift integration.

**Key Features**:
- Loads Stage-A JSONL directly (no pre-built Stage-B outputs)
- Builds prompts dynamically during rollout
- LLM-only LoRA on last-K transformer blocks (default K=4)
- Frozen vision encoder + aligner
- Custom reward functions (label + format)

**Usage**:
```bash
# Single mission
python scripts/run_grpo.py \
  --model output/summary_merged/10-24 \
  --train_dataset output_post/stage_a/挡风板安装检查_stage_a.jsonl \
  --output_dir output_post/grpo/run_1 \
  --lora_last_k 4 \
  --num_generations 4 \
  --device cuda:0

# Multiple missions (combined)
python scripts/run_grpo.py \
  --model output/summary_merged/10-24 \
  --train_dataset output_post/stage_a/*.jsonl \
  --output_dir output_post/grpo/run_all \
  --lora_last_k 4 \
  --num_generations 4 \
  --device cuda:0 \
  --dry_run  # Validate setup without training
```

**Configuration**:
- `--lora_last_k`: Number of final LLM blocks to tune (default: 4)
- `--num_generations`: On-policy samples per batch (≥ 2, default: 4)
- Reward weights: `[1.0, 0.2]` (label:format)

## Model Output Contract

**Exactly two lines**:
1. **Line 1**: `通过` or `不通过` (no extra tokens)
2. **Line 2**: `理由: <中文自然语言reasoning>`

**Example**:
```
通过
理由: 图片_1和图片_2均显示BBU设备安装牢固，螺丝符合要求，无需安装挡风板。
```

## Integration with Stage-A

```
Stage-A output (per mission):
  output_post/stage_a/挡风板安装检查_stage_a.jsonl
  (contains: summaries + GT labels + mission info)
    ↓
GRPO training (loads Stage-A directly):
  python scripts/run_grpo.py --train_dataset output_post/stage_a/*.jsonl
    ↓
  [During GRPO rollout]
  1. Build prompts dynamically (system + user with summaries)
  2. Model generates Stage-B responses (two-line verdicts)
  3. Compute rewards (label + format)
  4. Update LoRA parameters
    ↓
Trained LoRA checkpoint:
  output_post/grpo/run_1/checkpoint-XXX
```

**Key difference from traditional supervised learning**:
- No pre-built Stage-B outputs needed
- Model learns through online generation + reward feedback
- GT labels from Stage-A used only for reward computation

## ms-swift GRPO Integration Notes

The current `run_grpo.py` is a **dry-run launcher** that validates configuration and shows the intended structure. To complete integration:

1. **Install ms-swift with GRPO support**:
   ```bash
   cd /data/ms-swift
   pip install -e .
   ```

2. **Use `GRPOTrainer` from ms-swift**:
   ```python
   from swift.trainers.rlhf_trainer.grpo_trainer import GRPOTrainer
   from swift.trainers.rlhf_arguments import GRPOConfig
   ```

3. **Pass reward function**:
   - Via `external_plugins` parameter
   - Or direct callback in trainer initialization

4. **Reference examples**:
   - `/data/ms-swift/examples/train/grpo/`
   - `/data/ms-swift/swift/ui/llm_grpo/`

## Troubleshooting

**Dataset loading errors**:
- Ensure Stage-A JSONL has required fields: `group_id`, `mission`, `label`, `per_image`
- Check mission names match exactly (including parentheses and full-width chars)
- Verify Stage-A JSONL files exist at specified paths

**Reward function issues**:
- Run unit tests: `python -m src.stage_b.test_rewards`
- Check model-generated outputs match two-line format
- Inspect reward logs during training to verify computation

**GRPO setup**:
- Verify `num_generations >= 2`
- Ensure batch size is divisible by `num_generations`
- Check device availability and memory
- Confirm vision encoder and aligner are frozen (check parameter counts)

## Future Enhancements (v2)

- **Consistency reward**: Check reasoning vs Stage-A summaries
- **Soft length penalty**: Discourage overly brief/verbose responses
- **Diversity bonus**: Reward varied phrasings
- **Multimodal Stage-B**: Optional variant that re-feeds images during GRPO


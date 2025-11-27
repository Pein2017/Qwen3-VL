---
title: Add GRPO Group-Level Reasoning Pipeline for BBU QC
author: Qwen3-VL
created: 2025-10-24
change-id: 2025-10-24-add-grpo-group-reasoning
status: in-progress
---

## Why
BBU installation QC requires group-level pass/fail judgments based on multiple images. We have per-image summaries (Stage-A) but need reasoning over groups with sparse binary labels. GRPO enables learning from minimal supervision (label + format rewards) while maintaining strict two-line output format.

## What Changes
- **Stage-A**: Per-image Chinese text summary inference with batch processing and strict 图片_{i} validation
- **Stage-B**: Text-only group reasoning dataset from Stage-A summaries (no images)
- **GRPO Integration**: LLM-only LoRA training (freeze ViT/Aligner) with label + format rewards
- **Output Contract**: Strict two-line format (verdict + reasoning in Chinese)

## Impact
- Affected specs: [NEW] grpo-integration (unified capability with 4 phases)
- Affected code:
  - `src/stage_a/` - Image-level inference engine
  - `src/stage_b/` - Dataset builder, prompts, rewards
  - `scripts/run_grpo.py` - Training launcher
  - `configs/grpo_*.yaml` - Training configurations

## Implementation Status

### ✅ Phase 1: Stage-A Image-Level Inference (COMPLETED)
- Per-image inference with mission-dependent Chinese prompts
- Batch processing with configurable batch_size (~4-5x speedup)
- Streaming JSONL output with 图片_{i} keys
- Strict validation: fail-fast on empty summaries or alignment mismatches
- Group ID extraction with regex fallback
- Implemented in `src/stage_a/` with shell launcher

### ✅ Phase 2: Stage-B Dataset (COMPLETED)
- Converts Stage-A JSONL → GRPO-ready format
- One sample per group (text-only, no images)
- Messages format: system prompt + user message embedding Stage-A summaries
- Preserves task_type and group_label for reward computation
- Implemented in `src/stage_b/dataset.py`, `prompts.py`

### ✅ Phase 3: GRPO Training (COMPLETED)
- Python launcher (`scripts/run_grpo.py`) using ms-swift structure
- LLM-only LoRA targeting last-K transformer blocks (default K=4)
- Freeze vision encoder + aligner (Stage-B is text-only)
- Reward functions: label matching (1.0) + format validation (0.2)
- Unit tests for rewards in `src/stage_b/test_rewards.py`

### ⏳ Phase 4: Integration & Deployment (PENDING)
- End-to-end pipeline orchestration
- Production inference with trained adapters
- Monitoring and evaluation tools

## Progress
- **40/47 tasks completed** (~85%)
- Remaining: Pipeline orchestration, deployment scripts, evaluation tooling

## Related Documentation
- `docs/DENSE_SUMMARY_VARIANT.md` - Summary mode background
- `docs/GROUP_QC_GRPO.md` - Detailed pipeline documentation
- ms-swift GRPO trainer integration

## Terminology

**Stage-A**: Image-level text summary generation
- Input: Raw images
- Process: Per-image inference using vision+language model
- Output: Chinese single-line text summary per image

**Stage-B**: Group-level judgment inference
- Input: Stage-A text summaries only (no images)
- Process: Text-to-text reasoning using LLM (frozen vision encoder)
- Output: Two lines (verdict + reasoning)

**GRPO**: Group Relative Policy Optimization
- Reinforcement learning fine-tuning method
- Optimizes policy using group-based comparisons
- Suitable for sparse reward scenarios

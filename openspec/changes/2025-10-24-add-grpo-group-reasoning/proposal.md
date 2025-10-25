# Change Proposal: Add GRPO Group-Level Reasoning Pipeline for BBU QC

- change-id: 2025-10-24-add-grpo-group-reasoning
- date: 2025-10-24
- owner: Qwen3-VL
- status: in-progress

## Summary
Introduce a two-stage reasoning pipeline for BBU installation QC:

**Stage-A: Image-Level Text Summary** (✅ COMPLETED)
- **Input**: Individual images from mission-based directories
- **Output**: Chinese single-line text summary per image, grouped by scene
- **Implementation**: Batch inference with 4-5x speedup; streaming JSONL writes; strict 图片_{i} validation
- **Location**: `src/stage_a/` module + `scripts/stage_a_infer.sh` launcher

**Stage-B: Group-Level Judgment Inference** (IN PROGRESS)
- **Input**: Stage-A text summaries (no images)
- **Output**: Two-line verdict (通过/不通过) + reasoning
- **Training**: GRPO with binary label reward + format reward
- **Components**: Dataset builder (Stage-A JSONL → GRPO format) + Python launcher

**Integration** (PENDING)
- ms-swift GRPO trainer with LLM-only LoRA (last-K transformer blocks)
- Freeze ViT + Aligner (text-only reasoning, no vision inputs)
- Reward functions: label matching + two-line format validation

## Motivation
We have summary-mode SFT that generates per-image text summaries (Stage-A). The next step is group-level reasoning: given summaries from multiple images in a scene, infer a Pass/Fail verdict with reasoning (Stage-B). Since we only have binary group-level labels (通过/不通过), GRPO with minimal rewards (label + format) enables learning from this sparse supervision while maintaining strict output format.

## Goals

✅ **Stage-A: Image-Level Text Summary Engine** (COMPLETED)
- Per-image inference with mission-dependent Chinese prompts
- Batch processing with configurable batch_size (~4-5x speedup)
- Streaming JSONL output with 图片_{i} keys
- Strict validation: fail-fast on empty summaries or alignment mismatches
- Implemented in `src/stage_a/` with shell launcher

🔄 **Stage-B: Group-Level Judgment Dataset Builder** (IN PROGRESS)
- Convert Stage-A JSONL → GRPO-ready format
- One sample per group (text-only, no images)
- Messages format: system prompt + user message embedding Stage-A summaries
- Preserve task_type and group_label for reward computation

🔄 **Stage-B: Model Output Contract**
- Exactly two lines in Chinese
- Line 1: "通过" or "不通过" (verdict only, no extra tokens)
- Line 2: "理由: <reasoning>" (can reference 图片_{i}, no direct copy-paste)

⏳ **GRPO Integration** (PENDING)
- Python launcher (`scripts/run_grpo.py`) using ms-swift `GRPOTrainer`
- LLM-only LoRA targeting last-K transformer blocks (default K=4)
- Freeze vision encoder + aligner (Stage-B is text-only)
- Reward functions: label matching (1.0 weight) + format validation (0.2 weight)

## Non-Goals
- Building a new trainer (reuse ms-swift GRPO/TRL).
- Changing SFT pipeline or chat template behavior.
- Implementing a complex consistency reward in v1 (will follow).

## Implementation Status

- ✅ **Stage-A**: Fully implemented
  - `src/stage_a/` module with batch inference
  - Mission-dependent prompts
  - Streaming JSONL output with 图片_{i} alignment
  - `scripts/stage_a_infer.sh` launcher

- ✅ **Stage-B**: Fully implemented
  - `src/stage_b/dataset.py` - loads Stage-A JSONL for GRPO
  - `src/stage_b/prompts.py` - Chinese system/user templates
  - `src/stage_b/rewards.py` - label + format rewards
  - `src/stage_b/test_rewards.py` - unit tests
  - Dynamic generation during GRPO rollout (no pre-built outputs)

- ✅ **GRPO Integration**: Launcher ready
  - `scripts/run_grpo.py` - Python launcher with ms-swift structure
  - LLM-only LoRA configuration (last-K blocks)
  - Vision/aligner freezing
  - Reward function integration
  - Ready for ms-swift GRPOTrainer completion

## Open Questions (Resolved)
- ✅ Group ID extraction: Implemented with regex `^(QC-[A-Za-z]+-[0-9]{8}-[0-9]+)` + subdirectory fallback
- ✅ Task types: Fixed enum of 4 missions implemented in Stage-A prompts
- ✅ Batch strategy: Hybrid batching with automatic chunking for large groups

## Terminology (Clear Definitions)

**Stage-A**: Image-level text summary generation
- Input: Raw images
- Process: Per-image inference using vision+language model
- Output: Chinese single-line text summary per image

**Stage-B**: Group-level judgment inference given Stage-A summaries
- Input: Stage-A text summaries only (no images)
- Process: Text-to-text reasoning using LLM (frozen vision encoder)
- Output: Two lines (verdict + reasoning)

## Related
- docs/DENSE_SUMMARY_VARIANT.md (summary mode background)
- docs/GROUP_QC_GRPO.md (detailed pipeline documentation)
- /data/Qwen2.5-VL-main/src_post (legacy prompting reference)
- ms-swift GRPO trainer integration

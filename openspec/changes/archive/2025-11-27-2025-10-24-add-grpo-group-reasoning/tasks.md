# Tasks: GRPO Group-Level Reasoning

## ✅ Completed Tasks

### 1) Stage-A Inference Engine
- [x] Implement `src/stage_a/inference.py` with batch inference and group aggregation
- [x] Add mission-dependent prompts in `src/stage_a/prompts.py`
- [x] Create CLI entry point `src/stage_a/cli.py`
- [x] Add convenience launcher `scripts/stage_a_infer.sh`
- [x] Validate 图片_{i} alignment and fail-fast on mismatches
- [x] Add `src/stage_a/README.md` with usage examples
- [x] Test batch inference speedup (~4-5x validated)
- [x] Verify streaming JSONL output and resumability

## ✅ Completed Tasks (continued)

### 2) Stage-B Dataset Loader & Prompts
- [x] Create `src/stage_b/dataset.py` module to load Stage-A JSONL for GRPO
- [x] Implement `load_stage_a_for_grpo()` function with multi-file support
- [x] Define Chinese system/user message templates per task_type in `src/stage_b/prompts.py`
- [x] Build messages dynamically (no pre-built Stage-B outputs)
- [x] Embed Stage-A summaries (图片_{i}) into user message as plain text
- [x] Return dataset with schema: `{group_id, task_type, group_label, stage_a_summaries, messages}`
- [x] Add `src/stage_b/README.md` with full documentation

### 3) Reward Functions
- [x] Create `src/stage_b/rewards.py` module
- [x] Implement `label_reward`: binary match for 通过/不通过
- [x] Implement `format_reward`: two-line format validation (第一行: verdict, 第二行: 理由)
- [x] Add `consistency_reward` placeholder for grpo_summary_1024_attr_key_recall
- [x] Create unit tests in `src/stage_b/test_rewards.py` with curated examples
- [x] Document reward function signatures and expected inputs
- [x] Add reward function registry and getter

### 4) GRPO Python Launcher
- [x] Create `scripts/run_grpo.py` with programmatic ms-swift integration structure
- [x] Configure LLM-only LoRA targeting last-K transformer blocks (default K=4)
- [x] Implement module freezing for ViT and Aligner
- [x] Register reward functions (label + format) with weighted combination
- [x] Set reward weights (default: label:format = 1.0:0.2)
- [x] Validate `num_generations >= 2` in config
- [x] Add CLI with Stage-A JSONL input support
- [x] Document ms-swift integration steps for completion

### 5) Documentation & Integration
- [x] Update `src/stage_b/README.md` with complete workflow
- [x] Document dynamic generation approach (no pre-built Stage-B outputs)
- [x] Add usage examples for all components
- [x] Document integration with Stage-A outputs
- [x] Add troubleshooting section
- [x] Update OpenSpec proposal, design, and tasks

### 6) Documentation Updates
- [x] Update `docs/GROUP_QC_GRPO.md` with pipeline overview
- [x] Add Stage-B dataset schema examples
- [x] Document GRPO launcher usage and configuration
- [x] Add troubleshooting section for common issues

## 🔮 Future Extensions (grpo_summary_1024_attr_key_recall+)

### 7) Enhanced Rewards (Backlog / not in v1 scope)
- Implement consistency reward (evidence vs Stage-A summaries)
- Add soft length penalty reward
- Add diversity bonus reward

### 8) Advanced Features (Backlog / not in v1 scope)
- Multimodal Stage-B variant (re-feed images during GRPO)
- Multi-mission joint training
- Incremental GRPO with checkpoint resumption
- Alternative group-id extraction strategies

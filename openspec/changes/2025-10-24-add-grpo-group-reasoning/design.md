# Design: GRPO Group-Level Reasoning for BBU QC

## Architecture Overview (Updated)

### Stage-A Inference Engine (✅ IMPLEMENTED)
- **Location**: `src/stage_a/` module + `scripts/stage_a_infer.sh` launcher
- **Input**: Mission-based directory structure: `<root>/<mission>/{审核通过,审核不通过}/<group_id>/*.{jpg,jpeg,png}`
- **Processing**: Batch inference (configurable batch_size, default 4); hybrid batching with automatic chunking for large groups; ~4-5x speedup over sequential
- **Prompts**: Mission-dependent system and user prompts in Chinese (see `src/stage_a/prompts.py`)
- **Output**: Streaming JSONL (one file per mission); each line = one group with `{group_id, mission, label, images, per_image{图片_i: summary}, raw_texts, clean_texts, timestamp}`
- **Validation**: Strict fail-fast on empty summaries or 图片_{i} misalignment
- **Key features**: Resumable (writes incrementally), monitorable (`tail -f`), safe interruption (Ctrl+C preserves completed groups)

### Stage-B GRPO Dataset (IN PROGRESS)
- **Builder**: `scripts/build_stage_b_dataset.py` (to be implemented)
- **Input**: Stage-A JSONL files (one or more missions)
- **Output**: GRPO-ready JSONL with fields: `{group_id, task_type, group_label∈{通过,不通过}, stage_a_summaries{图片_i: str}, messages[{role, content}]}`
- **Messages format**:
  - System: Chinese task-specific context and output format instructions
  - User: Embeds task focus + all Stage-A summaries as plain text (no images)
- **Model output contract**: Two lines（第一行：通过 or 不通过；第二行：理由: <中文自然语言>）

### GRPO Training (PENDING)
- **Launcher**: `scripts/run_grpo.py` (Python, not CLI)
- **Model config**: LLM-only LoRA on last-K transformer blocks (default K=4); freeze ViT + Aligner
- **Rewards v1**: Binary label reward + two-line format reward (consistency reward deferred to v2)
- **Integration**: Uses ms-swift `GRPOTrainer` with custom reward functions

## Data Flow (Refined)
```
Mission images directory (with labels)
  ↓ (Stage-A: batch inference, streaming writes)
Grouped JSONL per mission (图片_{i} summaries)
  ↓ (Stage-B dataset builder)
GRPO-ready JSONL (text-only messages)
  ↓ (GRPO launcher: load model + LoRA config)
Rollout generation (num_generations ≥ 2)
  ↓ (Reward functions: label + format)
LoRA updates (LLM last-K blocks only)
  ↓ (Save checkpoint)
Merged or adapter checkpoint for deployment
```

## Key Decisions (Updated)
- **Batch inference**: Implemented hybrid strategy with configurable batch_size; automatic chunking for large groups; ~4-5x speedup validated
- **Streaming writes**: Stage-A writes incrementally (not atomic per mission) for resumability and progress monitoring
- **Mission-based organization**: One output JSONL per mission; simplifies Stage-B filtering and debugging
- **Chinese-only outputs**: All prompts and summaries in Chinese; allow English tokens (BBU, ODF, etc.)
- **Strict validation**: Fail-fast on 图片_{i} misalignment to protect data quality; no silent corrections
- **Minimal rewards v1**: Start with label + format only; defer consistency to v2 when heuristics mature
- **Text-only Stage-B**: No image inputs during GRPO; purely reasoning over Stage-A text summaries

## Risks & Mitigations (Updated)
- **Group ID parsing variance** → Implemented regex `^(QC-[A-Za-z]+-[0-9]{8}-[0-9]+)` with subdirectory fallback; covers observed patterns
- **Noisy Stage-A summaries** → Using temperature=0.0 (greedy), max_new_tokens=1024, repetition_penalty=1.05; validated on sample data
- **Batch OOM** → Configurable batch_size (default 4); automatic fallback to sequential if needed; max_pixels cap at 786432
- **Overfitting to format in GRPO** → Plan to mix reward weights (e.g., label:format = 1.0:0.2) and cap max_new_tokens=128 for Stage-B
- **Inconsistent verdicts** → Stage-B will use greedy decoding initially; consider temperature scheduling in later iterations

## Extensibility (Roadmap)
- **v2 rewards**: Consistency reward (evidence vs Stage-A summaries); soft length penalty; diversity bonus
- **Multimodal Stage-B**: Optional variant that feeds images again during GRPO (toggle via config flag)
- **Alternative group-id extractors**: Strategy pattern in `src/stage_a/inference.py` allows custom extraction logic
- **Multi-mission training**: Combine all 4 missions in one GRPO dataset for better generalization
- **Incremental GRPO**: Resume from previous checkpoint with new Stage-A data batches

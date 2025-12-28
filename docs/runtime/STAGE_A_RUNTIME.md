# Stage-A Runtime Guide

Single-image summarization runbook for generating the evidence JSONL that Stage-B consumes. Use this file as the engineer-facing reference for commands, inputs, outputs, and common flags.

---

## Purpose
- Produce per-image summaries with strong rare/long-tail coverage for downstream group-level verdicts.

## Entrypoint
- `scripts/stage_a.sh` wraps `python -m src.stage_a.cli` with mission defaults and device selection.

```bash
# Single-GPU mode
mission=挡风板安装检查 gpus=0 verify_inputs=true \
  bash scripts/stage_a.sh

# Override defaults (checkpoint/input/output) + optional postprocess
checkpoint=/path/to/ckpt input_dir=/path/to/groups output_dir=/path/to/out \
  mission=挡风板安装检查 postprocess=true gpus=0 \
  bash scripts/stage_a.sh

# Multi-GPU mode (ticket-parallel via torchrun)
mission=BBU安装方式检查（正装） gpus=0,1,2,3 \
  bash scripts/stage_a.sh

# Use sampling to balance pass/fail (seed default: 42)
mission=BBU安装方式检查（正装） gpus=0 pass_group_number=500 fail_group_number=500 \
  bash scripts/stage_a.sh
```

## Inputs
- Layout: `<root>/<mission>/{审核通过|审核不通过}/<group_id>/*.{jpg,jpeg,png}`
- Labels inferred from parent directory: `审核通过` → `pass`, `审核不通过` → `fail`.
- Resubmissions are allowed: if the same `group_id` exists in both label folders, Stage-A treats each occurrence as a separate record with its respective label.

## Outputs
- JSONL at `<output_dir>/<mission>_stage_a.jsonl` with keys `group_id`, `mission`, `label`, `images`, and normalized `per_image` entries (`image_1`, `image_2`, ...).
- Format aligns with the canonical schema in `../data/DATA_JSONL_CONTRACT.md`.
- In multi-GPU mode:
  - `sharding_mode=per_group`: each rank writes to `<mission>_stage_a.rank{rank}.jsonl` temporarily; rank 0 merges into the final `<mission>_stage_a.jsonl` and removes temp files.
  - `sharding_mode=per_image`: each rank writes to `<mission>_stage_a.images.rank{rank}.jsonl` temporarily; rank 0 merges into the final `<mission>_stage_a.jsonl` and deletes intermediates by default.

## Critical: Batched Generation Requires Left Padding (Top Urgency)
Stage-A uses **batched generation** (multiple independent samples in a batch) rather than “packing” multiple samples into one long sequence.

For Qwen3-VL **decoder-only** generation, **variable-length prompts** are common because the number of vision tokens depends on image resolution (`image_grid_thw`). In this setting:
- **Right padding is unsafe** for batched generation: shorter samples end with `PAD` tokens, and generation can start from the wrong context, producing unstable/garbage prefixes.
- **Left padding is required**: shorter samples are padded on the left so the last non-pad token is aligned across the batch.

Implementation note:
- `transformers/models/qwen3_vl/processing_qwen3_vl.py` does not force a padding side; it inherits `tokenizer.padding_side`.
- Stage-A MUST set `tokenizer.padding_side="left"` (and typically `truncation_side="left"`) before calling `processor(..., padding=True)` for batch inference.
- Stage-A enforces this in `src/stage_a/inference.py` (`load_model_processor`).

Troubleshooting symptom:
- If summaries suddenly start with unrelated English prefixes (e.g., `CoreApplication...`) especially under `sharding_mode=per_image` or mixed-resolution batches, check that **left padding** is enabled (right padding can trigger this failure mode).

## Key Flags / Env Vars
- `mission` (required) — must match `SUPPORTED_MISSIONS`.
- `dataset` — `bbu|rru`; controls which domain knowledge pack is appended to the runtime summary prompt.
- `prompt_profile` — summary prompt profile (`summary_runtime` default).
  - `summary_runtime` 组合方式：system prompt = summary 任务基座 + 全局“非现场/图纸”规则；user prompt = 摘要指令 + BBU/RRU 场景提示块 + 任务重点。
- `checkpoint` / `input_dir` / `output_dir` — overrides the defaults inside `scripts/stage_a.sh`.
- `verify_inputs` — logs first-chunk hashes and grid/token counts; optional.
- `gpus` — GPU device selection (comma-separated list, e.g., `gpus=0` or `gpus=0,1,2,3`). Use `gpus=cpu` for CPU mode.
  - Single GPU: `gpus=0` → single-process execution
  - Multiple GPUs: `gpus=0,1,2,3` → auto-launches `torchrun` with ticket-parallel sharding across ranks
  - `sharding_mode=per_group`: each rank processes `groups[rank::world_size]` and writes to a temp file; rank 0 merges all outputs into the canonical JSONL
  - `sharding_mode=per_image`: each rank processes per-image jobs `jobs[rank::world_size]`; rank 0 merges intermediates into the canonical JSONL
- `pass_group_number` — Optional cap on the number of `pass` groups to keep. When the mission contains more than this number, Stage-A samples down to the value using the `sample_seed`.
- `fail_group_number` — Optional cap on the number of `fail` groups to keep; sampling is done independently for the fail subset.
- `sample_seed` — Random seed controlling pass/fail sampling (default: 42). Use the same seed to reproduce the same sampled split across runs.
- `batch_size_per_rank` — per-rank batch size (`--batch_size` in CLI). Default is 32 in `scripts/stage_a.sh`.
- `max_pixels` — image resizing budget (e.g., `1048576` = 1024×1024). Default is higher for `dataset=rru` unless overridden.
- `max_new_tokens` — generation cap for summaries (default: 1024 in `scripts/stage_a.sh`).
- `temperature` / `top_p` / `repetition_penalty` — decoding parameters (defaults in `scripts/stage_a.sh`).
- `log_level` / `debug` — logging controls for Stage-A CLI.
- `sharding_mode` — Distributed sharding strategy (default: `per_group`):
  - `per_group`: shard work at group granularity; batching stays within each group (no cross-group mixing).
  - `per_image`: shard work at image granularity; batches are per-rank; rank 0 merges per-image intermediates into group-level JSONL.
    - Note: per-image text outputs may differ under stochastic decoding (e.g., `temperature>0`); use `temperature=0.0` for regression comparisons.
- `keep_intermediate_outputs` — Keep intermediate per-rank per-image JSONL outputs in `per_image` mode (default: delete after successful merge).
- `postprocess` — run `python -m src.stage_a.postprocess --inplace` after inference (RRU/BBU cleanup).
- `add_gt_fail_reason` / `excel_path` — optionally attach `gt_fail_reason_text` from Excel (BBU-only; uses `scripts/add_gt_fail_reason_to_stage_a.py`, default Excel path `output_post/BBU_scene_latest.xlsx`).

```bash
# Image-level sharding (single GPU)
mission=挡风板安装检查 gpus=0 sharding_mode=per_image \
  bash scripts/stage_a.sh

# Image-level sharding (multi GPU)
mission=挡风板安装检查 gpus=0,1,2,3 sharding_mode=per_image \
  bash scripts/stage_a.sh

# Keep per-image intermediates for debugging
mission=挡风板安装检查 gpus=0,1 sharding_mode=per_image keep_intermediate_outputs=true \
  bash scripts/stage_a.sh
```

## Checkpoint Choices
- LoRA adapters for iteration; merged checkpoints for production (see `scripts/train.sh` exports or `swift export --merge_lora true`).
- 生产约束：Stage‑A 与 Stage‑B 在最终部署中共用同一个 Qwen3‑VL 模型（同一组权重/LoRA 组合），避免双模型部署成本；所有针对摘要任务的微调都需要显式考虑 Stage‑B 判决能力的保留。

## See Also
- Stage-B runtime (group verdicts): `./STAGE_B_RUNTIME.md`
- Business overview of the two-stage pipeline: `./STAGE_A_STAGE_B.md`

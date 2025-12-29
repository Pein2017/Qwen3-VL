# RL Post-Training for Summary Format Stabilization (Findings to Date)

## Objective
Stabilize summary outputs so they always follow the strict two-line contract:
- Line 1: `<DOMAIN=...>, <TASK=...>`
- Line 2: `无关图片` **or** a single-line JSON string ending in `}` with no trailing characters.

Reference contracts:
- `src/prompts/summary_core.py`
- `docs/data/DATA_AND_DATASETS.md`

## Current Contract Anchors
- Summary prompt explicitly mandates the two-line format, strict JSON validity, and no trailing characters after `}`.
- Summary mode requires a non-empty `summary` string in each record; irrelevant images must be `summary: 无关图片`.

## ms-swift RLHF Capabilities (Local Docs)
### Supported Algorithms & Multimodal Coverage
- GRPO is listed as supported for multimodal training.
- PPO is listed as **not** supported for multimodal training.
- DPO/KTO/GKD are listed as multimodal-capable but require preference-style datasets.

### GRPO Reward Integration
- GRPO accepts custom rule-based reward functions via `reward_funcs` and `external_plugins`.
- Reward functions receive model completions plus all dataset columns as kwargs.
- Reward models can be attached via `reward_model` + `reward_model_plugin` (slower, usually unnecessary for format-only tasks).

### Data Access Inside Reward Functions
- Dataset columns are passed through; standard multimodal keys (`images`, `objects`, etc.) are preserved.
- Image fields are normalized to dicts like `{path: ...}` or `{bytes: ...}` by the dataset preprocessor.
- GRPO rollouts remove any assistant response from dataset messages before sampling, so the dataset can remain in standard summary JSONL format.

## Repository Integration Points
- `src/sft.py` switches into RLHF mode when `rlhf_type` is set (uses `SwiftRLHF`).
- `custom.assistant_prefix_format` is **required** for BBU/RRU fusion training; it emits line-1 tags.
- Domain/task tokens are derived from dataset name and mode via `src/datasets/assistant_prefix.py`.
- Fusion metadata includes `_fusion_domain_token` (BBU/RRU) for `<DOMAIN>` verification, and summary-mode rows attach `metadata.summary_ref` for content accuracy rewards.

## LoRA + GRPO Notes
- For LoRA GRPO, ms-swift recommends setting the **same adapter** for both policy and reference:
  - `adapters: [<sft_lora>]` (trainable policy)
  - `ref_adapters: [<sft_lora>]` (frozen reference)
- This keeps the KL penalty aligned with the LoRA-adapted policy rather than the base model only.

## Transformers / TRL Status (Local Env)
- `transformers` is installed, but RLHF trainers are not provided by core transformers.
- TRL is installed, but this project’s pipeline is already wired to ms-swift via `src/sft.py`.

## Implications for BBU + RRU Summary Stabilization (LoRA)
- GRPO with a **rule-based format reward** is the most direct fit for format compliance.
- The reward can be purely structural: two-line format, exact `<DOMAIN>/<TASK>` match, JSON parse validity, and no trailing characters.
- Existing fusion configs (e.g., `configs/fused_data/bbu_rru_summary.yaml`) already enforce the tag prefix via `assistant_prefix_format` and summary mode defaults.

## Open Questions / Decisions Needed
- Exact reward weighting scheme (single composite reward vs. multiple sub-rewards).
- Whether to include any light length penalty (e.g., overlong) to discourage verbose outputs.
- Rollout settings (current project guidance): set backward size via `training.effective_batch_size` (with `per_device_train_batch_size` as the micro-batch), set rollout size via `rlhf.generation_batch_size` (global trajectories per generation), ensure `rlhf.num_generations` divides `generation_batch_size`, and keep `temperature=0.3`, `max_completion_length=2048` unless retuning.

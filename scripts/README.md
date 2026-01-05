# Scripts Overview

Canonical entrypoints for training, inference, and inspection runs. Prefer these over ad-hoc commands so logging, seeds, and env setup stay consistent.

| Script | Purpose | Read more |
|--------|---------|-----------|
| `train.sh` | Launches `python -m src.sft` / `torchrun` with config auto-resolution and ms env setup. | `docs/training/TRAINING_PLAYBOOK.md`, `docs/training/REFERENCE.md` |
| `fuse_datasets.py` | Builds fused JSONL from fusion YAML (deterministic mixing of target+aux sources). | `docs/data/UNIFIED_FUSION_DATASET.md` |
| `download.py` | Downloads raw/public corpora per instructions. | `docs/data/DATA_AND_DATASETS.md`, `docs/data/PUBLIC_DATA.md` |
| `stage_a.sh` | Stage‑1 basic object recognition; emits per-image summaries JSONL. | `docs/runtime/STAGE_A_RUNTIME.md`, `docs/runtime/STAGE_A_STAGE_B.md` |
| `stage_b.sh` | Stage‑2 rule-search loop (ingest → rollout → propose rules → gate/eval → apply) returning per-ticket verdicts + rule-search artifacts. | `docs/runtime/STAGE_B_RUNTIME.md`, `docs/runtime/STAGE_A_STAGE_B.md` |
| `stage_b_smoke.py` | No-model Stage‑B smoke/audit (config+ingest+guidance+prompt+parse+export). | `docs/runtime/STAGE_B_RUNTIME.md` |
| `merge_stage2_lora.sh` | Merges Stage‑2 LoRA checkpoints for deployment. | `docs/runtime/STAGE_B_RUNTIME.md` |
| `stage_b_split_distill.py` | Splits `distill_chatml.jsonl` into train/val JSONL files. | `docs/runtime/STAGE_B_RUNTIME.md` |
| `postprocess_rule_search_hard_cases.py` | Extracts GT-fail & pred-pass hard cases and attaches GT fail reasons from Excel. | `docs/runtime/STAGE_B_RUNTIME.md` |
| `run_rule_search_postprocess.sh` | Wrapper for rule-search hard-case postprocess with local paths. | `docs/runtime/STAGE_B_RUNTIME.md` |
| `debug_fusion_template_clone.py` | Regression probe for fusion template reuse (mask ratio / template cloning). | `docs/data/UNIFIED_FUSION_DATASET.md` |
| `validate_sft_config.py` | Fast YAML validation for SFT configs (no model weights). | `docs/training/REFERENCE.md` |
| `validate_dense_jsonl_contract.py` | Fast JSONL contract validation for dense-caption records. | `docs/data/DATA_JSONL_CONTRACT.md` |
| `build_irrelevant_summary_jsonl.py` | Builds `data/irrelevant_summary/train.jsonl` from `data/irrelevant_summary/images/*.jpg|*.jpeg` with summary `无关图片` (dummy full-frame bbox). | `docs/data/DATA_JSONL_CONTRACT.md` |

Usage tips
- Run with `ms` conda env activated (`conda activate ms`), unless the script handles it.
- Training configs now live under `configs/train/` (runnable presets) and `configs/components/` (reusable blocks); `configs/debug.yaml` remains the default quick entrypoint.
- Prefer setting seeds in configs; scripts propagate env when present.
- Keep inputs/output paths mission-specific to avoid cross-run collisions.
- Stage‑B no-model audit: `bash scripts/stage_b.sh smoke`

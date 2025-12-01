# Scripts Overview

Canonical entrypoints for training, inference, and inspection runs. Prefer these over ad-hoc commands so logging, seeds, and env setup stay consistent.

| Script | Purpose | Read more |
|--------|---------|-----------|
| `train.sh` | Launches `python -m src.sft` / `torchrun` with config auto-resolution and ms env setup. | `docs/TRAINING_PLAYBOOK.md`, `docs/REFERENCE.md` |
| `fuse_datasets.py` | Builds fused JSONL from fusion YAML (deterministic mixing of target+aux sources). | `docs/UNIFIED_FUSION_DATASET.md` |
| `download.py` | Downloads raw/public corpora per instructions. | `docs/DATA_AND_DATASETS.md`, `docs/PUBLIC_DATA.md` |
| `stage_a_infer.sh` | Stage‑1 basic object recognition; emits per-image summaries JSONL. | `docs/STAGE_B_RUNTIME.md`, `docs/STAGE_A_STAGE_B.md` |
| `stage_b_run.sh` | Stage‑2 verdict loop (ingest → rollout → selection → reflection) returning `pass|fail` per ticket. | `docs/STAGE_B_RUNTIME.md`, `docs/STAGE_A_STAGE_B.md` |
| `merge_stage2_lora.sh` | Merges Stage‑2 LoRA checkpoints for deployment. | `docs/STAGE_B_RUNTIME.md` |
| `debug_fusion_template_clone.py` | Regression probe for fusion template reuse (mask ratio / template cloning). | `docs/UNIFIED_FUSION_DATASET.md` |

Usage tips
- Run with `ms` conda env activated (`conda activate ms`), unless the script handles it.
- Prefer setting seeds in configs; scripts propagate env when present.
- Keep inputs/output paths mission-specific to avoid cross-run collisions.

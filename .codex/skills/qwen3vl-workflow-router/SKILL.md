---
name: qwen3vl-workflow-router
description: "Router for Qwen3-VL workflows (dense-caption SFT, summary SFT, dataset fusion, Stage-A inference, Stage-B training-free rule_search). Use to select entry scripts, configs, fast validations, and canonical docs/code locations."
---

# Qwen3-VL Workflow Router

Use this skill to route requests to the correct workflow, scripts, configs, and docs.

## 0) Routing questions (reduce thrash)

- Which workflow: SFT training, data conversion/fusion, Stage-A inference, or Stage-B rule_search?
- What changed: YAML only vs code (datasets/augmentation/prompts/stage_b logic)?
- Smallest reproducible input: JSONL path(s), mission name, and 1–3 representative records/tickets.

## 1) Canonical docs (do not duplicate)

- Doc index + map: `docs/README.md`
- Training: `docs/training/TRAINING_PLAYBOOK.md`, `docs/training/REFERENCE.md`
- Data contract & fusion: `docs/data/DATA_JSONL_CONTRACT.md`, `docs/data/DATA_AND_DATASETS.md`, `docs/data/UNIFIED_FUSION_DATASET.md`
- Stage-A runtime: `docs/runtime/STAGE_A_RUNTIME.md`
- Stage-B runtime: `docs/runtime/STAGE_B_RUNTIME.md`, `docs/runtime/STAGE_A_STAGE_B.md`
- Stage-B mission knowledge: `docs/stage-B-knowledge-Chinese.md`
- OpenSpec: `openspec/AGENTS.md`, `openspec/project.md`

## 2) Entry scripts

SFT training:
```bash
conda run -n ms bash scripts/train.sh config=/abs/path/to/config.yaml gpus=0
```

Fusion (offline JSONL builder):
```bash
conda run -n ms python scripts/fuse_datasets.py --config /abs/path/to/fusion.yaml
```

Stage-A inference:
```bash
mission=<mission> gpus=0 bash scripts/stage_a.sh
```

Stage-B rule_search (only supported mode):
```bash
config=bbu_line gpus=0 bash scripts/stage_b.sh
```

Stage-B no-model smoke:
```bash
bash scripts/stage_b.sh smoke
```

Stage-B baseline-only (skip proposer/reflection; for manual audit):
```bash
jump_reflection=true config=bbu_line gpus=0 bash scripts/stage_b.sh
```

## 3) Fast validations (run before full jobs)

SFT:
- `conda run -n ms python scripts/validate_sft_config.py --config <yaml>`
- `conda run -n ms python scripts/validate_dense_jsonl_contract.py --jsonl <train.jsonl> --limit 50`

Fusion:
- Confirm source JSONL satisfies `docs/data/DATA_JSONL_CONTRACT.md`
- Prefer `scripts/fuse_datasets.py` over ad-hoc mixing

Stage-B:
- `bash scripts/stage_b.sh smoke`
- For prompt changes, inspect `rule_candidates.jsonl` + `benchmarks.jsonl` first

## 4) Code map (where to edit)

- SFT runner: `src/sft.py`, `src/config/loader.py`, `src/config/schema.py`
- Datasets/builders: `src/datasets/`, `src/datasets/builders/jsonlines.py`, `src/datasets/unified_fusion_dataset.py`
- Augmentation: `src/datasets/augmentation/`, `src/datasets/preprocessors/augmentation.py`, `src/datasets/geometry.py`
- Stage-A: `src/stage_a/`, `scripts/stage_a.sh`
- Stage-B: `src/stage_b/` (config/ingest/prompts/rollout/scoring/rule_search/export)

## 5) Debug configs to start from

- SFT dense tiny: `configs/debug.yaml` or `configs/smoke/sft_dense_tiny.yaml`
- SFT summary tiny: `configs/smoke/sft_summary_tiny.yaml`
- Stage-B tiny: `configs/smoke/stage_b_tiny.yaml` or no-model smoke

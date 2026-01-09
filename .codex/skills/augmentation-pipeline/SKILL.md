---
name: augmentation-pipeline
description: "Qwen3-VL augmentation pipeline router: YAML -> Compose ops -> record preprocessors -> curriculum/telemetry -> tests/vis. Use for adding/debugging augmentation ops without duplicating docs."
---

# Qwen3-VL Augmentation Pipeline (Router)

Keep `docs/` as the source of truth. This skill is a fast navigation + validation loop.

## 0) Canonical docs

- Data augmentation overview: `docs/data/DATA_AUGMENTATION.md`
- JSONL contract (geometry keys): `docs/data/DATA_JSONL_CONTRACT.md`

## 1) Quick loop (do this before training)

1) Validate the training YAML (no model):
```bash
conda run -n ms python scripts/validate_sft_config.py --config <yaml>
```

2) Visual sanity (pixel-level):
```bash
conda run -n ms python vis_tools/vis_augment_compare.py \
  --config-yaml <yaml> --jsonl <train.jsonl> --out-dir vis_out/augment_check
```

3) Unit tests:
```bash
conda run -n ms pytest tests/augmentation -q
```

## 2) Where augmentation is wired

- Entry: `scripts/train.sh` → `python -m src.sft --config <yaml>`
- Build pipeline: `src/datasets/augmentation/builder.py`
- Apply per-record: `src/datasets/preprocessors/augmentation.py`
- Compose semantics + flush rules: `src/datasets/augmentation/base.py`
- Ops registry: `src/datasets/augmentation/ops.py`
- Geometry math: `src/datasets/geometry.py`

## 3) “If you change X, open Y”

- Add/modify an op: `src/datasets/augmentation/ops.py`
- Change warp/flush rules: `src/datasets/augmentation/base.py`
- Change crop filtering / object duplication: `src/datasets/preprocessors/augmentation.py`
- Curriculum behavior: `src/datasets/augmentation/curriculum.py` + `src/callbacks/augmentation_curriculum.py`

## 4) Guardrails (don’t regress silently)

- Determinism: only use provided RNG; never global `random`/NumPy RNG.
- Geometry: preserve point order/identity; never drop/reorder points silently.
- Exactly one geometry key per object: `bbox_2d` or `poly` or `line`.

Search tips:
- List ops: `rg -n \"@register\\(\\\"\" src/datasets/augmentation/ops.py`

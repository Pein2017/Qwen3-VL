---
name: qwen3vl-sft-training
description: "Dense-caption + summary-mode SFT training for Qwen3-VL. Use for config creation/validation, dataset JSONL requirements, augmentation wiring, and safe smoke runs via scripts/train.sh."
---

# Qwen3-VL SFT Training (Dense + Summary)

This skill is a **procedural runbook**. Keep `docs/` as canonical truth; link out instead of duplicating.

## 0) Quick start (safe loop)

1) Pick a config (start tiny):
   - Dense tiny: `configs/debug.yaml` or `configs/smoke/sft_dense_tiny.yaml`
   - Summary tiny: `configs/smoke/sft_summary_tiny.yaml`

2) Validate config (no model):
```bash
conda run -n ms python scripts/validate_sft_config.py --config <yaml>
```

3) Validate JSONL schema quickly:
```bash
conda run -n ms python scripts/validate_dense_jsonl_contract.py --jsonl <train.jsonl> --limit 50
```

4) Launch training:
```bash
conda run -n ms bash scripts/train.sh config=<yaml> gpus=0
```

## 1) What must be in the JSONL (dense vs summary)

Canonical contract: `docs/data/DATA_JSONL_CONTRACT.md`.

- Dense mode (`custom.use_summary: false`): records must have `images/objects/width/height`; assistant target is JSON object.
- Summary mode (`custom.use_summary: true`): records must also have a non-empty `summary` **JSON string** (single line). Irrelevant-image samples may use the literal `无关图片`.
- For BBU/RRU targets, configure `custom.assistant_prefix_format` so the assistant output is prefixed with `<DOMAIN=...>, <TASK=...>` + newline (sources remain unchanged).

Tiny example data lives in:
- `demo/data/train_tiny.jsonl`
- `demo/data/val_tiny.jsonl`

## 2) Key config surfaces you’ll edit most

- `custom.train_jsonl` / `custom.val_jsonl`
- `custom.emit_norm` (`none|norm100|norm1000`)
- `custom.json_format` (currently `standard`)
- `custom.use_summary` (toggles summary-mode prompts + builder behavior)
- `custom.assistant_prefix_format` (required for BBU/RRU targets; see `docs/data/DATA_AND_DATASETS.md`)
- `custom.augmentation` (+ curriculum) when training dense (see augmentation skill)
- `custom.fusion_config` to enable dataset fusion (see `configs/fusion/*.yaml`)

## 3) Where augmentation plugs in (don’t guess)

- Training runner: `src/sft.py` constructs augmenter from `custom.augmentation`.
- Dataset: `src/datasets/dense_caption.py` + `src/datasets/preprocessors/augmentation.py`.
- Fusion dataset has per-dataset policies: `src/datasets/unified_fusion_dataset.py`.

If the change is augmentation-related, prefer:
- `conda run -n ms python vis_tools/vis_augment_compare.py`
- `conda run -n ms pytest tests/augmentation -q`

## 4) Docs to open for “how it works”

- `docs/training/REFERENCE.md`
- `docs/training/TRAINING_PLAYBOOK.md`
- `docs/data/DATA_AND_DATASETS.md`
- `docs/data/DATA_AUGMENTATION.md`

## 5) Guardrails / gotchas

- Packing is removed; training uses padded batches only.
- Any config containing `custom.hard_sample_mining` fails validation (deprecated).
- `data.dataset: ["dummy"]` is required by ms-swift validation even for custom datasets.

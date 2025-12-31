# Data Preprocessing & Intake (Annotation → Training JSONL)

Purpose-built guide for the optional offline preprocessing stage that turns human-annotated exports into training-ready JSONL plus QA artifacts.

## When to Run
- You receive raw exports from the annotation platform (BBU/RRU or new domains) and need train/val splits that match the project JSONL contract.
- You want long-tail/rare-object coverage before training or fusion.
- You must validate geometry/taxonomy quality before data reaches `src/datasets/`.

## Position in the Pipeline

```
Human annotations → data_conversion/convert_dataset.sh
  → unified_processor (taxonomy + geometry canonicalization)
  → train/val/tiny JSONL + QA reports
  → (a) direct training via custom.train_jsonl
  → (b) auxiliary sources for fusion (custom.fusion_config)
  → Stage‑1 inference → Stage‑2 pass/fail verdicts
```

## Inputs & Assumptions
- Raw annotation JSON + images from the labeling platform (V2 schema variants supported).
- Taxonomy definitions in `data_conversion/attribute_taxonomy.json` + `hierarchical_attribute_mapping.json` (update both when adding classes/attributes).
- Geometry expected as bbox/poly/line; EXIF may be present and is handled during preprocessing.

## Processing Steps (Code Map)
- **Orchestration**: `data_conversion/pipeline/unified_processor.py`
- **Geometry pipeline**: `pipeline/coordinate_manager.py` (EXIF → rescale → smart-resize → clamp) keeps pixel↔geometry aligned.
- **Taxonomy & text**: `pipeline/flexible_taxonomy_processor.py` builds hierarchical `desc` strings; normalizes rare/long-tail labels and applies fixed‑value compression for BBU/RRU (e.g., `可见性=完整/部分`, `符合性=符合/不符合`, `挡风板需求=免装/空间充足需安装`).
- **Validation**: `pipeline/validation_manager.py` + `pipeline/constants.py` enforce geometry bounds, size thresholds, required `desc`; invalid objects/samples are reported, not silently dropped.
- **Summary builder**: `pipeline/summary_builder.py` emits a JSON-string summary with per-category stats (`dataset`, `objects_total`, `统计`, optional `异常` when non-zero) and optional `备注` (BBU, non-empty) / `分组统计` (RRU, when present). Only observed values are counted. Missing objects/empty `desc` or any invalid/unknown/conflict markers raise `ValueError` so anomalies are handled upstream. Irrelevant-image streams keep `summary: 无关图片` and do not use this builder.
- **QA artifacts**: `invalid_objects.jsonl`, `validation_results.json`, deterministic `train_tiny.jsonl` / `val_tiny.jsonl` for smoke tests.

## How to Run
Use the wrapper to keep env flags and seeds consistent:

```bash
# Example
DATASET=bbu \               # or rru
OUTPUT_DIR=data \           # root for converted artifacts
DATASET_NAME=bbu_full_768 \ # optional override
MAX_PIXELS=3500000 \        # smart-resize pixel budget
IMAGE_FACTOR=32 \           # resize stride (multiple-of)
SEED=123 \                  # deterministic splits
NUM_WORKERS=8 \             # parallel workers
bash data_conversion/convert_dataset.sh
```

Key flags in `convert_dataset.sh`:
- `MAX_PIXELS` / `IMAGE_FACTOR` — smart-resize budget and stride (applied in `coordinate_manager.py`)
- `VAL_RATIO` — train/val split fraction (defaults to 0.2)
- `STRIP_OCCLUSION`, `SANITIZE_TEXT`, `STANDARDIZE_LABEL_DESC` — text/taxonomy hygiene toggles
- `FAIL_FAST` — exit on first invalid sample (default true)
- `SEED` — deterministic split and tiny subsets

Outputs land in `${OUTPUT_ROOT}/` with `train.jsonl`, `val.jsonl`, tiny subsets, QA reports, and resized images when enabled.

Repo convention (examples checked in):
- `data_new_schema/bbu_full_1024_poly_new_schema/` and `data_new_schema/rru_full_1024_poly_new_schema/` keep full conversion artifacts, including `all_samples.jsonl`, `train/val(_tiny).jsonl`, `label_vocabulary.json`, and validation reports. These are convenient as “frozen” references when upstream annotation schema changes.

## Handoff to Training & Fusion
- **Direct training**: point `custom.train_jsonl` / `custom.val_jsonl` at the outputs (see `../training/TRAINING_PLAYBOOK.md`).
- **Fusion training**: use the converted set as target or auxiliary in `custom.fusion_config` (see `./UNIFIED_FUSION_DATASET.md`).
- **Schema alignment**: all outputs follow `./DATA_JSONL_CONTRACT.md`; geometry uses the canonical `poly`/`bbox_2d`/`line` keys expected by `src/datasets/geometry.py`.

## Quality Checklist
- Validate QA artifacts before promotion: zero invalid samples, expected class mix, polygon vertex counts within budget.
- Confirm summary JSON is parseable, single-line, and includes required keys (`dataset`, `objects_total`, `统计`), except for irrelevant-image samples which remain `summary: 无关图片`.
- Keep taxonomy files and downstream prompts in sync when introducing new object types or attributes.

## Directory Pointers
- Scripts: `data_conversion/convert_dataset.sh`
- Pipeline modules: `data_conversion/pipeline/`
- Templates & examples: `data_conversion/raw_data_template.md`
- Troubleshooting & background: `data_conversion/README.md`

This stage is optional but strongly recommended whenever upstream annotations change or new domains are onboarded.

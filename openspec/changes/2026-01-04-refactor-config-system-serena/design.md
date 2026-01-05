# Design: Modular training config hierarchy

## Goals
- Reduce duplicated YAML blocks across dense/summary/GRPO configs.
- Establish a clear hierarchy: runtime defaults → reusable components → runnable presets.
- Preserve the existing `extends` model and allow YAML anchors/merge keys.
- Keep training behavior unchanged (pure refactor).

## Proposed directory layout

```
configs/
  base.yaml                 # runtime defaults (unchanged)
  debug.yaml                # default entrypoint for scripts/train.sh
  components/
    model/
      qwen3_vl_4b.yaml
    tuner/
      dora_full.yaml
      summary_freeze.yaml
    training/
      sft_defaults.yaml
      sft_summary.yaml
      grpo_summary.yaml
    data/
      loader_defaults.yaml
    custom/
      common.yaml
      summary.yaml
      augmentation_dense.yaml
      augmentation_summary.yaml
      augmentation_grpo_summary.yaml
  fusion/
    base/
      dense.yaml
      summary.yaml
      summary_grpo.yaml
    variants/
      bbu_rru_1024.yaml
      bbu_rru_2048.yaml
  train/
    sft/
      dense_1024.yaml
      dense_2048.yaml
      summary_1024.yaml
      summary_2048.yaml
    grpo/
      summary_1024.yaml
      summary_2048.yaml
      summary_server.yaml
    stage_b/
      distill.yaml
  smoke/
    sft_dense_tiny.yaml
    sft_summary_tiny.yaml
    group_metrics.yaml
```

Notes:
- `configs/debug.yaml` stays as the default config used by `scripts/train.sh`.
- `configs/components/*` files only contain a single top-level section (e.g., `training:` or `custom:`) to keep overrides explicit.
- `configs/train/*` are runnable presets composed via `extends` lists in deterministic order.

## Legacy preset mapping (for reference)
- `configs/fusion_train/sft_base.yaml` → `configs/train/sft/dense_base.yaml`
- `configs/fusion_train/bbu_rru_dense_new_schema_1024.yaml` → `configs/train/sft/dense_1024.yaml`
- `configs/fusion_train/bbu_rru_dense_new_schema_2048.yaml` → `configs/train/sft/dense_2048.yaml`
- `configs/fusion_train/bbu_rru_summary_new_schema_1024.yaml` → `configs/train/sft/summary_1024.yaml`
- `configs/fusion_train/bbu_rru_summary_new_schema_2048.yaml` → `configs/train/sft/summary_2048.yaml`
- `configs/grpo/summary_grpo_base.yaml` → `configs/train/grpo/summary_base.yaml`
- `configs/grpo/summary_grpo_server.yaml` → `configs/train/grpo/summary_server.yaml`

## Composition rules

### Training configs
- Presets use `extends` lists in order: `configs/base.yaml` → component blocks → mode overlay → experiment overrides.
- Component YAMLs are re-used across SFT/summary/GRPO; each preset only overrides deltas such as `custom.fusion_config`, `training.output_dir`, and `training.run_name`.
- YAML anchors/merge keys are allowed within a file for readability, but cross-file reuse is done through `extends`.

### Fusion configs
- Fusion configs accept `extends` (similar to `ConfigLoader`).
- Merge behavior is name-based for `targets` and `sources`:
  - Entries match by `name` (fallback to `dataset` if `name` missing).
  - Matching entries are deep-merged (override replaces scalar values; nested mappings merge).
  - Entries that exist only in the overlay are appended after base entries.
  - An explicit empty list in the override replaces the base list (e.g., GRPO sources = []).
- This enables a base file with shared dataset entries and small overlays for 1024/2048 paths.

## Example composition

### Training preset
```
# configs/train/sft/dense_1024.yaml
extends:
  - ../../base.yaml
  - ../../components/model/qwen3_vl_4b.yaml
  - ../../components/tuner/dora_full.yaml
  - ../../components/training/sft_defaults.yaml
  - ../../components/data/loader_defaults.yaml
  - ../../components/custom/common.yaml
  - ../../components/custom/augmentation_dense.yaml

custom:
  fusion_config: configs/fusion/variants/bbu_rru_1024.yaml
  use_summary: false
```

### Fusion overlay
```
# configs/fusion/variants/bbu_rru_1024.yaml
extends: ../base/dense.yaml

targets:
  - name: bbu_dense
    train_jsonl: data_new_schema/bbu_full_1024/train.jsonl
    val_jsonl: data_new_schema/bbu_full_1024/val.jsonl
  - name: rru_dense
    train_jsonl: data_new_schema/rru_full_1024_poly/train.jsonl
    val_jsonl: data_new_schema/rru_full_1024_poly/val.jsonl
```

## Validation impact
- `scripts/validate_sft_config.py` will continue to resolve `extends` for training configs and should additionally validate `custom.fusion_config` files using the fusion loader (now extend-aware).
- Validation should fail fast on unknown dataset names when overlay entries do not match a base entry.

## Documentation updates
- Update the config catalog in `docs/training/TRAINING_PLAYBOOK.md` to list new paths.
- Update `docs/training/REFERENCE.md` and `scripts/README.md` to reflect the new hierarchy.
- Update `docs/README.md` directory map if any top-level config paths change.

# Tasks

- [x] Implement mandatory config inspection/diff tooling as `scripts/config_tools/inspect_config.py` (inspect resolved config + extends chain; diff two resolved configs).
- [x] Document the canonical inspect/diff commands and expected outputs in the change design (including exit codes for parity gating).
- [x] Define a repeatable parity workflow and the explicit allowlist/denylist for parity diffs (see design.md); treat parity as a migration gate.
- [x] Inventory which `configs/train/**` presets are actively used (SFT/GRPO/stage_b/smoke/debug) and prioritize the “must keep working” set.
- [x] Create/maintain a parity mapping table (old preset → new preset) and record parity status for each migrated preset.
- [x] Inline key defaults into runnable presets under `configs/train/**` and represent coarse augmentation variants as small overlays (2–3 overlays per base preset) to keep tuning discoverable without a separate `configs/types/` layer.
- [x] Migrate one dense SFT preset and validate via `scripts/validate_sft_config.py`; run parity diff and record result.
- [x] Migrate one GRPO preset (dense or summary) and validate via `scripts/validate_sft_config.py`; run parity diff and record result.
- [x] Migrate remaining experiments under `configs/train/**` and smoke presets under `configs/smoke/**`; run parity diff per preset and record results.
- [x] Update docs config catalog and “how to modify configs” guidance (`docs/training/TRAINING_PLAYBOOK.md`, `docs/training/REFERENCE.md`, `scripts/README.md`, `docs/README.md` as needed).
- [x] Remove legacy `configs/components/**` and any now-unused base presets after parity is confirmed.
- [x] Run `openspec validate 2026-01-11-refactor-config-hierarchy-shallow --strict` and ensure the change passes.

## Active presets (must keep working)

These are the in-tree runnable presets referenced by scripts/docs/tests and treated as “must keep working” during the migration.

- Debug / quick validation:
  - `configs/debug.yaml`
- SFT presets:
  - `configs/train/sft/dense_1024.yaml`
  - `configs/train/sft/dense_2048.yaml`
  - `configs/train/sft/summary_1024.yaml`
  - `configs/train/sft/summary_2048.yaml`
- GRPO presets:
  - `configs/train/grpo/dense_2048.yaml`
  - `configs/train/grpo/dense_2048_lowaug_t04_low_lr.yaml`
  - `configs/train/grpo/summary_1024.yaml`
  - `configs/train/grpo/summary_2048.yaml`
  - `configs/train/grpo/summary_server.yaml`
- Stage-B distillation (training, not Stage-B inference):
  - `configs/train/stage_b/distill.yaml`
- Smoke presets (training config smoke runs):
  - `configs/smoke/group_metrics.yaml`
  - `configs/smoke/grpo_dense.yaml`
  - `configs/smoke/grpo_dense_tiny.yaml`
  - `configs/smoke/sft_dense_tiny.yaml`
  - `configs/smoke/sft_summary_tiny.yaml`

## Parity mapping table

All parity diffs are stored under `analysis/config_parity/` and were generated via:

```bash
conda run -n ms python scripts/config_tools/inspect_config.py diff --left <old.yaml> --right <new.yaml> --profile parity > analysis/config_parity/<name>.parity.txt
```

| Preset | Old preset (pre-refactor) | New preset (post-refactor) | Parity | Artifact |
|---|---|---|---|---|
| `configs/debug.yaml` | `configs/debug.yaml` (components-based) | `configs/debug.yaml` (train-only, no types) | ✅ (`total_diffs: 0`) | `analysis/config_parity/debug.parity.txt` |
| `configs/train/sft/dense_1024.yaml` | `configs/train/sft/dense_1024.yaml` (components-based) | `configs/train/sft/dense_1024.yaml` (train-only, no types) | ✅ (`total_diffs: 0`) | `analysis/config_parity/dense_1024.parity.txt` |
| `configs/train/sft/dense_2048.yaml` | `configs/train/sft/dense_2048.yaml` (components-based) | `configs/train/sft/dense_2048.yaml` (train-only, no types) | ✅ (`total_diffs: 0`) | `analysis/config_parity/dense_2048.parity.txt` |
| `configs/train/sft/summary_1024.yaml` | `configs/train/sft/summary_1024.yaml` (components-based) | `configs/train/sft/summary_1024.yaml` (train-only, no types) | ✅ (`total_diffs: 0`) | `analysis/config_parity/sft_summary_1024.parity.txt` |
| `configs/train/sft/summary_2048.yaml` | `configs/train/sft/summary_2048.yaml` (components-based) | `configs/train/sft/summary_2048.yaml` (train-only, no types) | ✅ (`total_diffs: 0`) | `analysis/config_parity/sft_summary_2048.parity.txt` |
| `configs/train/grpo/dense_2048.yaml` | `configs/train/grpo/dense_2048.yaml` (components-based) | `configs/train/grpo/dense_2048.yaml` (train-only, no types) | ✅ (`total_diffs: 0`) | `analysis/config_parity/grpo_dense_2048.parity.txt` |
| `configs/train/grpo/dense_2048_lowaug_t04_low_lr.yaml` | `configs/train/grpo/dense_2048_lowaug_t04_low_lr.yaml` (components-based) | `configs/train/grpo/dense_2048_lowaug_t04_low_lr.yaml` (train-only, no types) | ✅ (`total_diffs: 3`, all allowed) | `analysis/config_parity/grpo_dense_2048_lowaug_t04_low_lr.parity.txt` |
| `configs/train/grpo/summary_1024.yaml` | `configs/train/grpo/summary_1024.yaml` (components-based) | `configs/train/grpo/summary_1024.yaml` (train-only, no types) | ✅ (`total_diffs: 0`) | `analysis/config_parity/grpo_summary_1024.parity.txt` |
| `configs/train/grpo/summary_2048.yaml` | `configs/train/grpo/summary_2048.yaml` (components-based) | `configs/train/grpo/summary_2048.yaml` (train-only, no types) | ✅ (`total_diffs: 0`) | `analysis/config_parity/grpo_summary_2048.parity.txt` |
| `configs/train/grpo/summary_server.yaml` | `configs/train/grpo/summary_server.yaml` (components-based) | `configs/train/grpo/summary_server.yaml` (train-only, no types) | ✅ (`total_diffs: 0`) | `analysis/config_parity/grpo_summary_server.parity.txt` |
| `configs/train/stage_b/distill.yaml` | `configs/train/stage_b/distill.yaml` (components-based) | `configs/train/stage_b/distill.yaml` (train-only, no types) | ✅ (`total_diffs: 0`) | `analysis/config_parity/stage_b_distill.parity.txt` |
| `configs/smoke/group_metrics.yaml` | `configs/smoke/group_metrics.yaml` (components-based) | `configs/smoke/group_metrics.yaml` (train-only, no types) | ✅ (`total_diffs: 0`) | `analysis/config_parity/smoke_group_metrics.parity.txt` |
| `configs/smoke/grpo_dense.yaml` | `configs/smoke/grpo_dense.yaml` (components-based) | `configs/smoke/grpo_dense.yaml` (train-only, no types) | ✅ (`total_diffs: 0`) | `analysis/config_parity/smoke_grpo_dense.parity.txt` |
| `configs/smoke/grpo_dense_tiny.yaml` | `configs/smoke/grpo_dense_tiny.yaml` (components-based) | `configs/smoke/grpo_dense_tiny.yaml` (train-only, no types) | ✅ (`total_diffs: 0`) | `analysis/config_parity/smoke_grpo_dense_tiny.parity.txt` |
| `configs/smoke/sft_dense_tiny.yaml` | `configs/smoke/sft_dense_tiny.yaml` (components-based) | `configs/smoke/sft_dense_tiny.yaml` (train-only, no types) | ✅ (`total_diffs: 0`) | `analysis/config_parity/smoke_sft_dense_tiny.parity.txt` |
| `configs/smoke/sft_summary_tiny.yaml` | `configs/smoke/sft_summary_tiny.yaml` (components-based) | `configs/smoke/sft_summary_tiny.yaml` (train-only, no types) | ✅ (`total_diffs: 0`) | `analysis/config_parity/smoke_sft_summary_tiny.parity.txt` |

## Parity checklist (repeat for every migrated preset)

- [x] Identify old preset path (pre-migration) and new preset path (post-migration).
- [x] Inspect old resolved config:
  - `conda run -n ms python scripts/config_tools/inspect_config.py inspect --config <old.yaml> > <artifact_dir>/old.inspect.txt`
- [x] Inspect new resolved config:
  - `conda run -n ms python scripts/config_tools/inspect_config.py inspect --config <new.yaml> > <artifact_dir>/new.inspect.txt`
- [x] Diff old vs new under parity profile:
  - `conda run -n ms python scripts/config_tools/inspect_config.py diff --left <old.yaml> --right <new.yaml> --profile parity > <artifact_dir>/parity.diff.txt`
- [x] Confirm the diff contains **only allowed diffs** (run identity + logging/telemetry + save/eval schedule) and **no forbidden diffs** (training semantics).
- [x] Record parity status in the mapping table (✅ pass / ❌ fail) and attach the diff artifact path.

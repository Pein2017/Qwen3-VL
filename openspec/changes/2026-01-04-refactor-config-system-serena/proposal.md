# Proposal: Refactor training config hierarchy for modular reuse

## Why
- Current YAML configs under `configs/fusion_train/` and `configs/dataset_mix/` repeat large sections for model, tuner, training, augmentation, and fusion settings.
- Duplication increases maintenance risk (inconsistent overrides across dense/summary/GRPO variants) and obscures which values are defaults versus experiment-specific deltas.
- The hierarchy is unclear, which slows addition of new modes (SFT vs GRPO, dense vs summary) and new dataset variants.

## What
- Introduce a modular config hierarchy under `configs/components/` for reusable blocks (model, tuner, training, data loader, custom defaults, augmentation, rlhf, deepspeed).
- Move runnable presets to `configs/train/<mode>/` and keep `configs/debug.yaml` as the default entrypoint.
- Reorganize fusion dataset mix configs under `configs/fusion/` with base definitions and per-resolution overlays.
- Add inheritance support to fusion configs (`extends`/`inherit`) with name-based merging for `targets` and `sources` to reduce duplication across 1024/2048 variants.
- Update `scripts/validate_sft_config.py` and documentation to reflect the new layout.

## Scope
- YAML configs in `configs/` (training presets, fusion configs, debug/smoke).
- Config loading/merging for training configs and fusion configs.
- Validation script and schema checks for new layout.
- Documentation updates in `docs/training/*` and `scripts/README.md`.

## Non-goals
- No changes to training behavior, model architecture, or dataset semantics.
- No changes to `scripts/train.sh` CLI surface.
- No backward compatibility for old config paths or file names.

## Impact / Breaking changes
- Existing config file paths and names will change; old YAMLs will be removed or replaced.
- Fusion configs gain `extends`/`inherit` with name-based merge semantics for targets/sources.

## Success criteria
- Runnable presets under `configs/train/` are composed from components with minimal overrides and show >50% reduction in repeated sections.
- All training configs validate via `scripts/validate_sft_config.py` (SFT, GRPO, dense, summary, smoke/debug).
- `scripts/train.sh` continues to work without modification and resolves `configs/debug.yaml` by default.
- Documentation reflects the new hierarchy and catalog.

## Risks
- Name-based merge for fusion configs could hide typos in dataset names; validation must fail fast on unknown names.
- Large-scale path changes risk stale doc references; docs map requires a full audit update.

## Rollout plan (high-level)
1. Add OpenSpec deltas and finalize hierarchy design.
2. Implement fusion-config inheritance and create new component/preset YAMLs.
3. Update validation and docs.
4. Validate configs and remove legacy YAMLs.

# Proposal: Configure LM-Head KD Weight

## Problem Statement
- Stage-1/3 GKD runs always include the LM-head JSD loss with an implicit weight of `1.0`. Tuning or disabling the term requires code edits and recompiling configs.
- Feature-level (aligner/visual) KD is already optional via `custom.visual_kd`, but there is no symmetrical control for the logits-level distillation; users cannot keep visual KD on while zeroing out LM KD.
- Trainers log the unweighted LM KD term, so telemetry does not reflect how heavily it influences the total loss once we apply custom scaling.

## Desired Outcomes
- Add a YAML knob (`rlhf.llm_kd_weight`) that scales or disables the LM-head distillation without touching code.
- Preserve feature-level KD so users can mix-and-match LM KD and visual KD, including running visual-only distillation with LM KD disabled.
- Telemetry and aggregated metrics report the weighted LM KD term so dashboards reflect the actual loss composition.
- Backwards compatibility: default configs (no new field) behave exactly as today.

## Scope
- `configs/` YAML schema (`rlhf` section) and config loading helpers.
- `src/trainers/gkd_monitor.py` loss composition + metrics aggregation.
- Unit tests covering config parsing and trainer behavior.
- Docs/config comments that describe the new knob.

## Non-Goals
- No changes to ms-swift core trainer APIs beyond attaching attributes that already exist on our wrapper.
- No rework of beta scheduling or teacher forward logic beyond gating on the new weight.
- Do not alter dataset pipelines, augmentation, or Stage-B flows.

## Proposed Approach (high level)
1. **Config & Loader**: Allow `rlhf.llm_kd_weight` (float ≥ 0). Default to `1.0` when omitted, validate in `ConfigLoader`, and surface the value on both `TrainArguments` and nested `training_args`.
2. **Trainer behavior**: Read the weight at init time, short-circuit the LM KD branch when the weight is `0`, and multiply the JSD loss by the configured weight before contributing to `total_loss`.
3. **Telemetry**: When the weight is positive, push the weighted value into `_metrics[mode]["llm_kd_loss"]`; omit the metric entirely when the weight is `0`.
4. **Compatibility**: Ensure `visual_kd` hooks still register and execute even if LM KD is disabled so feature-level distillation can run on its own.
5. **Validation & Docs**: Extend `tests/test_gkd_monitor_integration.py` (and related fixtures) for coverage, document the new knob in Stage configs and relevant reference docs.

## Validation Plan
- Unit/integration tests:
  - config loader returns `llm_kd_weight` with default, positive, and invalid (negative) values.
  - trainer loss/metering covers weight > 0 and weight == 0 cases, plus coexistence with visual KD.
- Manual sanity: run `pytest tests/test_gkd_monitor_integration.py -k llm_kd` after implementation.
- Configuration diff: add comments/examples in Stage configs demonstrating how to disable LM KD while keeping visual KD enabled.



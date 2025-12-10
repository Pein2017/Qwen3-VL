# Proposal: add-fusion-per-dataset-mode

## Problem
- `custom.use_summary` is a global toggle in the SFT fusion pipeline. When true, every fused sample is forced into summary mode, so dense-only auxiliary datasets (no `summary` field) fail validation. When false, summary-oriented targets lose their summary prompts/validation.
- Templates/prompts are switched once per run, preventing per-dataset prompt selection that matches each dataset’s mode.
- Telemetry currently only records a single mode flag, masking mixed-mode debugging.

## Goals
- Allow each fusion dataset (target or source) to declare its own mode (`dense` | `summary`), defaulting to the legacy global flag when unspecified for backward compatibility.
- Route prompts/templates per dataset according to its mode without unsafe template cloning or cross-sample leakage.
- Keep strict validation: summary mode requires non-empty `summary`; dense mode requires objects/geometry.
- Preserve telemetry/debug visibility per sample (mode, prompt source, validation paths).

## Non-Goals
- No changes to Stage-B or non-fusion single-dataset training paths.
- No smart auto-conversion between dense and summary; datasets must declare the intended mode.

## Impact
- Fusion configs can mix summary targets with dense auxiliary sources without schema errors.
- Debug logs and epoch plans surface per-dataset mode/prompt information.

## Rollout & Compatibility
- Legacy configs that only set `custom.use_summary` continue to work: the flag becomes the default mode for datasets lacking an explicit mode.
- New fusion config field `mode` (or `use_summary` alias) is opt-in per dataset.
- Documentation and examples will be updated; add focused unit tests for mixed-mode fusion.

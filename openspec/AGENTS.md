# OpenSpec Agent Guide

This playbook is for feature and capability work. Use it whenever a task involves new behavior, architecture shifts, or coordinated changes across configs/code/specs.

## When to Engage the Spec Workflow
- The request adds or retires user-facing capability, modifies data contracts, or adjusts training/inference behavior beyond a straight bug fix.
- Work impacts multiple surfaces (e.g., YAML config + dataset builder + docs) or requires coordination across stages.
- Ambiguity exists and clarifying questions do not fully resolve scope.

Skip formal proposals only for self-contained fixes that restore intended behavior, comment/doc touch-ups, or mechanical config edits.

## Ramp-Up Checklist (30–60 min)
1. **Read the global guide** in `/data/Qwen3-VL/AGENTS.md` for high-level context.
2. **Targeted docs**:
   - `docs/README.md` and `docs/DATA_AND_DATASETS.md` for pipelines & JSONL contract.
   - `docs/AUGMENTATION.md` plus `vis_tools/README_CROP_VIS.md` when touching geometry or crops.
   - `docs/experiments/` for precedent runs and config patterns.
3. **Code reconnaissance**:
   - `src/sft.py` for launch flow and how YAML is consumed.
   - `src/datasets/` subpackages that map to the planned change (preprocessors, augmentation, geometry, builders, collators).
   - `src/config/` for prompt schemes, TrainArguments helpers, and config loaders.
4. **Config survey**: open `configs/base.yaml` plus the stage or summary config closest to the requested scenario.
5. **Scripts & tooling**: note entrypoints in `scripts/` and visual aids in `vis_tools/` that will help validate the change.

Document findings in the proposal so reviewers can follow the thread quickly.

## Proposal & Task Structure
1. `openspec init` is already set up; create change scaffolding under `openspec/changes/<change-id>/` with a verb-led kebab case name.
2. Provide in `proposal.md`:
   - Problem statement and desired behavior.
   - Affected surfaces (configs, datasets, augmentation, prompts, docs, scripts).
   - Validation plan (tests, visualization, training smoke).
3. Enumerate implementation steps in `tasks.md` and keep them updated (`- [ ]` → `- [x]`).
4. For capability specs, add or modify `specs/<capability>/spec.md` deltas using OpenSpec formatting (`## ADDED|MODIFIED Requirements` + scenarios).
5. Run `openspec validate <change-id> --strict` before requesting review.

## Implementation Patterns by Surface
- **Configs (`configs/`)**: use inheritance; add overrides in the narrowest stage config. Ensure new keys are consumed in `src/sft.py` or relevant helper.
- **Augmentation / geometry**: centralize math in `src/datasets/geometry.py` and register operators via `src/datasets/augmentation/registry.py`. Mirror changes in docs + visual tools.
- **Preprocessors & builders**: extend `src/datasets/preprocessors/base.py` or `builders/base.py`; update tests under `tests/` or add new ones mirroring dataset flows.
- **Prompts**: update `src/config/prompts.py` and reflect scheme changes in YAML (`prompts.scheme`, `custom.user_prompt`).
- **Scripts**: shell helpers belong in `scripts/`; ensure they are runnable via `conda run -n ms`.
- **Docs**: keep `docs/` synchronized with any behavioral change. Summarize new knobs in the relevant README.

## Validation Expectations
- Unit or integration tests covering new behavior (augment ops, dataset builders, config loaders).
- Visualization or telemetry when touching geometry or crops (use `vis_tools/vis_augment_compare.py`).
- Config diffs: demonstrate resulting command (train script) and highlight expected artifacts.
- Training smoke test plan: note dataset slice, config, and metrics to observe (even if not executed yet).

## Command Quick Reference
```bash
openspec list                # Active changes
openspec spec list --long     # Existing capabilities
openspec show [item]          # Inspect change/spec
openspec diff [change-id]     # Delta preview
openspec validate [item] --strict
openspec archive [change-id] --yes
```
Use `rg -n "Requirement:" openspec/specs` for fast spec discovery.

## Review & Handoff
- Before requesting approval: proposal updated, tasks checked off, validation complete, docs/configs/tests touched as promised.
- After merge/deploy: move change under `openspec/changes/archive/` (or run `openspec archive`) and roll permanent spec updates into `openspec/specs/` if needed.
- Keep CHANGELOG entries and example configs in sync with shipped behavior.

Stay disciplined about tracing feature work from proposal → implementation → validation → documentation; it keeps this repo approachable for the next person.

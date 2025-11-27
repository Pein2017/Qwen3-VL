# Proposal: Augmentation Curriculum Scheduling for Dense Captioning

## Problem Statement
Dense captioning trains with a single static augmentation recipe (bypass probability + per-op probabilities/ranges). Users want to ramp augmentation strength smoothly over time so early steps see cleaner data and later phases see stronger perturbations. The current pipeline lacks a scheduling hook, and multi-worker dataloaders complicate consistent updates across processes.

## Goals
- Add a config-driven augmentation curriculum keyed on **training progress** (global-step percentage milestones, optionally absolute steps) with default linear ramps between boundaries.
- Adjust `bypass_prob` and numeric op parameters (`prob` and numeric ranges like `max_deg`, `[lo, hi]`, scale ranges) over time without rebuilding datasets.
- Ensure schedules apply consistently across distributed ranks and dataloader workers with strict upfront validation.

## Non-Goals
- No new augmentation operators.
- No change to eval/inference preprocessing.
- No overhaul of existing op semantics; only scheduling their parameters.

## Proposed Changes
1. **Config Surface**: Add `custom.augmentation.curriculum` with ordered phases keyed by `until_percent` (preferred, 0–1 or 0–100), with optional `until_step` fallback. Each phase may set target `bypass_prob` and per-op numeric overrides. Between boundaries, values linearly interpolate from the previous phase target to the current target; piecewise-constant phases are supported by repeating targets.
2. **Scheduler**: Implement a progress-driven scheduler that returns effective `bypass_prob` and op params for a given global step. It linearly interpolates numeric fields, leaves non-numeric fields untouched, and holds final values after the last phase. For percentage configs, resolve boundaries using trainer-reported total steps on startup.
3. **Propagation**: Add a TrainerCallback that updates a shared, read-only state (visible to all ranks/workers) each step/phase so workers apply identical effective params without rebuilding the dataloader.
4. **Validation & Safety**: Validate on startup (before training): monotonic boundaries, known op names/fields, non-negative probs, ascending ranges, and matching base ops. Invalid configs fail fast. Runtime logging stays minimal (no extra verbosity by default).

## Validation Plan
- Unit test the scheduler: phase selection by step, linear interpolation of `prob` and numeric ranges, boundary behavior, and invalid-config rejection.
- Smoke test a short multi-worker run with 2 phases to confirm consistent effective params across workers (via shared state introspection) and absence of additional logging.
- Confirm eval path unchanged and existing augmentation geometry tests still pass (reuse `pytest tests/augmentation`).

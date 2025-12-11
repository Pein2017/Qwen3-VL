# Proposal: Refactor augmentation ops into unified base operators

## Problem
- Current augmentation ops are numerous and partially duplicated: affine/color logic is repeated per op; crop/copy-paste semantics are bespoke per operator.
- Harder to add “object-centric” augmentations (multi-object copy‑paste, line paste) without copying boilerplate for geometry transforms, validation, and telemetry.
- Curriculum control has to reason about every op’s bespoke params, making schedules brittle.

## Goal
- Introduce three base operator types—AffineOp, ColorOp, PatchOp—that encapsulate shared pipelines (matrix sampling, per-image color map, patch selection/placement).
- Refactor existing ops to the bases without changing visible behavior or YAML surface.
- Standardize PatchOps so crop and copy‑paste share lifecycle hooks (select → transform → place), keeping geometry/telemetry consistent.
- Keep curriculum control focused on a small set of parameters (prob + numeric ranges) per op type.
- Ensure that, after augmentation, dense-caption objects are re-sorted into the same **top-to-bottom, left-to-right (TL→BR)** order described by the prompts and data conversion docs, so training data and model expectations stay aligned.

## Scope
- Full refactor plus tests for the augmentation pipeline across the codebase (ops, Compose, builder, curriculum, preprocessors, visualization, and tests).
- Cover affine/color/patch ops, Compose interaction, telemetry, curriculum exposure, and related tooling.
- Stay backward compatible with existing YAML schemas and observable augmentation effects for existing configs.

## Non-Goals
- Do not change dataset schemas or training configs beyond internal augmentation wiring.
- New augmentation ops (e.g., multi-object or line PatchOps) may be added, but MUST NOT change behavior of existing configs unless covered by separate specs.

# Proposal: Deprecate `quad` geometry key in favor of canonical `poly`

## Why
- `quad` is just a special-case polygon; keeping two keys causes validator/augmentation divergence and user confusion.
- Recent pipelines already prefer `poly`; maintaining `quad` paths increases test surface without benefit.
- Removing the `quad` key simplifies data contracts and avoids silent drops when mixed geometries appear.

## What
- Treat `poly` as the sole polygon key; reject `quad` in configs/data/builders/validators.
- Update geometry utilities, visualization, and tests to operate on `poly` only.
- Clarify docs/specs to state accepted geometries are `bbox_2d`, `poly`, and `line`.

## Scope
- Specs: data-augmentation capability.
- Code: geometry helpers, dataset utilities, visualization helpers.
- Tests/docs/config comments referencing `quad`.

## Non-Goals
- Changing polygon math/algorithms.
- Adding new geometry types or normalizing lines/bboxes differently.

## Risks / Mitigations
- **Legacy data with `quad`** will now fail fast; mitigation: explicit error message to convert to `poly`.
- **Doc drift**: update key docs + OpenSpec delta to avoid mismatch.

---
title: Retire Dynamic Pairing and 图片_{n} Shorthand
author: Qwen3-VL
created: 2025-11-01
change-id: 2025-11-01-retire-dynamic-pairing-shorthand
status: draft
---

## Why

- **Generalization regressions**: SFT runs that enabled dynamic pairing (`images_per_user_turn > 1`) show higher hallucination rates and weaker localization on unseen evaluation sets. Qualitative reviews point to diluted attention on the primary image because turns fuse multiple scenes.
- **Token efficiency**: The current chat template wraps every assistant reply inside a top-level `{"图片_{n}": {...}}` object. With single-image training this wrapper is redundant and costs ~12 extra tokens per sample, slowing convergence for minimal benefit.
- **Consistency with downstream inference**: Serving stacks expect per-image prompts to map to exactly one response payload. Keeping `object_{n}` and `line_points` indexing while removing dynamic grouping simplifies adapters and avoids special cases for non-paired runs.

## What Changes

- **Remove dynamic pairing support**
  - Delete `DynamicPairDataset` and the `DynamicPairingConfig` shim; collapse loaders onto a single-image dataset that still applies augmentation/preprocessing.
  - Drop `images_per_user_turn` from the schema, configs, and docs. Config validation must fail fast if the key is supplied.
- **Reshape JSON builder output**
  - Update `JSONLinesBuilder` (dense + summary modes) so the assistant emits bare object hierarchies: `{ "object_1": {...}, ... }` without a `图片_{n}` container.
  - Retain object indices (`object_{n}`) and geometry metadata (`line_points`, `line` coordinates, etc.) to preserve determinism and stop infinite continuation loops.
  - Ensure summary mode returns a plain string (single image) while exposing the same minimal structure.
- **Template & prompt alignment**
  - Refresh prompts/system instructions to reference the new structure (no `图片_{n}` tokens) and emphasize the expected JSON schema.
- **Docs & tooling**
  - Update `docs/training/REFERENCE.md`, `docs/DATA_AND_DATASETS.md`, and `src/README.md` to describe the simplified format.
  - Adjust visualization or demo scripts that parse `图片_{n}`.
- **Testing & telemetry**
  - Expand dataset/unit tests to ensure configs reject pairing and that builders emit the minimal structure.

## Impact

- Surfaces: `src/datasets/dynamic_pair.py`, `src/datasets/dense_caption.py`, `src/datasets/builders/jsonlines.py`, `src/config/schema.py`, stage configs under `configs/`, prompt helpers, docs, unit tests.
- Downstream: Stage-A/B pipelines and demo tooling consuming grouped outputs must adapt to the new assistant schema.

## Validation Plan

1. **Unit**: Add/refresh tests covering config validation (reject `images_per_user_turn`), dataset length/ordering, and builder payload shape without `图片_{n}`.
2. **Integration**: Run a smoke SFT training pass on a stage-3 config to confirm dataloaders, augmentation, and logging succeed without dynamic pairing.
3. **Evaluation**: Re-run the previously regressed evaluation set to document improved grounding; track hallucination counts and caption BLEU/Recall deltas.
4. **Docs**: Double-check updated guides render correctly and demo/visualization scripts operate with the new schema.

## Open Questions

- Should summary mode continue to emit JSON (e.g., `{ "summary": "..." }`) or return plain text? Default assumption: emit plain string when in summary mode, but confirm with consumers.
- Do any Stage-A/B workflows rely on multi-image batches for throughput rather than pairing semantics? Need confirmation before deleting the pairing hooks outright.


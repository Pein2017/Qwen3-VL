# Change Proposal: update-geometry-poly-fusion

## Summary

Unify all detection-style geometries across Qwen3-VL and external datasets under a single polygon field `poly` (alongside `bbox_2d` and `line`), and introduce a modular multi-dataset fusion pipeline so BBU dense-caption training can be refined with additional sources (e.g., COCO/LVIS) without entangling dataset-specific logic in the core dataset classes.

This proposal formalizes the recent rename from `quad` → `poly` in `src/` and extends the spec surface to cover:
- Canonical geometry contract (`bbox_2d` / `poly` / `line`) in pixel space.
- Augmentation expectations for polygons under affine/crop ops after the rename.
- An offline fusion pipeline that mixes BBU and auxiliary datasets into a single training JSONL with per-source augmentation control.

## Motivation

- The previous `quad` naming implied a fixed 4-point rectangle, while the implementation is now generalized to arbitrary polygons with even-length coordinate lists. Using `poly` aligns the name with behavior and simplifies future ingestion of polygon-rich datasets.
- BBU dense-caption training benefits from a small fraction of general-domain detection data (e.g., COCO/LVIS) to improve robustness and reduce collapse, but the pipeline currently assumes a single `train_jsonl`. We need a spec’ed way to fuse multiple sources while keeping evaluation focused on the BBU target domain.
- `/data/public_data` already contains LVIS/COCO converters; aligning them with the canonical `poly` contract and a clean fusion interface avoids one-off hacks inside `DenseCaptionDataset` or `AugmentationPreprocessor`.

## Goals

1. **Geometry contract**
   - Replace `quad` with `poly` as the canonical polygon field in specs and implementation, keeping all geometry in pixel space before encoding.
   - Ensure augmentation specs and helpers treat `poly` as the generic polygon type (4-point and N-point) and preserve the single-geometry-per-object invariant.

2. **Multi-dataset fusion**
   - Define a fusion capability that takes multiple canonical JSONL datasets (BBU + auxiliary sources) and produces a single fused training JSONL according to configurable weights.
   - Require per-record provenance (e.g., `metadata.dataset`) so augmentation and analysis can be source-aware.
   - Allow per-source augmentation toggles (e.g., enable augment for BBU, disable for LVIS) without modifying `DenseCaptionDataset`.

3. **Evaluation focus**
   - Keep evaluation on the BBU target domain by default; auxiliary sources are used for training only unless explicitly enabled for analysis.

## Non-Goals

- Changing the high-level JSONL structure beyond geometry field names: `images` / `objects` / `width` / `height` remain as in existing docs.
- Redesigning Stage-A/Stage-B pipelines; this proposal is scoped to dense-caption training data and augmentation.
- Introducing new geometry types beyond `bbox_2d`, `poly`, and `line`.

## Affected Areas

- Specs:
  - `openspec/specs/data-augmentation/spec.md` (geometry terminology and polygon behavior).
  - New capability spec: `multi-dataset-fusion` (training data mixing and source-aware augmentation).
- Code:
  - `src/datasets/geometry.py`, `src/datasets/utils.py`, `src/datasets/augmentation/ops.py`, `src/datasets/preprocessors/augmentation.py`, `src/datasets/builders/jsonlines.py` (already migrated in code; spec needs to catch up).
  - `/data/public_data` converters and validators (LVIS/COCO) to emit/accept `poly` instead of `quad`, with an optional "large polygon → bbox_2d" fallback.
- Docs:
  - `docs/DATA_AND_DATASETS.md`, `docs/DATA_AUGMENTATION.md`, `POLYGON_SUPPORT.md` (reflect `poly` and the fusion story).


# Proposal: add-detection-smart-resize

## Summary
- Introduce a unified detection preprocessor that applies the same smart-resize policy (max_pixels + grid factor) across target and source datasets before entering fusion/training.
- Reuse the existing resize logic from `data_conversion` so converters and the online fusion loader share one implementation, rescaling geometries and image sizes consistently for Qwen-VL patchifying.

## Motivation
- Fused training OOMs when source images (e.g., LVIS) bypass smart resize and carry large patch counts.
- Today only BBU conversions reliably enforce `max_pixels`/`image_factor`; public datasets may skip it, leading to inconsistent geometry and memory blow-ups.
- A single preprocessor contract will keep all detection JSONLs patch-aligned (28/32 grid), bounded by a configurable pixel budget, and ready for augmentation/inference without per-dataset hacks.

## Scope
- Spec and design for a shared “detection smart-resize” preprocessor.
- Requirements for geometry rescaling, path handling, configurability, and idempotency.
- Hooks for both offline converters and the online fusion loader to invoke the preprocessor.

## Non-Goals
- Changing augmentation semantics or loss functions.
- Redesigning the existing `smart_resize` algorithm beyond making it reusable.

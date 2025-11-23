# Design: add-detection-smart-resize

## Overview
Unify the “smart resize” logic used by BBU conversions into a reusable detection preprocessor that any dataset (target or source) can invoke before fusion/training. This preprocessor enforces:
- A pixel budget (`max_pixels`) to bound patch tokens and memory.
- Dimension snapping to an `image_factor` grid (28 or 32) for Qwen-VL patchifying.
- Geometry rescaling (bbox/poly/line) and updated `width`/`height` to match resized images.
- Image path normalization (relative to JSONL, resolved to absolute at load time).

## Implementation Direction (refined)
- **Single shared module (source of truth)**: `src/datasets/preprocessors/resize.py` owns smart-resize for all domains. `data_conversion/pipeline/vision_process.py` and `data_conversion/resize_dataset.py` delegate to it; public converters call it via the shared wrapper.
- **Converter contract**: Each dataset implements only “cook annotations → canonical JSONL” (no resize). Schema: `images`, `objects` (exactly one geometry: `bbox_2d` | `poly`+`poly_points` | `line`), `width`, `height`, `desc`; paths relative to JSONL by default. New datasets customize only this step.
- **Shared offline invocation**: Standard flags (`--smart-resize`, `--max_pixels`, `--image_factor`, optional `--min_pixels`) apply to any dataset; outputs co-locate resized images with JSONL under the chosen root, keeping JSONL-relative paths.
- **Generic runner**: Provide a pluggable runner (converter module/name + shared resize flags) so new datasets get the same flow without forking code.
- **Unified fusion policy**: No online guard; unified fusion requires inputs to be pre-resized offline using the shared module.
- **Telemetry**: Log scale factors; warn on large shrink (>2×) with remediation hints.

## Open Questions
- Where to store resized images for public datasets by default (proposed: `public_data/<ds>/resized/` with JSONL next to it).
- Whether to force `max_pixels` parity with BBU defaults (`786432` or `921600`) vs. keeping it a config knob per dataset.

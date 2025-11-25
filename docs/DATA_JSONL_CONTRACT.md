# Data JSONL Contract (Global)

This document defines the universal JSONL format consumed by all training/eval datasets (BBU, LVIS, and future domains). Every record MUST adhere to this contract so that a single chat template pipeline can process all sources.

## Top-Level Record
- `images` (list[str], required): Relative paths to image files; resolved against the JSONL directory.
- `objects` (list[object], required): Structured annotations (see below).
- `width` (int, required): Image width in pixels (original or post-resize if applied offline).
- `height` (int, required): Image height in pixels.
- `summary` (str, optional): Single-line summary text (present on BBU; may be absent on public sources).
- `metadata` (object, optional): Free-form metadata; fusion injects `_fusion_source`, `_fusion_template`, `_fusion_domain` here at load time.

## Objects
Each object MUST contain exactly one geometry field plus a non-empty `desc`.
- `desc` (str, required): Text description (domain-specific; e.g., Chinese attributes for BBU, concise English class for LVIS/COCO).
- One geometry (required, mutually exclusive):
  - `bbox_2d`: `[x1, y1, x2, y2]` pixel coordinates.
  - `poly`: flat list `[x1, y1, x2, y2, ...]` (even length, ≥6 values / ≥3 points). Optional `poly_points` (int) should equal `len(poly)/2` when present.
  - `line`: flat list `[x1, y1, ..., xn, yn]`. Optional `line_points` (int) should equal `len(line)/2` when present.
- No additional geometry fields are allowed on the same object.

## Invariants
- Coordinates are pixel-space integers or floats in the source frame. Templates normalize (e.g., norm1000) during encoding.
- Image paths remain relative in JSONL; loaders resolve them to absolute paths.
- Geometry is validated; records with multiple geometry fields per object are rejected.
- Optional fields (e.g., `summary`, `poly_points`, `line_points`, `metadata`) may be absent; templates and preprocessors must tolerate absence.

## Example (with optional summary)
```json
{
  "images": ["images/0001.jpg"],
  "objects": [
    {"poly": [12, 34, 56, 34, 56, 78, 12, 78], "poly_points": 4, "desc": "设备/属性A"},
    {"bbox_2d": [100, 120, 180, 200], "desc": "标签/黄色"}
  ],
  "summary": "设备×1，标签×1",
  "width": 768,
  "height": 512
}
```

## Current Sources (checked)
- `data/bbu_full_768_poly/train.jsonl`: includes `summary`; objects use `poly` or `bbox_2d`.
- `public_data/lvis/rescale_32_768_poly_max_12/train.jsonl`: same structure, no `summary`; objects may include `poly_points` metadata.

All future domains MUST emit this contract to remain compatible with the shared chat template pipeline.

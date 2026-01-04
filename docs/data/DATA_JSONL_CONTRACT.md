# Data JSONL Contract (Global)

Status: Active
Scope: Global JSONL schema for all datasets consumed by training/eval pipelines.
Owners: Data Pipeline
Last updated: 2026-01-02
Related: [DATA_AND_DATASETS.md](DATA_AND_DATASETS.md), [DATA_PREPROCESSING_PIPELINE.md](DATA_PREPROCESSING_PIPELINE.md), [DATA_AUGMENTATION.md](DATA_AUGMENTATION.md)

This document defines the universal JSONL format consumed by all training/eval datasets (BBU, LVIS, and future domains). Every record MUST adhere to this contract so that a single chat template pipeline can process all sources.

See also:
- `README.md` — index of data documentation under `docs/data/`
- `BBU_RRU_BUSINESS_KNOWLEDGE.md` — BBU/RRU domain glossary (categories/attributes/grouping rules)

## Top-Level Record
- **Provenance**: Records are typically produced either by the offline converter (`data_conversion/convert_dataset.sh`, see `./DATA_PREPROCESSING_PIPELINE.md`) or by domain-specific public converters. Regardless of source, they MUST match this contract.
- `images` (list[str], required): Relative paths to image files; resolved against the JSONL directory.
- `objects` (list[object], required): Structured annotations (see below).
- `width` (int, required): Image width in pixels (original or post-resize if applied offline).
- `height` (int, required): Image height in pixels.
- `summary` (str, optional): Single-line **JSON string** summary (present on BBU/RRU; may be absent on public sources). For BBU/RRU converters, the summary JSON includes `dataset`, `统计`, and optional `异常` (only when non-zero), with optional `备注` (BBU only, non-empty) or `分组统计` (RRU only, when present). Summary statistics only include observed values (no missing counts). Irrelevant-image streams may use the literal string `无关图片` instead of JSON. Conversion is fail-fast: missing objects, empty `desc`, or invalid/unknown/conflict markers raise `ValueError` during build.
- `metadata` (object, optional): Free-form metadata; fusion injects `_fusion_source`, `_fusion_template`, `_fusion_domain` here at load time.

## Record Variants
- **Dense-caption records (images/objects)**: Follow the full contract below (`images`, `objects`, `width`, `height`, optional `summary`).
- **Chat-only records (template: chatml)**: Provide `messages` only (pre-authored chat turns). They MAY omit `images`, `objects`, `summary`, `width`, and `height`. The loader treats these as pass-through conversations and skips dense-caption construction.

## Objects
Each object MUST contain exactly one geometry field plus a non-empty `desc`.
- `desc` (str, required): Text description (domain-specific). For LVIS/COCO and other generic detection sources, this is typically a single category token or short phrase. For BBU/RRU, it uses comma-separated `key=value` pairs with no spaces; `类别` is always first. `文本`/`备注` are free text: whitespace is removed, but punctuation (including `,|=`) is preserved; stray comma tokens without `key=` are folded into `备注`.
  - **Fixed value compression (BBU/RRU)**: non‑free‑text values are normalized to compact forms (e.g., `可见性=完整/部分`, `符合性=符合/不符合`, `挡风板需求=免装/空间充足需安装`, `保护措施=有保护/无保护`, `弯曲半径=半径合理/半径不合理<4cm或成环`, `安装状态=合格/不合格`, `标签=有标签/无标签`, `套管保护=有套管/无套管`).
- One geometry (required, mutually exclusive):
  - `bbox_2d`: `[x1, y1, x2, y2]` pixel coordinates.
  - `poly`: flat list `[x1, y1, x2, y2, ...]` (even length, ≥8 values / ≥4 points; current runtime validation). Optional `poly_points` (int) should equal `len(poly)/2` when present.
  - `line`: flat list `[x1, y1, ..., xn, yn]`. Optional `line_points` (int) should equal `len(line)/2` when present.
- No additional geometry fields are allowed on the same object.
- **Groups (RRU only)**: There is **no top-level `groups` key**. Group membership is encoded directly in `desc` as `组=<id>` (multiple groups joined with `|`). BBU MUST NOT include `组`. Conversion fail-fast rejects samples where a declared group contains only one object.

## Invariants
- Coordinates are pixel-space integers or floats in the source frame. Templates normalize (e.g., norm1000) during encoding.
- Image paths remain relative in JSONL; loaders resolve them to absolute paths.
- Geometry is validated; records with multiple geometry fields per object are rejected.
- Polygon vertices are canonicalized offline: duplicate closing points are removed, vertices are ordered clockwise around the centroid, and the starting vertex is the top-most (then left-most) point. This prevents self-crossing orderings across converters and visualization tools.
- Optional fields (e.g., `summary`, `poly_points`, `line_points`, `metadata`) may be absent; templates and preprocessors must tolerate absence.

## Example (with optional summary)
```json
{
  "images": ["images/0001.jpg"],
  "objects": [
    {"poly": [12, 34, 56, 34, 56, 78, 12, 78], "poly_points": 4, "desc": "类别=设备,属性=属性A"},
    {"bbox_2d": [100, 120, 180, 200], "desc": "类别=标签,文本=黄色"}
  ],
  "summary": "{\"dataset\": \"BBU\", \"统计\": [{\"类别\": \"设备\", \"属性\": {\"属性A\": 1}}, {\"类别\": \"标签\", \"文本\": {\"黄色\": 1}}]}",
  "width": 768,
  "height": 512
}
```

## Current Sources (checked)
- `data_new_schema/bbu_full_1024_poly_new_schema/all_samples.jsonl`: BBU domain (new-schema export); includes `summary`; objects use `poly`/`bbox_2d`/`line`. Train/val splits exist in the same directory.
- `data_new_schema/rru_full_1024_poly_new_schema/all_samples.jsonl`: RRU domain (new-schema export); includes `summary`; objects use `bbox_2d`/`poly`/`line`, `desc` carries group membership via `组=<id>` and station distance as `类别=站点距离,站点距离=<int>` (digits). `summary` may include `分组统计`.
- `data/bbu_full_768_poly/train.jsonl`: legacy BBU dense+summary; objects use `poly` or `bbox_2d`.
- `data/rru_full_1024_poly/all_samples.jsonl`: RRU domain (current training/eval); same structure as above.
- `public_data/lvis/rescale_32_768_poly_max_12/train.jsonl`: same structure, no `summary`; objects may include `poly_points` metadata.

All future domains MUST emit this contract to remain compatible with the shared chat template pipeline.

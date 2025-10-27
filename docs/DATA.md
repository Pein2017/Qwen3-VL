# Data

Source of truth: `src/datasets/data_details.md`, `docs/DATA_FORMATS.md`, `src/datasets/geometry.py`, `src/datasets/utils.py`

## JSONL schema (per record)
```json
{
  "images": ["path/to/img1.jpg", "path/to/img2.jpg"],
  "objects": [
    {"bbox_2d": [x1, y1, x2, y2], "desc": "..."},
    {"quad": [x1, y1, x2, y2, x3, y3, x4, y4], "desc": "..."},
    {"line": [x1, y1, ..., xn, yn], "line_points": N, "desc": "..."}
  ],
  "width": 1920,
  "height": 1080,
  "summary": "可选: 单行中文汇总"
}
```
- Image paths resolve relative to the JSONL file directory; absolute paths allowed
- Exactly one geometry field per object; `line_points` must match the number of line coords/2

## Modes: dense vs summary vs mixed
- Dense (default): grouped JSON with geometry per 图片_i
- Summary: one-line per image; requires `summary` in every record
- Mixed: per-group selection via `summary_ratio`; deterministic per epoch (seeded)

## Coordinates and normalization
- On disk: pixel coordinates with `width`/`height`
- Assistant text: controlled by `custom.emit_norm` (`none|norm100|norm1000`)
- Template encoding: top-level `objects.bbox` always normalized to norm1000 for grounding

## Summary field standard（全部斜杠）
- Replace commas with slashes; group identical items with ×N; preserve order
- Optional final `备注:` segment as the last slash level; no special tokens

## Verification
- Geometry validator: one geometry per object; correct dims; `line_points` matches
- Summary validator: all-slash format and presence when summary modes are used
- Mode preview: confirm what the model sees with mixed/summary ratios

## Best practices
- Keep pixel coords on disk; let the template normalize
- Prefer relative image paths; keep original `width`/`height`
- Validate before training; fail fast on schema errors

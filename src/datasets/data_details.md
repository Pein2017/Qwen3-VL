 # BBU Installation Quality Inspection Dataset

## Purpose & Scope
- Vision-language dataset for detecting BBU-related components and assessing installation quality in telecom cabinets.
- Downstream consumer of JSONL produced by `data_conversion/unified_processor.py`. This document is the concise, English-only spec.
- Supports a fourth SFT dialogue variant (optional): image → one-line Chinese summary. Summaries are deterministically distilled from per-object `desc` fields and used only for SFT summary training.

## Files
- Location: `data/ds_v2_full/`
  - `train.jsonl`, `val.jsonl`
  - `teacher_pool.jsonl`
  - `all_samples.jsonl`
- Optional summary SFT files (if generated):
  - `data/summary_sft/train.jsonl`, `data/summary_sft/val.jsonl` (see Summary Field below)

## Record Schema (JSONL per line)
- images: List[str] — processed image paths (usually 1 path per record).
- objects: List[Object]
  - Exactly one geometry field per object:
    - bbox_2d: [x1, y1, x2, y2]
    - quad: [x1, y1, x2, y2, x3, y3, x4, y4]
    - line: [x1, y1, x2, y2, ..., xn, yn]
  - desc: Hierarchical, slash- and comma-separated string built from the templates below. The literals in data are canonical and defined by the taxonomy/mapping; this document explains them in English. A final slash-level free-text "remarks" segment is allowed (see Description Construction).
- width, height: int — processed image size.
- meta: Dict — optional; not stored in v2_full JSONL (see Meta block).
- summary: str — single-line Chinese summary included in v2_full; also usable as the SFT "summary" target. Per-image only: must NOT include any group-level or pass/fail decisions; must contain no coordinates or special tokens.

## Geometry & Object Types
- Geometry formats
  - bbox_2d: axis-aligned rectangle
  - quad: 4-point polygon
  - line: polyline (fibers/wires)
- Allowed geometry per object type
  - bbu: quad, bbox_2d
  - bbu_shield: quad, bbox_2d
  - connect_point (screws, fiber connectors): quad, bbox_2d
  - fiber: line
  - wire: line
  - label: quad, bbox_2d

## Attributes by Object Type
All values in data are canonical (defined in the taxonomy). Below are their meanings in English.

- bbu
  - brand: {Huawei, ZTE, Ericsson}
  - visibility: {complete, partial}
  - windshield_requirement: {not_required, required}
  - windshield_conformity (only if required): {shield_present_per_requirement, shield_missing_or_nonconformant}
  - remarks (free text, optional) as final slash segment in desc

- bbu_shield
  - brand: {Huawei, ZTE}
  - visibility: {complete, partial}
  - obstruction: {clear, blocked}
  - install_direction: {correct, incorrect}
  - remarks (free text, optional) as final slash segment in desc

- connect_point (screws, fiber connectors)
  - type: {bbu_mount_screw, cabinet_ground_screw, busbar_ground_screw, fiber_connector_bbu_end, fiber_connector_odf_end}
  - visibility: {complete, partial}
  - compliance: {compliant, noncompliant}
  - specific_issues (only if noncompliant; multi-value): {loose, exposed_copper, double_connection, rusty}
  - remarks (free text, optional) as final slash segment in desc

- fiber
  - obstruction: {clear, blocked}
  - protection: {none, protected}
  - protection_details (only if protected): {snake_tube, armored, both}
  - bend_radius: {ok, violation}
  - remarks (free text, optional) as final slash segment in desc

- wire
  - obstruction: {clear, blocked}
  - organization: {neat, disorganized}
  - remarks (free text, optional) as final slash segment in desc

- label
  - text: free text (OCR-derived); may be absent if unreadable

## Description (desc) Construction
- Separators
  - comma ",": separate attributes within a level
  - slash "/": separate hierarchical levels
  - conditional attributes appear only when parent condition is met
  - free text is inserted as-is for labels and for a final optional "remarks" segment
- Object templates (English structure; optional segments in brackets)
  - bbu: "BBU/brand,visibility,windshield_requirement/[windshield_conformity]/[remarks]"
  - bbu_shield: "Shield/brand,visibility,obstruction,install_direction/[remarks]"
  - connect_point: "ConnectPoint/type,visibility,compliance/[specific_issues]/[remarks]"
  - fiber: "Fiber/obstruction,protection,bend_radius/[protection_details]/[remarks]"
  - wire: "Wire/obstruction,organization/[remarks]"
  - label: "Label/[text]"
- Slash-level semantics (positional contract)
  - Level-0 (prefix): object type literal (e.g., "BBU", "Shield", ...)
  - Level-1: comma-joined canonical attributes for the object (e.g., brand, visibility, ...)
  - Level-2: conditional segment only when the parent condition is met (e.g., bbu `windshield_conformity` when requirement=required; connect_point `specific_issues` when compliance=noncompliant; fiber `protection_details` when protected)
  - Level-Last: optional final "remarks" free-text segment. When present, it always occupies the last slash level after all canonical and conditional segments.
- Remarks (free text)
  - Purpose: carry special circumstances that affect QC interpretation (e.g., "cannot determine", "angle issue", "partially sleeved snake tube", "screw missing", "rectified", "space limited").
  - Placement: only as the final slash segment of desc (after all canonical attributes and any conditional segments). Consumers should detect remarks by position (last slash segment), not by keywords.
  - Sanitization: must not contain coordinates or special tokens (no `<|...|>`, `<`, `>`, `[`, `]`); keep short (recommended ≤ 40 Chinese characters or equivalent length).

## Meta (per record)
- tokens: List[str] — normalized vocabulary present in the sample
- object_types_present: List[str]
- geometry_set: List[str] — {bbox_2d, quad, line}
- brand: str — sample-level brand context if determinable; otherwise "unknown"
- object_count: int — number of objects
- small_box_frac: float — fraction of small objects
- line_points: int — total number of polyline points
- line_length_norm: float — normalized polyline length
- has_negative: bool — whether any negative/noncompliant attribute exists
- negatives: List[str] — negative attributes present
- difficulty: str — {easy, medium, hard}

## Validation Rules
- One geometry field per object; geometry must match the object type.
- desc must follow the template structure and separator rules.
- For bbu with windshield_requirement = required, include windshield_conformity.
- For fiber with protection = protected, include protection_details.
- For connect_point with compliance = noncompliant, specific_issues may include multiple values (comma-separated).
- Remarks (if present) must appear only as the final slash segment and pass sanitization (no special tokens/coordinates).
- Labels may omit text when unreadable.

## Modeling Notes
- Training: parse desc into structured labels (object type + attributes) for multi-task learning: detection + attribute classification + compliance. A final free-text remarks segment may be present and is treated as unstructured text.
- Optional SFT summary variant: a single-line Chinese summary per image distilled from objects (no coordinates/special tokens; label readability only as clear/unclear). This stabilizes Stage-A behavior before RL.
- Inference: reconstruct desc per object and compute image-level QC metrics (e.g., shield presence when required, fiber bend violations, connector compliance, wiring organization). Remarks may carry human notes such as "cannot determine" or "rectified"; do not force parsing.

## On-Disk JSONL (v2_full)

- Location: `data/ds_v2_full/train.jsonl`, `data/ds_v2_full/val.jsonl`
- Top-level fields (per line/record):
  - `images`: List[str] — usually one processed image path.
  - `objects`: List[Object] — each object has exactly one geometry field and a `desc` string.
    - Geometry: one of `bbox_2d: [x1,y1,x2,y2]`, `quad: [x1,y1,x2,y2,x3,y3,x4,y4]`, or `line: [x1,y1, ..., xn,yn]`.
    - `desc`: Chinese canonical, slash- and comma-separated as described above, with the final optional slash-level used for free-text remarks. On disk, Chinese tokens are used (examples below), not the English labels from this spec.
  - `width`, `height`: int — processed image size.
  - `summary`: str — one-line Chinese summary enumerating per-type counts using the `×N` convention; may include a trailing `备注:` segment. No coordinates or special tokens.
  - `meta`: not present in v2_full; compute downstream if needed.

Notes on Chinese canonical tokens seen on disk (non-exhaustive):
- Object types: `BBU设备` (BBU), `挡风板` (BBU shield), `螺丝、光纤插头` (connect_point), `光纤` (fiber), `电线` (wire), `标签` (label)
- Visibility: `显示完整`, `只显示部分`
- Windshield requirement: `机柜空间充足需要安装`, `无需安装`
- Windshield conformity (level-2 when required): `这个BBU设备按要求配备了挡风板`, `这个BBU设备未按要求配备挡风板`
- Shield install direction: `安装方向正确` (and analogously `安装方向错误` when incorrect)
- Connect point types: `BBU安装螺丝`, `机柜处接地螺丝`, `地排处接地螺丝`, `BBU端光纤插头`, `ODF端光纤插头`
- Compliance: `符合要求` (and `不符合要求` when noncompliant; may include specific issues as an extra slash-level when present)
- Fiber protection: `有保护措施` with details `蛇形管`, `铠装`, `同时有蛇形管和铠装`; bend radius: `弯曲半径合理` / `弯曲半径不合理`
- Wire organization: `捆扎整齐` (neat)
- Label unreadable sentinel: `无法识别` (otherwise arbitrary text)
- Remarks: final slash-level free text often prefixed with `备注:` in data; position, not the keyword, determines semantics.

### Minimal JSONL examples

```json
{"images": ["images/example1.jpeg"], "objects": [
  {"quad": [0, 120, 300, 130, 290, 430, 0, 420], "desc": "BBU设备/华为,显示完整,机柜空间充足需要安装/这个BBU设备按要求配备了挡风板"},
  {"quad": [210, 160, 260, 165, 255, 800, 205, 790], "desc": "挡风板/华为,显示完整,安装方向正确"},
  {"bbox_2d": [150, 440, 175, 460], "desc": "螺丝、光纤插头/BBU端光纤插头,显示完整,符合要求"}
], "summary": "BBU设备/华为/显示完整/这个BBU设备按要求配备了挡风板×1，螺丝、光纤插头/显示完整/符合要求×1，挡风板/显示完整/安装方向正确×1", "width": 420, "height": 896}
```

```json
{"images": ["images/example2.jpeg"], "objects": [
  {"line": [120, 80, 150, 100, 200, 130], "desc": "光纤/有保护措施,弯曲半径合理/蛇形管"},
  {"bbox_2d": [260, 310, 300, 345], "desc": "螺丝、光纤插头/BBU安装螺丝,只显示部分,符合要求"},
  {"quad": [280, 300, 300, 305, 286, 350, 266, 345], "desc": "标签/NR900-RRU1-光纤"}
], "summary": "光纤/有保护措施/蛇形管/弯曲半径合理×1，螺丝、光纤插头/只显示部分/符合要求×1，标签/可以识别×1", "width": 532, "height": 728}
```

```json
{"images": ["images/example3.jpeg"], "objects": [
  {"quad": [10, 100, 380, 110, 360, 600, 0, 580], "desc": "BBU设备/华为,只显示部分,无需安装/备注:疑似华为设备"},
  {"bbox_2d": [60, 850, 85, 875], "desc": "螺丝、光纤插头/BBU安装螺丝,显示完整,符合要求"},
  {"quad": [90, 500, 140, 510, 120, 560, 70, 545], "desc": "标签/无法识别"}
], "summary": "BBU设备/华为/只显示部分/无需安装×1，螺丝、光纤插头/显示完整/符合要求×1，标签/无法识别×1，备注: 疑似华为设备", "width": 420, "height": 896}
```

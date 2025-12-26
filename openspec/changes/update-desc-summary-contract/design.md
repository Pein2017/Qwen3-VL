# Design: update-desc-summary-contract

## Goals
- Replace slash-delimited desc strings with deterministic `key=value` pairs.
- Emit JSON-string summaries with per-category counts (no `×N` aggregation).
- Enforce BBU vs. RRU differences: BBU keeps `备注` and has no groups; RRU uses optional `组=<id>` and omits `备注`.
- Preserve OCR/备注 payloads (remove whitespace only; keep punctuation including `,|=`).

## Description Contract
- **Format**: comma-separated `key=value` pairs; no spaces; `类别` always first.
- **Value sanitization**: remove whitespace; keep punctuation (including `,|=`) for free-text `文本/备注`.
- **Free-text folding**: stray comma tokens without `key=` are folded into `备注`; `这里已经帮助修改,请注意参考学习` is stripped from `备注` when present.
- **OCR**: labels use `文本=<OCR>`; if unreadable or empty → `可读性=不可读` (no “可以识别/无法识别” rewrite).
- **Station distance**: normalize `站点距离` to an integer token (strip non-digits). Current RRU exports only contain digit-bearing strings (e.g., `373、`), so training output is always `站点距离=<int>` (no fallback token).
- **Groups**: RRU-only `组=<id>` (multiple groups joined with `|`, ascending numeric); BBU ignores groups.
- **Negative precedence**: if both positive and negative values are present for a single attribute, keep only negative values.
- **Occlusion**: drop obstruction/遮挡 values entirely.

## Summary JSON Schema
Top-level keys:
- `dataset` (string, `BBU`/`RRU`)
- `objects_total` (int)
- `统计` (list): each item has `类别` plus any observed attribute counts (`{value: count}`)
- `异常` (object): `无法解析`, `未知类别`, `冲突值`, `示例`
- BBU adds `备注` (list); RRU omits `备注` and MAY include `分组统计`

Notes:
- Counts only include observed values; no missing counts.
- Group counts appear under per-category `组` and optional top-level `分组统计`.
- Irrelevant-image streams keep the literal `summary: 无关图片` (non-JSON).

## Compatibility Strategy
- Builders/parsers accept both new `key=value` descs and legacy slash-delimited strings.
- Stage-A/Stage-B prompt tooling parses JSON summaries when available, with legacy fallbacks to avoid breaking older data.

## Geometry JSON Spacing
- No functional change: JSONLinesBuilder continues to serialize assistant JSON with `", "` and `": "` separators to preserve tokenizer behavior for geometry arrays.

## Risks & Mitigations
- **Summary format breakage**: migrate summary preprocessors/tests and Stage-B prompt parsing; keep legacy fallback parsing.
- **Mixed-format data**: format converter and sanitizers accept both formats; summary builder treats non-`key=value` as invalid and reports via `异常`.

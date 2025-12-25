# Proposal: update-desc-summary-contract

## Summary
Refactor BBU/RRU conversion outputs to a key=value description contract, remove slash-delimited descs, emit JSON-string summaries instead of xN aggregation, and codify BBU vs. RRU differences (BBU has remarks but no groups; RRU has optional groups but no remarks). Keep geometry output formatting stable (with spaces in JSON text) to preserve tokenizer distribution.

## Background & Evidence
- Current desc strings are slash/comma hybrids and include 需复核, which is redundant and user-rejected.
- Summary currently aggregates as desc×N, which causes model bias. Requested replacement is JSON-string summaries.
- OCR content must preserve '-' and '/' characters; values may include those tokens in real data.
- Raw data review:
  - BBU: fixed value sets match taxonomy; only conflict found is 符合要求 + 不符合要求/问题 in the same list (5 samples). Rule: if any negative value exists, choose negative.
  - RRU: no conflicts detected; labels/keys align with expected vocabulary.
  - RRU groups only appear as 组1/组2 and are optional.
- train_tiny.jsonl confirms no additional categories beyond plan; 地排接地端 is RRU-only and must remain distinct from BBU.

## Goals
- Make desc unambiguous and parseable with key=value pairs and comma separators.
- Remove 需复核 from outputs entirely; preserve 备注 only for BBU.
- Preserve OCR payloads (keep '-' and '/') while maintaining desc parse safety.
- Make summary output a JSON string with per-category statistics (no × tokens).
- Maintain geometry JSON formatting with spaces to avoid tokenizer drift.

## Non-Goals
- No schema changes to objects geometry fields (bbox_2d, poly, line).
- No training/runtime changes beyond converter + builder handling required for new desc format.
- No summary aggregation across datasets beyond BBU/RRU conversion outputs.

## Impacted Areas (expected)
- data_conversion/pipeline/flexible_taxonomy_processor.py (desc construction, OCR, RRU grouping)
- data_conversion/pipeline/unified_processor.py (remove 需复核, remark handling)
- data_conversion/utils/sanitizers.py (desc safe-value normalization)
- data_conversion/pipeline/summary_builder.py (JSON-string summary output)
- src/datasets/builders/jsonlines.py (object ref parsing with key=value)
- src/datasets/augmentation/* (completeness updates for 可见性 form)
- Docs: docs/data/DATA_JSONL_CONTRACT.md, docs/data/DATA_PREPROCESSING_PIPELINE.md

## Risks
- Downstream consumers expecting '/'-delimited desc may break; mitigation is updating builder/augmentation logic to use 类别= and 可见性= tokens.
- OCR values include separators; strict value sanitation is required to keep parsing deterministic.
- Summary format change affects summary-mode training prompts and any analytics expecting ×N strings.

## Open Questions
- None. All behavioral choices have been confirmed (negative precedence, RRU no remarks, BBU no groups, JSON summary string, OCR whitespace removal).

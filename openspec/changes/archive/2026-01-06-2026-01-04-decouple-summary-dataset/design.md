# Design: Summary JSON decoupled from dataset identity

## Overview
This change removes `dataset` from summary JSON payloads and relocates domain identity to fusion metadata. The two-line header remains the supervised signal for summary-mode outputs; Stage-B drops headers and forwards the remaining payload as-is (no schema validation).

## Key decisions
1. **Summary JSON shape**
   - Required: `统计`.
   - Optional: `备注`, `分组统计`.
   - Forbidden: `dataset` (fail-fast where enforced).

2. **Domain identity**
   - No new metadata fields are introduced.
   - Domain tokens are derived at runtime from existing fusion metadata:
     - `_fusion_source == "irrelevant_summary"` → no header (even if `_fusion_template` is `summary_bbu`).
     - `_fusion_template == "summary_bbu"` → `BBU`
     - `_fusion_template == "summary_rru"` → `RRU`
   - Summary targets MUST resolve a domain token via the mapping above; otherwise fail fast before header construction.

3. **Header supervision**
   - Summary-mode outputs keep the two-line header.
   - Header uses `domain_token` derived from existing fusion metadata (`_fusion_template`) and fixed `<TASK=SUMMARY>`.

4. **Stage-B audit behavior**
   - Header is dropped before any parsing.
   - The remaining payload is forwarded as-is; no schema validation is enforced.
   - Any extra keys (including `dataset`) or non-JSON text are tolerated.

5. **No backward compatibility**
   - Training corpora and GRPO references MUST NOT include `dataset` in summary JSON.

## Data flow

### Offline conversion (data_conversion)
- Converters emit summary JSON without `dataset`.
- Summary JSON contains `统计` plus optional `备注` (BBU) or `分组统计` (RRU).

### SFT / GRPO (summary-mode)
- Input JSONL contains `summary` strings with `统计` and optional `备注` / `分组统计`.
- Header is generated dynamically from existing fusion metadata + mode (no new fields).
- GRPO rewards validate header `<DOMAIN>` against metadata-derived domain token, not JSON.

### Stage-B inference
- Stage-B drops the header line and forwards the remaining payload as-is.
- Any `dataset` keys (or non-JSON text) are tolerated without error.

## Failure modes and validation
- Summary JSON with `dataset` key fails validation in SFT and GRPO (training corpora only).
- Summary targets whose `_fusion_template` does not resolve to `BBU` or `RRU` fail header construction (SFT/GRPO).
- Irrelevant samples are exempt from header requirements.

## Compatibility
- This is a breaking change. All legacy summary JSONL corpora must be migrated.
- Stage-B drops headers and tolerates arbitrary second-line payloads (no schema enforcement).

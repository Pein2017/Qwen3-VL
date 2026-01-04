# Proposal: Decouple summary JSON from dataset identity

## Why
- Summary outputs currently duplicate domain identity in both the header (`<DOMAIN=...>`) and the JSON payload (`summary.dataset`). The header already provides the supervised domain signal for summary-mode training.
- Keeping `dataset` inside the JSON summary couples downstream parsing to a redundant field and makes the contract harder to evolve.
- Stage-B only needs the statistical content (`统计` + optional `备注/分组统计`) and can drop headers entirely during audit.

## What
- **Summary JSON no longer contains `dataset`.** The JSON payload contains `统计` and optional `备注` / `分组统计` only.
- **Domain identity is derived from existing fusion metadata** (`_fusion_template` and `_fusion_source`) without adding new fields. `_fusion_source == "irrelevant_summary"` emits no header (even if `_fusion_template` is `summary_bbu`). Summary templates match exactly: `summary_bbu` → `<DOMAIN=BBU>`, `summary_rru` → `<DOMAIN=RRU>`.
- **Two-line summary header remains** as the primary supervised signal for SFT/GRPO: `<DOMAIN=...>, <TASK=SUMMARY>` + JSON line.
- **GRPO dataset reward compares header to metadata**, not JSON.
- **Stage-B drops the header line** and forwards the remaining payload as-is (no schema validation; any extra keys or free text are tolerated).
- **No backward compatibility**: training corpora and GRPO references must not contain `dataset` in summary JSON (fail-fast in training/GRPO validation).

## Scope
- Summary-mode contract in data docs and prompt text.
- Offline converters in `data_conversion/` that emit training JSONL summaries.
- Fusion loader prompt/header construction (derive domain tokens from existing metadata).
- Summary-mode GRPO reward parsing.
- Stage-B summary parsing and prompt assembly.

## Non-goals
- Changing dense-caption JSON structure or LVIS/ChatML handling.
- Altering the two-line verdict protocol for Stage-B.
- Introducing new task tags; task remains derived from dataset mode.

## Impact / Breaking changes
- **Breaking**: summary JSON strings containing `dataset` are invalid and rejected.
- **Breaking**: reward/parse logic no longer consults JSON `dataset` fields.
- Migration is required for all summary JSONL corpora.

## Success criteria
- Summary-mode training continues to use two-line headers and produces stable outputs without JSON `dataset`.
- GRPO content rewards score `统计` + `备注` and `分组统计` (RRU) without using JSON `dataset`.
- Stage-B consumes Stage-A summaries by dropping headers and passing through the remaining payload.

## Illustration
**Summary output (training/GRPO supervised target)**:
```
<DOMAIN=BBU>, <TASK=SUMMARY>
{"统计":[{"类别":"标签","文本":{"5G-AAU2-光纤":1}}],"备注":["无法判断品牌"]}
```

**Stage-B ingestion behavior**:
- Drop the first line.
- Keep the second line unchanged (whether JSON or not).

## Risks
- Any legacy summary JSONL will fail validation until migrated.
- Incorrect summary template assignment (e.g., `summary_bbu` vs `summary_rru`) would corrupt header supervision; template values must remain deterministic and validated.

## Rollout plan (high-level)
1. Update contract + prompts.
2. Update `data_conversion/` summary emission to match the new schema.
3. Update fusion header derivation to use existing metadata only.
4. Update GRPO reward to compare header vs metadata.
5. Update Stage-B parsing.
6. Migrate summary JSONL corpora.
7. Add tests for failure on `dataset` in summary JSON.

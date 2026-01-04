# Tasks: Decouple summary JSON from dataset identity

- [x] Update summary JSON contract in docs to remove `dataset` from summary JSON and declare fail-fast behavior.
- [x] Update summary prompts to require `统计` (and optional `备注` / `分组统计`) without `dataset`.
- [x] Update `data_conversion/` summary emission to remove `dataset`.
- [x] Update fusion header construction to derive domain token from existing metadata (`_fusion_template` + `_fusion_source`) and suppress header for irrelevant samples.
- [x] Update summary-mode GRPO reward to compare header domain token to metadata-derived token from `_fusion_template` (no JSON `dataset`).
- [x] Update Stage-B summary handling to drop the header line and pass through the remaining payload without schema validation.
- [x] Migrate summary JSONL corpora (train/val) to remove `dataset` from summary payloads.
- [x] Add tests for:
  - summary JSON containing `dataset` fails fast;
  - header vs `_fusion_template`-derived token reward; 
  - Stage-B header drop with arbitrary second-line payload (JSON or non-JSON).
- [x] Update runbooks (Stage-B, training/GRPO) to document stats-only summaries and header derivation from fusion metadata.

# Tasks: update-irrelevant-summary-format

- [x] Update summary prompt text so irrelevant samples output a single line `无关图片` while non-irrelevant samples keep the two-line contract (`src/prompts/summary_core.py`).
- [x] Ensure irrelevant sample identification relies on `metadata._fusion_source` (e.g., `irrelevant_summary`) rather than output text; document any prompt selection logic that uses it (`src/config/prompts.py`, fusion loader metadata).
- [x] Remove/soften Stage-B prompt wording that prescribes a two-line format for summaries, keeping only the irrelevance cue (`src/prompts/stage_b_verdict.py`).
- [x] Update prompt documentation to reflect the single-line irrelevant output and Stage-B input tolerance (`docs/reference/PROMPTS_REFERENCE.md`).
- [x] Document that the irrelevant pool can remain under fusion targets for full coverage but should be treated as source-like via `_fusion_source` for analysis (`docs/data/UNIFIED_FUSION_DATASET.md` or `docs/data/DATA_AND_DATASETS.md`).
- [x] Implement deterministic per-sample alternation between `summary_bbu` and `summary_rru` for `_fusion_source=irrelevant_summary` without changing fusion sampling (`src/datasets/unified_fusion_dataset.py`).
- [x] Document irrelevant prompt alternation behavior and constraints in data/fusion docs (`docs/data/UNIFIED_FUSION_DATASET.md`, `docs/data/DATA_AND_DATASETS.md`).
- [x] Validate with `openspec validate update-irrelevant-summary-format --strict`.

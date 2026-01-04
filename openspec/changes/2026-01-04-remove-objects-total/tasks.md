# Tasks

- [x] Add OpenSpec delta for `detection-preprocessor` summary JSON schema (remove `objects_total` requirement).
- [x] Update `data_conversion/pipeline/summary_builder.py` to stop emitting `objects_total` in summary JSON.
- [x] Update summary prompt contract (`src/prompts/summary_core.py`) to remove `objects_total` mentions/requirements.
- [x] Update Stage-A summary sanitization/parsing to accept the new schema and ignore legacy `objects_total` when present (`src/stage_a/inference.py`, `src/stage_a/postprocess.py`).
- [x] Update Stage-B prompt builders and reflection logic to not require `objects_total` and to derive any complexity hint from the summary (`src/stage_b/sampling/prompts.py`, `src/stage_b/reflection/engine.py`, `src/prompts/stage_b_verdict.py`).
- [x] Update GRPO reward config surface to remove `summary.objects_total*` from default summary GRPO configs, and keep reward implementation code consistent.
- [x] Update tests covering JSONL summary building, Stage-B prompt formatting, and summary GRPO rewards (remove `objects_total` expectations and add coverage for the new schema).
- [x] Update docs: data JSONL contract + preprocessing pipeline + Stage-B runtime notes to reflect the new summary schema.
- [x] Run `openspec validate 2026-01-04-remove-objects-total --strict` and targeted pytest suite.

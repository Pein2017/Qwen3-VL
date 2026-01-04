# Tasks

- [x] Add OpenSpec delta for `detection-preprocessor` summary JSON schema (remove legacy total-count requirement).
- [x] Update `data_conversion/pipeline/summary_builder.py` to stop emitting legacy total-count fields in summary JSON.
- [x] Update summary prompt contract (`src/prompts/summary_core.py`) to remove legacy total-count mentions/requirements.
- [x] Update Stage-A summary sanitization/parsing to accept the new schema (legacy total-count field removed).
- [x] Update Stage-B prompt builders and reflection logic to not require legacy total-count fields and to derive any complexity hint from the summary (`src/stage_b/sampling/prompts.py`, `src/stage_b/reflection/engine.py`, `src/prompts/stage_b_verdict.py`).
- [x] Update GRPO reward config surface to drop legacy total-count rewards from default summary GRPO configs, and keep reward implementation code consistent.
- [x] Update tests covering JSONL summary building, Stage-B prompt formatting, and summary GRPO rewards (remove legacy total-count expectations and add coverage for the new schema).
- [x] Update docs: data JSONL contract + preprocessing pipeline + Stage-B runtime notes to reflect the new summary schema.
- [x] Run `openspec validate 2026-01-04-remove-objects-total --strict` and targeted pytest suite.

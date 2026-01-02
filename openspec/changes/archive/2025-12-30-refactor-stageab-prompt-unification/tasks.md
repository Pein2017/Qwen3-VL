# Tasks

- [x] Add shared prompt builders for Stage-A summary and Stage-B verdict under src/prompts (task base + domain scenario blocks).
- [x] Refactor Stage-A/Stage-B prompt call sites (stage_a/prompts.py, stage_b/sampling/prompts.py, stage_b/sampling/__init__.py) to delegate to the shared builders while preserving public APIs.
- [x] Update docs: docs/training/REFERENCE.md, docs/runtime/STAGE_A_STAGE_B.md, docs/stage-B-knowledge-Chinese.md (and adjust prompt references/description).
- [x] Optional: add a lightweight prompt assembly sanity check or skip with justification. (skipped: no new test added)

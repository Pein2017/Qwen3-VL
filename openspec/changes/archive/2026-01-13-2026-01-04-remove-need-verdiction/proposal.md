# Proposal: Remove legacy review placeholder key and handling

## Why
- The review placeholder adds a third state that is no longer desired and confuses Stage-B inference and training.
- Downstream users asked for strict binary outcomes; keeping the key encourages silent sanitization and extra heuristics.

## Scope
- Eliminate generation, parsing, and sanitization of legacy review markers across data conversion, Stage-A summaries, Stage-B prompts/runtime, configs, docs, and tests.
- Treat any occurrence as invalid input (fail fast), not as a soft fallback or stop-gradient queue.
- Regenerate affected datasets/prompts after code/doc changes; no backward compatibility path.

## Non-Goals
- No alternative review marker will be introduced.
- No changes to model architectures beyond retraining on the regenerated data.

## Impact
- Breaking change for any pipeline expecting a third-state review placeholder (schema and prompt contracts change).
- Requires regenerating conversion outputs, retraining SFT/Stage-B assets, and rerunning prompt export scripts.

## Rollout / Validation
- Update specs + code + docs first.
- Regenerate canonical JSONL and prompts; ensure ripgrep for legacy markers yields zero in tracked files.
- Run Stage-B smoketests and unit tests for prompt/validation surfaces.

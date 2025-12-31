# Proposal: Unify Stage-A Summary and Stage-B Verdict Prompt Templates

## Why
Stage-A and Stage-B prompt content is currently spread across multiple modules (src/stage_a/prompts.py, src/stage_b/sampling/prompts.py, src/prompts/domain_packs.py, and src/prompts/summary_profiles.py). With RRU support added, maintaining consistent prompt composition and guidance placement has become brittle and harder to audit.

## What
- Introduce centralized prompt templates for Stage-A summary and Stage-B verdict under src/prompts/.
- Split Stage-A prompts so the system prompt holds the summary task base + global non-site rules, while the user prompt carries domain scenario blocks (BBU/RRU) plus optional mission focus.
- Standardize Stage-B prompt assembly so the system prompt is static two-line verdict contract + guardrails (no domain block), while guidance + Stage-A summaries live only in the user prompt.
- Keep prompt text content equivalent where possible; only block placement is reorganized for clarity.

## Impact
- Improves prompt maintainability and consistency across BBU/RRU missions.
- Minimal behavioral risk: prompt blocks are preserved, but a small placement shift (system vs user) occurs for Stage-A domain warnings and Stage-B G0 duplication.
- No changes to training prompt profiles or JSONL schemas.

## Success Criteria
- Stage-A runtime prompt composition is system prompt (summary task base + global non-site rules) plus user prompt (summary instruction + BBU/RRU scenario blocks + mission focus).
- Stage-B rollout prompts are composed as system prompt (two-line contract + guardrails) plus user prompt (guidance + Stage-A summaries); domain knowledge is moved to Stage-A user prompts.
- All Stage-A/Stage-B prompt builders delegate to the new src/prompts modules.
- Docs updated to reflect the new prompt organization.

## Risks / Trade-offs
- Small behavior drift possible due to prompt block relocation; requires a quick spot check on one mission run.
- Additional refactor touches Stage-A and Stage-B runtime code paths; careful diff review needed.

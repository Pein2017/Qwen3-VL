# Refactor Stage-B to Simple Prompt-Only Guidance

## Context
Stage-B is currently built around a training-free GRPO-style loop with CriticEngine, deterministic signals, and multi-stage reflection. For the production missions (4 missions, Stage-A summaries already available), the workload is simpler: concatenate Stage-A tag captions, prompt the LLM once, aggregate a few rollouts, and optionally update short mission guidance snippets. The existing stack is over-complex and couples selection to critic outputs and confidence/self-consistency heuristics.

## Goals
- Replace CriticEngine and deterministic-signal plumbing with a lean prompt-only pipeline.
- Standardize rollout output to include JSON-array evidence fields (`Evidence_Positive`, `Evidence_Negative`) to judge explainability.
- Select via vote strength (majority fraction), no model-reported confidence.
- Route malformed or non-explainable mismatches to explicit artifacts: `manual_review_queue.jsonl` and a `failure` log for format errors.
- Keep per-mission guidance in run dir only (overwrite old framework) and keep docs aligned.

## Non-Goals
- No gradient updates / RL; no retrieval or external tools.
- No re-introduction of critic-style secondary prompts.
- No retry/auto-fix of malformed outputs beyond logging and quarantine.

## Motivations / Value
- Lower latency, fewer prompts, and simpler reasoning surface.
- Clearer governance: only explainable mismatches feed reflection; label-noise cases are quarantined.
- Easier ops: fewer configs (no critic knobs, no confidence/self-consistency tuning), smaller artifacts.

## Risks / Mitigations
- **LLM format drift** without critic: mitigate via strict schema and failure queue for malformed outputs.
- **Loss of critic second opinion**: rely on aggregated rollouts and evidence extraction; reflection uses evidence grounding.
- **Guidance bloat**: cap operations per reflection and reject duplicates; keep ≤32-char micro-guidance rules.

## Impacted Areas
- Code: `src/stage_b` runtime (rollout, selection, reflection), guidance IO, runner; configs under `configs/stage_b/`; scripts `stage_b_run.sh` if flags change.
- Docs: `docs/runtime/STAGE_B_RUNTIME.md`, `docs/runtime/STAGE_A_STAGE_B.md`, `docs/reference/stage-B-knowledge-Chinese.md`, `docs/training/REFERENCE.md`.

## Rollout / Migration
- Overwrite old Stage-B behavior with the simplified path; no dual modes.
- New artifacts (`manual_review_queue.jsonl`, `failure_malformed.jsonl`) live beside guidance/trajectories in the mission run directory.

## Alternatives Considered
- Keep critic optional: rejected (still adds complexity and prompt surface).
- Keep confidence/self-consistency as pseudo-signals: rejected (use vote strength only).

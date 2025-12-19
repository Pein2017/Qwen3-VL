# Design Notes: Simple Prompt-Only Stage-B

## Runtime Flow
1) **Ingest** Stage-A JSONL → normalize `per_image` → concat summaries.
2) **Rollout** a small decode grid; prompt requires outputs with:
   - `Verdict: 通过|不通过`
   - `Reason: <short>`
   - `Evidence_Positive: ["..."]`
   - `Evidence_Negative: ["..."]`
3) **Parse & Validate**: if parse fails or required fields empty → log to `failure_malformed.jsonl`, add to `manual_review_queue.jsonl`, skip reflection.
4) **Select** via majority verdict; tie-breaker lower temperature then order. Use vote strength (majority_fraction) in exports; no confidence/self-consistency.
5) **Manual Review Gate**: if GT≠model and relevant evidence list empty → queue for manual review (no reflection).
6) **Reflection (prompt-only)**: batches of configurable size (debug default 4); only explainable mismatches enter. LLM returns ≤3 micro-guidance edits (≤32 chars) grounded in provided evidence; reject duplicates/Stage-A artifacts; append to mission `guidance.json` in run dir.
7) **Artifacts**: `trajectories.jsonl`, `selections.jsonl`, `reflection.jsonl`, `guidance.json`, `manual_review_queue.jsonl`, `failure_malformed.jsonl` per mission under `{output.root}/{run_name}/{mission}/`.

## Key Changes vs Current Spec
- Remove CriticEngine entirely; no per-candidate critic fields persisted.
- Remove deterministic signals (confidence/self_consistency) usage; rely on vote strength.
- Introduce explicit failure and manual-review queues.
- Guidance scoped to run dir only; overwrite prior framework outputs.

## Open Details (resolved in discussion)
- Evidence format: JSON arrays on a single line.
- Deterministic guards: none; rely on manual-review queue for unexplained mismatches.
- Reflection batch size configurable; default 4 for debug.

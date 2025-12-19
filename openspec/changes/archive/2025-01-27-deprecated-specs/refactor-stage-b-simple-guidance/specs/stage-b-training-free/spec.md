# stage-b-training-free — Simple Prompt-Only Stage-B

## MODIFIED Requirements

### Requirement: Stage-B SHALL evaluate group tickets without gradient updates by orchestrating a prompt-only sampling and judging loop.
Stage-B SHALL rely solely on the frozen checkpoint and prompt-driven rollouts; candidate outputs SHALL include JSON-array evidence fields and no critic pass SHALL run.
#### Scenario: A mission batch of Stage-A summaries is processed for verdict inference
- Given normalized Stage-A summaries and historical human verdicts, when Stage-B ingest + rollout commands run, then the system MUST generate one or more candidates per group using the frozen checkpoint and require each candidate to emit:
  - `Verdict: 通过|不通过`
  - `Reason: <text>`
  - `Evidence_Positive: ["..."]`
  - `Evidence_Negative: ["..."]`
- The pipeline MUST reject/route to failure queue any candidate missing required fields or non-JSON evidence arrays and continue with remaining candidates.
- Selection MUST use majority vote (vote strength) with tie-break by lower temperature then candidate order; no confidence/self-consistency metrics are used.

### Requirement: Stage-B SHALL emit reproducible verdict artifacts with evidence arrays and vote strength.
Stage-B SHALL carry evidence lists and vote strength in exports and SHALL omit critic/confidence fields.
#### Scenario: Selection completes for a mission batch
- When Stage-B selects final verdicts, it SHALL write `selections.jsonl` under `{output.root}/{run_name}/{mission}/` containing `group_id`, `mission`, `verdict`, `reason`, `vote_strength` (majority_fraction), `reflection_cycle`, `guidance_step`, `warnings`, and `conflict/manual_review` flags.
- The pipeline SHALL also append `trajectories.jsonl` entries with decode params, raw response text, parsed verdict/reason, evidence arrays, `format_ok`, `vote_strength_contribution` (per-candidate 1/0), and epoch/reflection metadata. Confidence, self_consistency, and critic fields MUST be absent.

### Requirement: Stage-B SHALL persist mission guidance and review/failure artifacts as fail-fast files.
Mission guidance SHALL remain JSON per mission; new review artifacts SHALL be added and handled fail-fast.
#### Scenario: Artifacts are written after rollout or reflection
- The system SHALL maintain per-mission files under `{output.root}/{run_name}/{mission}/`: `guidance.json`, `trajectories.jsonl`, `selections.jsonl`, `reflection.jsonl`, `manual_review_queue.jsonl`, and `failure_malformed.jsonl` (malformed/format errors).
- Guidance writes MUST stay atomic (temp file + rename) and refuse states where `experiences` is empty; snapshots remain pruned per retention.
- If any artifact cannot be written, the run MUST abort with a clear error.

### Requirement: Stage-B SHALL assemble prompts with mission experiences and enforce evidence-JSON output.
Prompts SHALL demand evidence arrays and block empty experiences.
#### Scenario: A rollout batch is prepared
- The prompt builder SHALL prepend numbered experiences (non-empty) and mission focus, then the concatenated Stage-A summaries, and SHALL instruct the model to output the four required lines with `Evidence_Positive/Evidence_Negative` as JSON arrays.
- If the experiences dict is empty or prompt assembly fails, the pipeline SHALL abort.

### Requirement: Stage-B SHALL log warnings and quarantine problematic tickets with explicit queues.
Warnings SHALL be supplemented by explicit failure and manual-review logs.
#### Scenario: A candidate parse fails or a mismatch lacks explainable evidence
- On malformed outputs (missing/invalid evidence arrays, missing verdict/reason), the system SHALL append a record to `failure_malformed.jsonl`, log a warning, and add the group to `manual_review_queue.jsonl`; reflection MUST NOT run for that group.
- On GT/model mismatches where the relevant evidence list is empty, the system SHALL add the group to `manual_review_queue.jsonl` with reason `no_explainable_evidence` and skip reflection.

### Requirement: Stage-B tooling SHALL expose a fail-fast `run_all()` orchestration entry point with the simplified flow.
The orchestration flow SHALL drop critic/signal phases and keep a single-process entry.
#### Scenario: Operators invoke Stage-B end-to-end
- `run_all()` MUST execute ingest → rollout → parse/format-check → select → manual-review/failure logging → reflection (eligible explainable mismatches only) → guidance update → export, using a single loaded model instance.
- Reflection batch size SHALL be configurable (default 4 in debug config) and limited to ≤3 guidance operations per batch; applied operations increment guidance step and are logged to `reflection.jsonl` with evidence group ids.

## REMOVED Requirements

### Requirement: Stage-B SHALL record minimal deterministic signals for each candidate to support LLM reflection.
Removed: selection and reflection no longer consume confidence/self_consistency/label_match signals; vote strength is derived from rollout counts.
#### Scenario: Signals are generated after rollout
- Superseded by majority-vote aggregation; deterministic signal emission is no longer required or stored.

### Requirement: Stage-B SHALL include an LLM-based CriticEngine for per-candidate evaluations.
Removed: CriticEngine and its outputs are eliminated from the pipeline.
#### Scenario: A batch completes rollout
- Superseded; no critic prompts are issued, and no critic fields are persisted with trajectories or selections.

## MODIFIED Requirements

### Requirement: Stage-B SHALL include an LLM-based CriticEngine for per-candidate evaluations.
Stage-B SHALL produce human-auditable per-candidate evaluations using the CriticEngine, which outputs mandatory `SUMMARY` and `CRITIQUE` lines plus optional keyed lines (`VERDICT`, `NEEDS_RECHECK`, `EVIDENCE_SUFFICIENCY`, `RECOMMENDED_ACTION`) in `KEY: value` format without JSON fences; the system MUST parse these lines into structured critic records, enforce field length caps, and persist them alongside trajectories in JSONL.
#### Scenario: A batch completes rollout and critic evaluation
- Given up to `critic.max_candidates` responses for a group and a prompt that instructs line-based output, when the CriticEngine runs, then it SHALL reject outputs missing `SUMMARY` or `CRITIQUE`, normalize optional keyed fields, and write the parsed `summary/critique/...` to `trajectories.jsonl` with length caps applied.

### Requirement: Stage-B SHALL treat LLM reflection-guided experiences updates as the optimizer step with direct application.
Reflection proposals SHALL be emitted in a line-based protocol (`ACTION`, optional `SUMMARY/CRITIQUE/UNCERTAINTY/EVIDENCE_GROUP_IDS`, and an `OPERATIONS:` block of `- UPSERT|REMOVE|MERGE key=... text=... rationale=... evidence=...`) instead of strict JSON; the engine MUST parse these lines into the existing proposal schema, fail when `ACTION` is missing or when `OPERATIONS` is invalid for `refine`, and persist the structured proposal as JSONL with provenance and guidance step deltas.
#### Scenario: Reflection runs after a batch and proposes a change
- Given a bundle of trajectories and critic outputs, when reflection executes, then it SHALL expect a line-based response, parse it into `{action, summary, critique, operations[], evidence_group_ids, uncertainty_note}`, reject malformed operations or missing `ACTION`, and, if eligible and applied, increment `guidance_step` and append the structured proposal to `reflection.jsonl`.

## ADDED Requirements

### Requirement: Stage-B SHALL parse line-based LLM outputs into structured artifacts before persistence.
LLM generations for critic and reflection MUST be accepted as line-based text, parsed deterministically into dataclasses, and only then written to JSONL artifacts; storage schemas remain unchanged while generation format drops the JSON requirement.
#### Scenario: Line-based outputs are produced during inference
- When a mission run completes critic or reflection generation, then the system SHALL parse the line protocol into structured records, surface format violations via errors or counters (no silent coercion), and write the resulting structured data to existing JSONL files without altering their schema.

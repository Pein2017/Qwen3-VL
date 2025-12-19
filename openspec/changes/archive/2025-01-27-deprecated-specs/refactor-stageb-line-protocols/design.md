# Design: Line Protocols for Stage-B LLM Outputs

## Summary
LLM generations will move from “emit strict JSON” to “emit deterministic line-based text”. Parsers convert the lines into existing structured dataclasses; persistence stays JSONL. This reduces generation fragility at high temperatures while keeping downstream contracts stable.

## Protocols
### Critic (per candidate)
```
SUMMARY: <text>
CRITIQUE: <text>
VERDICT: 通过|不通过              # optional
NEEDS_RECHECK: yes|no            # optional
EVIDENCE_SUFFICIENCY: yes|no     # optional
RECOMMENDED_ACTION: 通过|不通过|人工复核  # optional
```
- Requirements: SUMMARY + CRITIQUE mandatory; one `KEY: value` per line; no fences/JSON/braces.
- Parser: strict line tokenizer; map aliases to existing `CriticOutput`; reject on missing required keys; legacy JSON parsing may be optionally allowed for rollout safety.

### Reflection (batch-level)
```
ACTION: refine|noop
SUMMARY: <optional>
CRITIQUE: <optional>
EVIDENCE_GROUP_IDS: g1,g2        # optional, default=current bundle
OPERATIONS:
  - UPSERT key=<id> text=<...> rationale=<...> evidence=<g1,g2>
  - REMOVE key=<id> rationale=<...>
  - MERGE key=<id> merged_from=<k1,k2> text=<...> rationale=<...>
UNCERTAINTY: <optional>
```
- Requirements: ACTION mandatory; OPERATIONS list may be empty only when ACTION=noop.
- Parser: consume `OPERATIONS:` block with `-` prefix; split fields by spaces and `key=value` pairs; map to `ExperienceOperation`; reject unknown ops; default evidence to bundle ids when omitted.

## Storage and Backward Compatibility
- Runtime parses lines to structured objects; persistence remains identical JSONL schemas (`trajectories.jsonl`, `reflection.jsonl`).
- JSON-generation is discouraged; an optional compatibility flag can keep JSON parsing enabled during rollout.

## Error Handling & Observability
- Violations increment counters/log warnings; Critic/Reflection should fail fast for missing required keys.
- Rollout fallback coercion will be removed or reduced to “drop + log”.

## Risks / Mitigations
- Risk: Line parsing ambiguity. Mitigation: tight grammar (KEY: value), explicit examples in prompts, stop tokens.
- Risk: Downstream expecting JSON text from LLM. Mitigation: storage format unchanged; only generation protocol shifts.

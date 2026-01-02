# Spec Delta — stage-b-training-free

## MODIFIED Requirements

### Requirement: Stage-B SHALL emit two-line verdict responses without evidence lists.
Stage-B SHALL replace the legacy四行协议 with a minimal两行协议 and SHALL reject any outputs containing Evidence_* lines.
#### Scenario: Generating a verdict for a mission ticket
- WHEN the Stage-B sampler asks the LLM for a verdict,
- THEN the model MUST reply with exactly two non-empty lines: `Verdict: 通过|不通过` and `Reason: <简体中文单行>`,
- AND the response MUST NOT include `Evidence_Positive`/`Evidence_Negative` or any JSON evidence arrays,
- AND Stage-B MUST treat any extra lines as noise and ignore them without failing the run.

### Requirement: Stage-B SHALL store rollout artifacts without evidence fields.
All persisted rollout/selection artifacts SHALL drop evidence-related fields to align with the new协议。
#### Scenario: Persisting rollout and selection artifacts
- WHEN trajectories and selections are written,
- THEN the emitted JSON objects MUST include verdict, reason, decode metadata, and label_match,
- AND MUST NOT include evidence_positive, evidence_negative, or confidence fields.

### Requirement: Stage-B reflection SHALL operate on verdict + reason only.
Reflection logic SHALL rely solely on verdict、reason、label/selection signals and SHALL NOT require evidence arrays.
#### Scenario: Building reflection prompts and plans
- GIVEN rollout candidates with verdict and reason,
- WHEN reflection summarises, critiques, or generates guidance updates,
- THEN it MUST reference the reason text (and labels/selection signals) without relying on evidence lists,
- AND reflection MUST remain eligible and produce plans even when no evidence arrays are present.

# Spec Delta — stage-b-training-free

## MODIFIED Requirements

### Requirement: Stage-B inference SHALL emit strict two-line binary outputs and SHALL forbid any third-state wording.
Stage-B inference outputs MUST be exactly two lines and strictly binary; the final output MUST NOT contain any third-state wording (including need-review). If deterministic guardrails override a sampled verdict, the system MUST rewrite the final `Reason:` so it remains consistent with the final `Verdict:` while still forbidding third-state wording.
#### Scenario: Valid inference output contract
- **WHEN** Stage-B generates an inference verdict for a group
- **THEN** it MUST output exactly two lines:
  - Line 1: `Verdict: 通过` OR `Verdict: 不通过`
  - Line 2: `Reason: <...>` (single-line Chinese natural language)
- **AND** the output MUST NOT contain any third-state phrases (including but not limited to: `需复核`, `需人工复核`, `need-review`, `证据不足`, `待定`, `通过但需复核`, `通过但需人工复核`).

#### Scenario: Fail-first override rewrites Reason to match final Verdict
- **GIVEN** a sampled model output indicates `Verdict: 通过`
- **AND** mission-scoped fail-first negative evidence is detected from the Stage-A summaries for the current mission (per `G0`)
- **WHEN** Stage-B finalizes the verdict with deterministic guardrails
- **THEN** the final output MUST be `Verdict: 不通过` with a rewritten `Reason:` that cites the mission-relevant negative evidence
- **AND** the rewritten output MUST still contain no third-state wording.

### Requirement: Stage-B prompts SHALL treat `需复核,备注:` as a soft signal and SHALL NOT equate it to fail.
Stage-B prompts MUST instruct the model to read `备注:` content as evidence and treat `需复核` as an attention marker only (soft signal), not a hard fail rule.
#### Scenario: Remark-aware reasoning for a single image summary
- **GIVEN** a per-image summary contains `需复核,备注:<detail>`
- **WHEN** the model forms a group verdict
- **THEN** it MUST use the remark detail as part of the evidence
- **AND** it MUST NOT infer `不通过` solely because the substring `需复核` appears.

### Requirement: Stage-B SHALL enforce mission-scoped fail-first using mission-agnostic negative triggers and generalized noncompliance patterns.
Stage-B MUST implement a deterministic fail-first guardrail: if any mission-relevant summary clause contains explicit negative evidence, the entire group verdict MUST be `不通过` (fail-first). “Mission relevance” MUST be determined by the current mission guidance `G0` (the same ticket may be audited under different missions, and verdicts may legitimately differ across missions). To support future missions without enumerating every issue string, Stage-B SHALL treat canonical noncompliance patterns used in Stage-A summaries as explicit negative evidence (grounded in the canonical summary vocabulary sources: `data_conversion/hierarchical_attribute_mapping.json` and `output_post/stage_b/initial_guidance.json`). Stage-B prompts SHALL include a short “pattern-first + mission-scope” instruction so the model can catch unseen negative issue descriptors without relying on a hand-written exhaustive list or a per-mission noun glossary.
#### Scenario: Mission-relevant negative trigger forces fail-first (core list)
- **GIVEN** a group where any per-image summary contains at least one of the required trigger phrases in a clause that is relevant to the current mission `G0`:
  - `未按要求`, `错误`, `缺失`, `松动`, `损坏`, `方向不正确`, `反向`, `不符合要求`, `未安装`, `未配备`, `不合格`, `不合理`
- **WHEN** Stage-B produces the final group verdict for this mission
- **THEN** the final verdict MUST be `不通过` (fail-first)
- **AND** the trigger list MUST contain no mission-specific nouns and MUST NOT include pending/uncertain phrases like `需复核`, `无法确认`, `无法判断`, `只显示部分`, `模糊`.

#### Scenario: Mission-irrelevant negative trigger MUST NOT force fail-first
- **GIVEN** a group where a per-image summary contains a trigger phrase, but the clause is not relevant to the current mission `G0`
- **WHEN** Stage-B produces the final group verdict for this mission
- **THEN** Stage-B MUST NOT force `不通过` solely due to that mission-irrelevant trigger hit.

#### Scenario: Mission-relevant noncompliance pattern `不符合要求/<issue>` triggers fail-first without enumerating `<issue>`
- **GIVEN** a per-image summary uses hierarchical attribute phrasing such as `不符合要求/未拧紧` (as seen in `data_conversion/hierarchical_attribute_mapping.json`)
- **AND** the clause is relevant to the current mission `G0`
- **WHEN** Stage-B produces the final group verdict for this mission
- **THEN** the final verdict MUST be `不通过` (fail-first)
- **AND** Stage-B MUST treat any substring matching the pattern `不符合要求/<issue>` as explicit negative evidence even if `<issue>` is unseen (future missions may add new issue descriptors).

### Requirement: Stage-B SHALL treat uncertainty phrases as pending signals, with a safety rule: no evidence for pass ⇒ fail.
Uncertainty phrases (e.g., `无法确认/无法判断/只显示部分/模糊`) MUST NOT be used as hard fail-first triggers; instead, they are pending signals that require the model to search for supporting evidence. If the model cannot provide evidence supporting pass for mission key points, it MUST output `不通过`.
#### Scenario: Uncertainty appears but pass evidence exists elsewhere
- **GIVEN** one image summary contains uncertainty phrasing, but other images and/or remarks provide clear supporting evidence for the mission G0 key points
- **WHEN** the model decides the group verdict
- **THEN** it MAY output `通过` only if the Reason provides explicit supporting evidence
- **AND** it MUST output `不通过` if it cannot provide supporting evidence for pass.

### Requirement: Stage-B prompts SHALL be mission-specific via `G0` only and SHALL NOT introduce a global noun glossary.
Prompts MUST use mission guidance `G0` as the sole source of mission-specific key points and MUST NOT include a hand-maintained noun list that couples prompt behavior to a mission.
#### Scenario: Prompt assembly for the four production missions
- **WHEN** Stage-B builds the system prompt for any of the four missions
- **THEN** it MUST include mission-specific `G0` content (as the “任务要点”)
- **AND** it MUST include the universal negative trigger phrases and remark/uncertainty handling guidance
- **AND** it MUST NOT include a global noun glossary (e.g., enumerating mission objects) beyond what is already present in `G0` and learned guidance snippets.

#### Scenario: Same ticket audited under different missions may yield different verdicts
- **GIVEN** the same group ticket (same Stage-A summaries) is audited under two different missions with different `G0`
- **WHEN** Stage-B produces verdicts per mission
- **THEN** it MUST scope reasoning and fail-first guardrails to the current mission `G0`
- **AND** it MUST allow different missions to produce different verdicts for the same ticket (e.g., pass under one mission and fail under another) based on mission-relevant evidence.

## ADDED Requirements

### Requirement: Reflection/Training SHALL quarantine label/stageA contradictions into a need-review queue and SHALL avoid learning fail rules from noisy evidence.
Need-review MUST exist only for reflection/training governance and MUST NOT affect the inference output protocol. When gt-label contradicts Stage-A per-image summaries (especially `gt=fail` with no explicit negative evidence and model tends to pass), the ticket MUST be added to a `need_review_queue.jsonl` queue (single queue) and MUST NOT be used to learn fail rules.
#### Scenario: gt=fail but no negative evidence in Stage-A summary
- **GIVEN** `gt-label=不通过` for a group, but Stage-A per-image summaries contain no explicit negative trigger evidence
- **AND** Stage-B candidates tend to `通过` (by vote or selected candidate)
- **WHEN** reflection/training evaluates whether to learn guidance updates
- **THEN** the system MUST record this group in `need_review_queue.jsonl` with a reason tag (e.g., `label_noise_suspect` or `stageA_noise_suspect`)
- **AND** it MUST NOT generate or apply “learned fail” guidance operations from this group.

#### Scenario: gt=pass but explicit negative evidence exists in Stage-A summary
- **GIVEN** `gt-label=通过` for a group, but Stage-A per-image summaries contain explicit negative trigger evidence
- **WHEN** reflection/training evaluates whether to learn guidance updates
- **THEN** the system MUST record this group in `need_review_queue.jsonl` with a reason tag (e.g., `label_noise_suspect`)
- **AND** it MUST NOT generate or apply “learned pass” guidance operations from this group.

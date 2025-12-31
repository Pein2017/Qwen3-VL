# Proposal: Refactor Prompt Profiles for Training vs Inference

## Summary
Introduce explicit prompt profiles that separate training prompts (format + task criterion only) from inference prompts (Stage‑A + Stage‑B) that include richer business/domain knowledge. Domain knowledge will be defined in Python dataclasses and composed into inference prompts per dataset (BBU vs RRU), while training remains minimal and domain‑agnostic by default.

## Motivation
- Current prompt construction mixes training and inference concerns in a single module, leading to redundancy and accidental leakage of business rules into training.
- Stage‑A and Stage‑B inference need clearer, domain‑specific knowledge separation (BBU vs RRU) without bloating training prompts.
- A structured, dataclass‑driven prompt layer improves maintainability and makes prompt role selection explicit.

## Scope
- Add prompt profiles with explicit roles: **summary_train_min** (training) and **summary_runtime** (inference).
- Define domain knowledge packs (BBU/RRU) as Python dataclasses and compose them into runtime prompts only.
- Update Stage‑A and Stage‑B inference to select runtime profiles and inject domain knowledge.
- Add config/CLI knobs to choose prompt profiles and domains where applicable.
- Update docs/runbooks to reflect profile selection and domain knowledge usage.

## Out of Scope
- Changing output schemas for Stage‑A JSONL or Stage‑B artifacts (unless required for domain resolution).
- Changing Stage‑B selection, reflection, or scoring logic.
- Re‑authoring business rules; this change only reorganizes how they are injected.

## Success Criteria
- Summary SFT training prompts contain only format + task criterion (no domain priors, no mission rules).
- Stage‑A and Stage‑B inference prompts include domain‑specific knowledge appropriate for BBU vs RRU.
- Prompt composition is centralized and typed via Python dataclasses.
- Documentation reflects the new prompt profiles and role separation.

## Risks
- Prompt drift between training and inference if profiles are mis‑selected.
- Domain resolution ambiguity (mission vs dataset) without a clear mapping.

## Mitigations
- Add validation/guardrails to fail fast on unknown domains.
- Provide defaults: training → summary_train_min, inference → summary_runtime.
- Add tests that assert training prompts exclude domain terms and runtime prompts include them.

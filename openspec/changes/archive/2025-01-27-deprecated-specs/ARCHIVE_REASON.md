# Archive Reason: Deprecated/Conflicting Specs

## Summary
These specs were archived on 2025-01-27 because they reference removed features (CriticEngine) or were never implemented and conflict with current architecture.

## Archived Specs

### 1. `refactor-stageb-line-protocols`
**Status**: Deprecated - References removed CriticEngine

**Reason for Archive**:
- Tasks claim implementation in `src/stage_b/critic/engine.py`, but this file does not exist in the codebase
- Spec mandates CriticEngine with line-based protocols, but CriticEngine was removed (see `update-stageb-reflection-llmfirst`)
- Line-based protocols for reflection were implemented (see `add-stageb-3step-reflection`), but CriticEngine line protocols are not applicable since CriticEngine doesn't exist
- Current implementation uses 3-step reflection with JSON-only protocols, not line-based critic protocols

**Current State**:
- Reflection uses 3-step JSON pipeline (summary → critique → batch update)
- No CriticEngine exists in codebase
- Reflection line parsing exists but is for reflection operations, not critic

### 2. `refactor-stage-b-simple-guidance`
**Status**: Not Implemented - Conflicts with current architecture

**Reason for Archive**:
- All tasks are unchecked (not implemented)
- Conflicts with current 3-step reflection implementation (`add-stageb-3step-reflection`)
- Proposes Evidence_Positive/Negative JSON arrays, but current implementation uses two-line protocol (`refactor-stageb-two-line-protocol`)
- Current architecture already uses simplified prompt-only guidance without CriticEngine
- The goals of this spec are already achieved by other implemented changes:
  - No CriticEngine (achieved by `update-stageb-reflection-llmfirst`)
  - Simplified reflection (achieved by `add-stageb-3step-reflection`)
  - Two-line protocol (achieved by `refactor-stageb-two-line-protocol`)

**Current State**:
- Stage-B uses two-line Verdict/Reason protocol (no evidence arrays)
- Reflection uses 3-step JSON pipeline
- No CriticEngine in codebase
- Guidance updates are simplified and deterministic

## Related Active Specs
- `add-stageb-3step-reflection` - ✅ Implemented (3-step JSON reflection)
- `refactor-stageb-two-line-protocol` - ✅ Implemented (Verdict/Reason protocol)
- `update-stageb-reflection-llmfirst` - ✅ Implemented (removed CriticEngine)
- `deterministic-reflection` - ✅ Implemented (deterministic reflection mode)

## Note on Main Spec
The main spec (`specs/stage-b-training-free/spec.md`) still mandates CriticEngine in some sections, but the codebase does not implement it. This is a known spec drift that should be addressed separately by applying the deltas from `update-stageb-reflection-llmfirst` to the main spec.


# Deprecated Specs Archive - 2025-01-27

This directory contains specs that were archived because they:
1. Reference removed features (CriticEngine)
2. Were never implemented and conflict with current architecture
3. Are superseded by other implemented changes

## Archived Specs

### `refactor-stageb-line-protocols`
**Status**: Deprecated - References removed CriticEngine

See `ARCHIVE_REASON.md` for details.

### `refactor-stage-b-simple-guidance`
**Status**: Not Implemented - Conflicts with current architecture

See `ARCHIVE_REASON.md` for details.

## Current Active Architecture

The current Stage-B implementation uses:
- **Two-line protocol**: `Verdict: 通过|不通过` + `Reason: <text>` (no evidence arrays)
- **3-step reflection**: Summary → Critique → Batch Update (JSON-only)
- **No CriticEngine**: Reflection uses the rollout model directly
- **Deterministic reflection mode**: Available as alternative to LLM reflection

These are implemented by:
- `add-stageb-3step-reflection` ✅
- `refactor-stageb-two-line-protocol` ✅
- `update-stageb-reflection-llmfirst` ✅
- `deterministic-reflection` ✅


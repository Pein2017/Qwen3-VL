# Proposal: Auto Context Router for Codex Skills

## Summary
Introduce an **auto-context-router** skill that immediately loads relevant pipeline docs and procedures for complex feature work (architecture/refactor/algorithm/perf/security/spec). The router uses a maintainable context map and `docs/README.md` as the source of truth for directory->doc routing, supports business/mathematical knowledge loading, preserves manual overrides, and fails safe to manual loading when detection is uncertain.

## Motivation
Current skill activation is manual and ad-hoc, which slows down complex feature work that needs comprehensive codebase context, business knowledge, and algorithmic details. A dedicated router skill with a lightweight mapping enables consistent and scalable context loading while keeping the skill-base and doc-base maintainable.

## Scope
- Add new `auto-context-router` skill with a routing workflow and a small, maintainable context map.
- Use `docs/README.md` as the canonical directory<->doc mapping source.
- Auto-load business knowledge for Stage-B and BBU/RRU work.
- Preserve manual override and fail-safe behavior.
- Keep `stageb-guidance-audit` content in Chinese.

## Non-Goals
- No changes to Codex runtime, authentication, or history formats.
- No full-text indexing or heavy scanning of the repository.

## Risks
- Context bloat if routing is too broad.
- Mapping drift if docs or directories change without updating the context map.

## Success Criteria
- Complex-feature tasks trigger immediate loading of relevant docs and procedures.
- Business and math knowledge files load correctly for Stage-B/BBU/RRU work.
- Manual override works and detection failure falls back to manual loading.
- The mapping remains small, explicit, and easy to update.

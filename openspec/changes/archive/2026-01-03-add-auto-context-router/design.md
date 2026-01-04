# Design: Auto Context Router

## Goals
- Immediately load relevant docs and procedures for complex feature work.
- Keep routing maintainable via a small, explicit context map.
- Preserve manual overrides and a safe fallback when detection is uncertain.

## Inputs
- User request text (including explicit file paths or directory mentions).
- Project doc map in `docs/README.md`.

## Routing Strategy
1. **Detect complex-feature intent** using a small keyword set (refactor, architecture, algorithm, optimization, performance, security, proposal, spec, design, multi-module).
2. **Detect path mentions** (e.g., `src/`, `src/stage_b/`, `configs/`, `scripts/`) and map to docs via the directory map in `docs/README.md`.
3. **Detect domain keywords** and load targeted docs:
   - Stage-B / guidance / verdict / rule_search -> Stage-B runtime docs + business knowledge.
   - Augmentation / geometry / polygon -> data/augmentation + polygon support.
4. **Immediate loading**: once matched, load the mapped docs without confirmation.

## Manual Override and Fail-safe
- If the user explicitly requests manual control (e.g., "do not auto-load", "manual only"), skip auto-loading and ask for the desired doc list.
- If no match is found, load only `docs/README.md` and ask a single clarifying question.

## Maintainability
- Keep the routing logic in the skill, and keep the map in a single file:
  `auto-context-router/references/context_map.yaml`.
- The map references `docs/README.md` as the source of truth for directory->doc routing.
- When new features are added, update `docs/README.md` first, then extend `context_map.yaml`.

## Language Policy
- Keep `stageb-guidance-audit` content in Chinese.
- Use Chinese business knowledge sources for Stage-B tasks.

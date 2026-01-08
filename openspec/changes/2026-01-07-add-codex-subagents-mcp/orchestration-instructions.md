# Codex Subagent Orchestration Guide (Canonical Reference)

This change package originally introduced the orchestration playbook as a change-scoped document.

The canonical, durable reference now lives in project documentation:
- `docs/reference/CODEX_SUBAGENTS_ORCHESTRATION.md`

This change package does not maintain a separate orchestration playbook to avoid redundancy.

Use the reference doc for the authoritative:
- boss–worker workflow model
- A2 optimistic concurrency + git rollback protocol
- worker prompt contract (no recursion, no commits, `modifiedFiles` reporting)
- concurrency defaults (`K_read / K_write`)
- cancellation semantics

Quick checklist:
- `docs/reference/CODEX_SUBAGENTS_ORCHESTRATION_QUICKREF.md`

Boss skill (Codex-internal instructions):
- `.codex/skills/codex-subagent-orchestrator/SKILL.md`
```

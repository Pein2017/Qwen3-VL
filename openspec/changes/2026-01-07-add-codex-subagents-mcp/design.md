# Design: Async Codex Sub-Agents (Option A, Job-Based)

This file is a change-scoped design summary for async Codex sub-agents.

To avoid redundancy and drift, the authoritative documents are:
- Canonical runbook: `docs/reference/CODEX_SUBAGENTS_ORCHESTRATION.md`
- Normative requirements: `openspec/changes/2026-01-07-add-codex-subagents-mcp/specs/codex-mcp-subagents/spec.md`
- Boss instructions (skill): `.codex/skills/codex-subagent-orchestrator/SKILL.md`

## Summary

- Orchestration model: **boss–worker** (main agent orchestrates; workers execute).
- Concurrency primitive: **job semantics** (`codex_spawn` + polling), not “parallel tool calls”.
- Workspace model: **shared worktree**; workers edit files but MUST NOT commit.
- Coordination strategy: **A2** optimistic concurrency + **git rollback** for recovery.
- No recursion: workers MUST NOT spawn sub-agents (prompt-enforced).

## Architecture (3 layers)

Layer 0: Boss main agent
  - Spawns and monitors jobs using MCP job tools
  - Performs git baseline/rollback
  - Synthesizes one final user response

Layer 1: `codex-mcp-server` job manager
  - Spawns worker processes (`codex exec --json`)
  - Buffers events/results and exposes them via job tools

Layer 2: Worker processes (N × `codex exec --json`)
  - Execute assigned task
  - Edit the shared worktree (sandbox-controlled)
  - Report `modifiedFiles` in final output

## Tool Surface (Job Tools)

Workers are spawned and observed via:
- `codex_spawn`, `codex_status`, `codex_events`, `codex_wait_any`, `codex_result`, `codex_cancel`

## Coordination (A2)

The boss uses A2 optimistic concurrency:
1) Establish a clean git baseline (record HEAD)
2) Run workers concurrently (bounded by `K_read / K_write` from the spec)
3) Collect worker-reported `modifiedFiles`
4) Detect conflicts by overlap in `modifiedFiles`
5) On conflict: roll back to baseline and re-run safely (usually sequential for the conflicting tasks)

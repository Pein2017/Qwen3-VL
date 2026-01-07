# Codex Subagents Orchestration (Quick Reference)

Status: Active  
Scope: Fast checklist for orchestrating Codex subagents via MCP.  
Owners: Tooling / DX  
Last updated: 2026-01-07

## Core Mental Model

- One “main agent” Codex session orchestrates multiple subagents.
- Subagents share the same workspace (same worktree/branch).
- Parallelism is achieved by issuing multiple MCP tool calls in a single assistant turn.

## Tool Names (this repo config)

- `mcp__codex-cli-wrapper__codex` (synchronous subagent run; blocks until complete)
- `mcp__codex-cli-wrapper__codex_spawn` (async; returns `jobId` immediately)
- `mcp__codex-cli-wrapper__codex_wait_any` (wait for first completion among jobIds)
- `mcp__codex-cli-wrapper__codex_events` (poll normalized incremental events)
- `mcp__codex-cli-wrapper__codex_result` (final/partial job result)
- `mcp__codex-cli-wrapper__codex_cancel` (cancel running job)

## Default Safe Pattern (Exploration → Writes → Validate)

1) Exploration (parallel, `read-only`)
2) Writes (serialized or non-overlapping, `workspace-write`)
3) Validation (tests/lint)

## Reactive Async Pattern (spawn → wait_any → result)

Use when the main agent must react early (“first-completed-wins”).

```
# Spawn N jobs (prefer parallel calls in one assistant response)
codex_spawn(...) -> jobIdA
codex_spawn(...) -> jobIdB
codex_spawn(...) -> jobIdC

# React to first completion
codex_wait_any({ jobIds: [jobIdA, jobIdB, jobIdC], timeoutMs: 60000 })
  -> { completedJobId }

# Inspect and adapt
codex_result({ jobId: completedJobId })
  -> { exitCode, finalMessage, ... }

# Continue or cancel
codex_wait_any({ jobIds: [remaining...], timeoutMs: 60000 })
codex_cancel({ jobId: jobIdC })
```

## Write Lock Rule (instruction-based)

- Parallel writes to the same file: forbidden
- Parallel writes to different files: allowed if explicitly scoped
- Default to a single writer (main agent) if overlap risk exists

## Common Pitfalls

- Jobs are in-memory: MCP server restart invalidates job IDs.
- Subagents must not call MCP tools (no recursion).
- Use `workspace-write` for file edits; `read-only` blocks writes by design.


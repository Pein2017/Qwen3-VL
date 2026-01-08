# Codex Async Sub-Agents Orchestration (Quick Reference)

Status: Active  
Scope: Fast checklist for orchestrating Codex async sub-agents (boss–worker) via MCP job tools.  
Owners: Tooling / DX  
Last updated: 2026-01-08

## Core Mental Model

- One boss main agent orchestrates multiple worker jobs.
- Concurrency comes from background jobs (worker processes), not from “parallel tool calls” (which may serialize upstream).
- Workers share the same workspace (shared worktree); workers edit files but MUST NOT commit.
- Coordination is A2 (optimistic concurrency) with git-based rollback on conflict.

## Tool Names (this repo config)

The server name is `codex-cli-wrapper`, so tool identifiers are prefixed with `mcp__codex-cli-wrapper__...`.

- `mcp__codex-cli-wrapper__codex_spawn` (async; returns `jobId` immediately)
- `mcp__codex-cli-wrapper__codex_wait_any` (wait for first completion among `jobIds`)
- `mcp__codex-cli-wrapper__codex_events` (poll normalized incremental events; cursor-based)
- `mcp__codex-cli-wrapper__codex_result` (final/partial job result)
- `mcp__codex-cli-wrapper__codex_status` (poll job status)
- `mcp__codex-cli-wrapper__codex_cancel` (cancel running job; SIGTERM default, SIGKILL when `force=true`)

## Defaults (Locked)

- `K_read = 8` (read-only workers; `sandbox="read-only"`)
- `K_write = 2` (write-enabled workers; `sandbox="workspace-write"`)
- Cancellation default: best-effort (SIGTERM)
- Git is required for rollback; workers never commit

## A2 Protocol Summary (Optimistic + Git)

### Pre-spawn (boss)
1) Ensure clean baseline (stash or roll back until `git status --porcelain` is empty).
2) Record baseline HEAD: `git rev-parse HEAD`.

### Execute (boss + workers)
1) Spawn workers via `codex_spawn` (read-only and write workers as needed).
2) Monitor with `codex_wait_any` (and optionally `codex_events` for progress).
3) Collect outputs via `codex_result` after completion.

### Post-completion (boss)
1) For each worker, extract `modifiedFiles` from the worker’s final message (required).
2) Detect conflicts: if the same file is listed by multiple workers → conflict.
3) On conflict: roll back to baseline and re-run conflicting work sequentially (or resolve manually).

Important: per-worker `git status` snapshots are not reliable under concurrent writers. Treat git as the source of truth for rollback and final diff, not per-worker attribution.

## Reactive Pattern (spawn → wait_any → result)

Use when early results should influence follow-up actions.

```
codex_spawn(...) -> jobIdA
codex_spawn(...) -> jobIdB

codex_wait_any({ jobIds: [jobIdA, jobIdB], timeoutMs: 1000 })
  -> { completedJobId }

codex_result({ jobId: completedJobId })
  -> { status, exitCode, finalMessage, ... }
```

## Worker Prompt Contract (minimum)

Role: Sub-agent worker.

Constraints:
- Complete the assigned task only.
- Do NOT spawn sub-agents or call orchestration tools.
- Do NOT create git commits or branches.
- Report modified files in the final response.

## Common Pitfalls

- Jobs are in-memory: MCP server restart invalidates `jobId` values.
- Cancellation can leave partial changes: always run `git status --porcelain` and roll back if needed.
- Missing `modifiedFiles` reporting makes conflict detection unreliable; require it in every worker prompt.

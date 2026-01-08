# Codex Async Sub-Agents Orchestration (Quick Reference)

Status: Active  
Scope: Fast checklist for orchestrating Codex async sub-agents (boss–worker) via MCP job tools.  
Owners: Tooling / DX  
Last updated: 2026-01-08

Canonical runbook:
- `docs/reference/CODEX_SUBAGENTS_ORCHESTRATION.md`

## Mental Model

- One boss main agent orchestrates multiple worker jobs.
- Concurrency is job-based (workers run concurrently after `codex_spawn`), even if tool calls serialize.
- Workers share one worktree; workers edit files but do not commit.
- Coordination is A2: optimistic concurrency + git rollback on conflict.

## Tools (this repo config)

Server name: `codex-cli-wrapper` → tool identifiers start with `mcp__codex-cli-wrapper__...`.

- `mcp__codex-cli-wrapper__codex_spawn` → spawn worker; returns `jobId` immediately
- `mcp__codex-cli-wrapper__codex_wait_any` → wait for first completion among `jobIds`
- `mcp__codex-cli-wrapper__codex_events` → poll incremental events (cursor-based)
- `mcp__codex-cli-wrapper__codex_result` → fetch final/partial job result
- `mcp__codex-cli-wrapper__codex_status` → poll job status
- `mcp__codex-cli-wrapper__codex_cancel` → cancel job (SIGTERM default; SIGKILL when `force=true`)

## Checklist (A2)

1) Preflight (boss): ensure clean baseline (`git status --porcelain`), record HEAD (`git rev-parse HEAD`).
2) Spawn: use `codex_spawn` for workers; follow the canonical defaults for read/write concurrency.
3) Monitor: `codex_wait_any` for completion, `codex_events` for progress if needed.
4) Collect: `codex_result` for each job; require worker to list `modifiedFiles`.
5) Detect conflict: overlap in `modifiedFiles` across jobs → conflict.
6) Recover: on conflict or cancellation, roll back to baseline and re-run safely (often `K_write=1`).

## Common Pitfalls

- Jobs are in-memory: MCP server restart invalidates `jobId` values.
- Missing `modifiedFiles` reporting makes conflict detection unreliable.
- Cancellation mid-write can leave partial changes: always run a git integrity check and roll back if needed.

# Codex Async Sub-Agents Orchestration (Quick Reference)

Status: Active  
Scope: Fast checklist for orchestrating Codex async sub-agents (coordinator + delegated jobs) via MCP job tools.  
Owners: Tooling / DX  
Last updated: 2026-01-09

Canonical runbook:
- `references/CODEX_SUBAGENTS_ORCHESTRATION.md`

Toy drills (safe scratch-only checks):
- `references/CODEX_SUBAGENTS_TOY_DRILLS.md`

## Mental Model

- One main agent coordinates multiple delegated jobs.
- Concurrency is job-based (jobs run concurrently after `codex_spawn`), even if tool calls serialize.
- All jobs share one worktree; jobs may edit files but do not commit.
- Dirty worktrees are OK; prefer chaining follow-up edits rather than rolling back.

## Tools

Tool identifiers vary by MCP server name (prefix). Always trust the tool list from your active Codex session.

Common job-tool function names (conceptually):
- `codex_spawn` → spawn worker; returns `jobId` immediately
- `codex_wait_any` → wait for first completion among `jobIds` (returns `timedOut` + `completedJobId`)
- `codex_events` → poll incremental events (cursor-based)
- `codex_result` → fetch final delegated message (default); use `view:"full"` for status + tails
- `codex_status` → poll job status
- `codex_cancel` → cancel job (SIGTERM default; SIGKILL when `force=true`)

Example tool identifiers when the MCP server name is `codex-cli-wrapper`:
- `mcp__codex-cli-wrapper__codex_spawn`
- `mcp__codex-cli-wrapper__codex_wait_any`
- `mcp__codex-cli-wrapper__codex_events`
- `mcp__codex-cli-wrapper__codex_result`
- `mcp__codex-cli-wrapper__codex_status`
- `mcp__codex-cli-wrapper__codex_cancel`

## Checklist

1) Preflight: record baseline state (dirty OK): `git status --porcelain`, `git diff --name-only`.
2) Spawn: use `codex_spawn` for delegated jobs; keep bounded concurrency (analysis vs edit).
3) Monitor: `codex_wait_any` for completion, `codex_events` for progress if needed.
4) Collect: `codex_result()` for each job (default returns final message); require job to list `modifiedFiles`.
5) Overlap: overlap in `modifiedFiles` ⇒ order-dependent edits; prefer chaining or a reconcile job, not automatic rollback.
6) Recover: on cancellation/crash, run `git status --porcelain` and decide whether to reconcile or roll back.

## Common Pitfalls

- Jobs are in-memory: MCP server restart invalidates `jobId` values.
- Missing `modifiedFiles` reporting makes overlap detection unreliable.
- Cancellation mid-edit can leave partial changes: always run a git integrity check and reconcile/roll back if needed.


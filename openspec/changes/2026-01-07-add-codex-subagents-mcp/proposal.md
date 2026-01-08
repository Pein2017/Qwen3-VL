# Proposal: Async Codex Sub-Agents (Job-Based, Option A)

## Summary
Enable Codex to orchestrate multiple worker “sub-agents” concurrently using **MCP async job primitives** (`codex_spawn` / `codex_events` / `codex_wait_any` / `codex_result` / `codex_cancel`) with a boss–worker pattern.

The boss is the main agent (prompt/skill-driven). The MCP server remains a primitives layer.

## Motivation
Upstream Codex tool execution may serialize MCP tool calls, so “parallel tool calls” are not a reliable concurrency primitive. Concurrency must come from **background job semantics**: workers are separate `codex exec --json` processes that continue running after `codex_spawn` returns.

The local `codex-mcp-server` already provides:
- An async job model (spawn, poll events/status, collect results, cancel)
- Structured event streams (JSONL → normalized events)
- A server-side maximum concurrent job cap

What is required is a **coherent orchestration strategy** and consistent documentation:
- Boss–worker workflow model
- A2 coordination protocol (optimistic concurrency + git rollback)
- Worker prompt contract (no recursion, no commits, `modifiedFiles` reporting)

## Scope
1) Update OpenSpec requirements/scenarios for Option A (job-based)
2) Update canonical orchestration documentation + quick reference
3) Update the boss orchestration skill (Codex-internal instructions)

## Hard Constraints (Locked)
- No upstream Codex changes (`/references/codex` is read-only)
- No recursive sub-agents (worker prompt contract; prompt-only enforcement)
- Shared worktree model; workers do not commit
- Git required; rollback is acceptable
- Coordination strategy A2 (optimistic + git rollback)

## Non-Goals
- No upstream Codex modifications
- No new “orchestration mega-tool” in MCP (boss logic stays in the main agent)
- No hard enforcement mechanism for blocking worker MCP tools (prompt-only is sufficient)
- No per-worker commits or branches

## Risks
- Worker prompt contract violations (missed `modifiedFiles`, scope creep)
- Conflicts between concurrent writers (handled by A2 rollback + re-run)
- Cancellation mid-write leaving partial changes (handled by git integrity checks + rollback)

## Success Criteria
- Happy path: boss can spawn 2+ write-enabled workers editing disjoint files; both complete; no conflict detected.
- Robustness: boss can detect a same-file overlap and roll back to baseline, then re-run safely.

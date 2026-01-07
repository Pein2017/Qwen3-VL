# Codex Subagent Orchestration Guide (Canonical Reference)

This change package originally introduced the orchestration playbook as a change-scoped document.

The canonical, durable reference now lives in project documentation:
- `docs/reference/CODEX_SUBAGENTS_ORCHESTRATION.md`

Use the reference doc for the authoritative 3-layer architecture, tool naming, sandbox rules, write-lock patterns, and example orchestration prompts.

## Pattern 4: Reactive Async Orchestration (Job-Based)

This pattern is recommended when orchestration must be reactive (do not wait for the slowest subagent):

- spawn N subagents immediately (returns `jobId` without blocking)
- react to the first completion (`codex_wait_any`)
- fetch `codex_result` for the completed job and adapt (spawn follow-ups or cancel others)
- optionally poll incremental progress via `codex_events`

### Parallel spawning (single assistant response)

In a single assistant response, issue multiple tool calls in parallel:

```
mcp__codex-cli-wrapper__codex_spawn({ prompt: "task A", sandbox: "workspace-write", workingDirectory: "/data/Qwen3-VL" }) -> jobIdA
mcp__codex-cli-wrapper__codex_spawn({ prompt: "task B", sandbox: "workspace-write", workingDirectory: "/data/Qwen3-VL" }) -> jobIdB
mcp__codex-cli-wrapper__codex_spawn({ prompt: "task C", sandbox: "workspace-write", workingDirectory: "/data/Qwen3-VL" }) -> jobIdC
```

### First-completed-wins loop

```
mcp__codex-cli-wrapper__codex_wait_any({ jobIds: [jobIdA, jobIdB, jobIdC], timeoutMs: 60000 })
  -> { completedJobId }

mcp__codex-cli-wrapper__codex_result({ jobId: completedJobId })
  -> { status, exitCode, finalMessage, stdoutTail, stderrTail }
```

Then either:
- continue waiting for remaining jobs, or
- cancel work that is no longer needed:

```
mcp__codex-cli-wrapper__codex_cancel({ jobId: jobIdC })
```

### Optional: incremental progress via events

```
mcp__codex-cli-wrapper__codex_events({ jobId: jobIdA, cursor: "0", maxEvents: 200 })
  -> { events, nextCursor, done }
```

Persist `nextCursor` and pass it back in subsequent polls.

### Constraints / gotchas

- Jobs are **in-memory** on the MCP server. If the MCP server restarts, jobs are lost.
- Reactive orchestration assumes a single long-lived “main agent” Codex session that continues calling MCP tools over time.
- Apply the existing write-lock guidance: do not allow two write-enabled subagents to modify the same file concurrently.

### Contrast with synchronous parallel pattern

- Synchronous fan-out (using `mcp__codex-cli-wrapper__codex` N times) returns only after each call completes.
- Async job fan-out (using `mcp__codex-cli-wrapper__codex_spawn`) returns immediately with `jobId` and enables reactive “first-completed-wins”.

DON'T USE SUBAGENTS:
  • Trivial reads
  • Single file operations
```

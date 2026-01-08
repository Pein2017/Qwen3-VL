# Codex Sub-Agents: Toy Drills (Stability + UX)

Status: Active  
Scope: Copy/paste drills for validating sub-agent orchestration UX and MCP job-tool behavior without doing “useful” work.  
Last updated: 2026-01-08

This document is intentionally **toy-only**: it should never touch production codepaths or important configs. All edits are confined to a scratch folder that is deleted at the end.

## Drill 1: Sub-Agent Stability Smoke (Workspace-Write + FinalMessage-only)

### What this drill validates

- `codex_spawn` returns immediately and jobs run concurrently.
- `codex_wait_any(timeoutMs: 0)` returns an explicit timeout shape (not `{}`).
- `codex_result` default (finalMessage-only) is sufficient for successful jobs (A–D).
- Chained same-file edits work (“continue from current state”).
- Cancellation works and terminates the job.
- Shared worktree stays clean after cleanup.

### Safety rules

- Do NOT modify any tracked project files.
- Only write under `tmp/subagents_toy/`.
- Do NOT create commits or branches.

### Coordinator prompt (copy/paste)

Paste this as a **single prompt** into a coordinating Codex session:

```
You are running a TOY stability drill for the async sub-agent system.

Goal: verify that sub-agent jobs (spawn/wait/result/cancel) are stable, and that finalMessage-only collection is usable.

Hard rules:
- Do NOT edit any tracked project files outside `tmp/subagents_toy/`.
- Do NOT create commits or branches.
- Use workspace-write (default) and keep the repo safe by only touching `tmp/subagents_toy/`.
- Do NOT spawn any recursive sub-agents from within a sub-agent (prompt discipline only).
- Provide ONE final consolidated report at the end (no incremental user chat).

Preparation (coordinator-only):
1) Ensure `tmp/subagents_toy/` exists.
2) Create a baseline dirty state by writing `tmp/subagents_toy/shared.txt` with the single line: "baseline".
3) Record baseline: `git status --porcelain` (report only; do not attempt to clean it).

Phase A — concurrency happy path (disjoint files):
- Spawn two sub-agent jobs concurrently:
  Job A: create/overwrite `tmp/subagents_toy/a.txt` with:
    - line1: "job=A"
    - line2: "marker=AAA"
  Job B: create/overwrite `tmp/subagents_toy/b.txt` with:
    - line1: "job=B"
    - line2: "marker=BBB"

Each sub-agent prompt MUST include:
- You are a delegated job. Do not orchestrate or spawn sub-agents.
- Do not run git commands that modify history.
- Only write files under tmp/subagents_toy/.
- Final response MUST be only:
  - a short summary
  - modifiedFiles list (one per line)
  - (no extra logs)

Monitoring requirements (coordinator):
- Call codex_wait_any(timeoutMs: 0) at least once early; confirm it returns an explicit timeout shape (not `{}`).
- Avoid busy polling. If nothing finishes, do something minimal (e.g., outline the final report), then check again.

Collection requirements:
- For each completed job, call codex_result in finalMessage-only mode (default).
- Verify `tmp/subagents_toy/a.txt` and `tmp/subagents_toy/b.txt` contents.

Phase B — “continue from current state” (sequential same-file edits):
- Spawn Job C (after A/B are done): append a new line to `tmp/subagents_toy/shared.txt`: "from=C".
  - Job C must re-read the file first and keep existing content.
- Spawn Job D (after C is done): append a new line to `tmp/subagents_toy/shared.txt`: "from=D".
  - Job D must re-read the file first and keep existing content.

Verify final shared.txt contains (in order):
baseline
from=C
from=D

Phase C — cancellation:
- Spawn Job E: do a short acknowledgement message first, then run a long sleep (~120s).
  - Important: ask the delegated job to emit an assistant message BEFORE running any tool commands.
- Optional: poll codex_events once to confirm a message event exists (so lastAgentMessage is non-empty).
- Cancel it using codex_cancel(force=false); if it does not stop quickly, use force=true.
- Report final status for Job E.

Expected: Even if the job is canceled before emitting any agent_message event, `codex_result` (finalMessage-only) SHOULD still return a small cancellation summary (not an empty string).

Cleanup (coordinator-only):
- Remove the scratch dir: rm -rf tmp/subagents_toy/
- Confirm `git status --porcelain` matches the baseline.

Final output (ONE message):
1) List each jobId and whether it succeeded/canceled.
2) Paste the finalMessage from each job (exactly as returned by codex_result).
3) Report whether codex_wait_any ever returned `{}` (it should NOT).
4) Confirm that finalMessage-only output was enough for completed jobs.
5) Ask the operator for feedback using the checklist in this doc.
```

### Delegated job prompt template (copy/paste)

Use this template for each delegated job you spawn:

```
Context: Delegated job (sub-agent). This is a toy drill.

Rules:
- Do NOT call codex_spawn/codex_wait_any/codex_cancel or orchestrate anything.
- Do NOT run git commits/branches/reset/checkout/stash.
- Only modify files under tmp/subagents_toy/.
- Keep output short and structured.
- If you are about to run a long tool command, emit a short assistant message first (this makes cancellation outcomes more informative).

Final response MUST contain only:
Summary:
- <1-3 bullets>
modifiedFiles:
<one path per line>
```

## Operator feedback checklist (copy/paste)

1) `wait_any` signal
- Did `codex_wait_any` ever return `{}`?
- Did it return `{ completedJobId: null, timedOut: true }` when nothing finished?

2) finalMessage-only behavior
- Were sub-agent results readable using only `codex_result` (default finalMessage-only) for completed jobs?
- Did any output appear truncated?

3) workspace-write + safety
- Did any sub-agent accidentally touch files outside `tmp/subagents_toy/`?
- Did any sub-agent try to mutate git history or spawn sub-agents?

4) “continue from current state”
- Did the sequential shared-file edits preserve prior content and append in order?

5) cancellation
- Did cancel work reliably (SIGTERM then SIGKILL if needed)?
- Did the canceled job end as `canceled` (not `failed`)?
- Did `codex_result` (finalMessage-only) return a non-empty cancellation summary for the canceled job?

6) UX / teammate coherence
- Were the prompts and final outputs coherent and teammate-like without role confusion?

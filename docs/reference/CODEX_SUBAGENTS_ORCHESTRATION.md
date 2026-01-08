# Codex Async Sub-Agents Orchestration (MCP Jobs, Option A)

Status: Active  
Scope: Operational + design guidance for async sub-agent orchestration (coordinator + delegated jobs) using `codex-mcp-server` job primitives.  
Owners: Tooling / DX  
Last updated: 2026-01-08

## Context and Goal

### Repositories
- **Current worktree**: `/data/Qwen3-VL`
- **Local MCP server (editable)**: `/data/Qwen3-VL/mcp/codex-mcp-server`
- **Upstream Codex (read-only, no modifications)**: `/data/Qwen3-VL/references/codex`

### Objective
Finalize the async sub-agent mechanism design and document how the main agent coordinates delegated sub-agent jobs using job semantics.

**Scope**: Design refinement + spec + documentation updates. No implementation changes are required to adopt the workflow at the spec level.

## Hard Constraints (Locked)

1) **No upstream Codex modifications** — `/references/codex` is read-only
2) **No recursive sub-agents** — delegated jobs cannot spawn additional delegated jobs
3) **Shared worktree model** — all delegated jobs edit the same workspace; no per-job commits
4) **Default sandbox is `workspace-write`** — “read-only subagents” are not assumed; read-only behavior is prompt-enforced
5) **Dirty worktrees are allowed** — coordination MUST work on top of an existing (dirty) workspace state
6) **Git is recommended** — use git for recovery and for “what changed?” inspection when needed

## Document Map (Avoid Redundancy)

This repository intentionally keeps **one canonical runbook** and treats other documents as thin entrypoints:

- Canonical runbook (this file): `docs/reference/CODEX_SUBAGENTS_ORCHESTRATION.md`
  - Owns: architecture model, coordination protocol, delegated prompt contract, and default concurrency limits.
- Quick reference: `docs/reference/CODEX_SUBAGENTS_ORCHESTRATION_QUICKREF.md`
  - Owns: short checklist and tool list only; MUST NOT re-specify full protocols.
- Toy drills: `docs/reference/CODEX_SUBAGENTS_TOY_DRILLS.md`
  - Owns: copy/paste stability drills + operator feedback checklist; MUST remain toy-only.
- OpenSpec (normative requirements): `openspec/changes/2026-01-07-add-codex-subagents-mcp/specs/codex-mcp-subagents/spec.md`
  - Owns: requirements and scenarios; SHOULD link here for operational detail.
- Boss instructions (skill): `.codex/skills/codex-subagent-orchestrator/SKILL.md`
  - Owns: “how to apply” guidance inside Codex; SHOULD point here rather than duplicating long prose.

## Why Job Semantics (Not “Parallel Tool Calls”)

Upstream Codex has a global tool execution gate that may serialize tool calls, including MCP tool calls:
- Global tool execution gate: `references/codex/codex-rs/core/src/tools/parallel.rs`
- MCP tool registration defaults (parallel support is not assumed): `references/codex/codex-rs/core/src/tools/spec.rs`
- MCP tool call is awaited: `references/codex/codex-rs/core/src/mcp_tool_call.rs`

**Implication**: “Multiple tool calls in one assistant message” MUST NOT be treated as a concurrency primitive.  
**Requirement**: Real concurrency MUST come from background worker jobs that keep running after `codex_spawn` returns.

## Architecture

### Layer Model

Layer 0: Main Agent (Coordinator)
    - Single Codex session
    - Calls MCP tools to coordinate delegated jobs
    - May use git for inspection/recovery (optional)
    - Synthesizes final result for the user
         │ MCP tool calls (codex_spawn, codex_events, codex_wait_any, …)
         ▼
Layer 1: codex-mcp-server (Job Manager)
    - Spawns worker processes
    - Manages job lifecycle
    - Buffers events and results
    - Enforces maximum concurrent jobs (server cap)
         │ child_process.spawn()
         ▼
Layer 2: Worker Processes (N × `codex exec --json`)
    - Independent Codex CLI instances
    - Emit JSONL events to stdout (parsed into normalized events)
    - Edit the shared workspace filesystem (sandbox-controlled)
    - MUST NOT recursively spawn sub-agents

### Grounded Implementation References (local server)
- Tool schemas/constants: `mcp/codex-mcp-server/src/types.ts`
- Tool handlers: `mcp/codex-mcp-server/src/tools/handlers.ts`
- Job lifecycle + event normalization: `mcp/codex-mcp-server/src/jobs/job_manager.ts`

## Existing MCP Tools (Job API)

These tools are the only required interface for orchestration.

| Tool | Purpose |
|------|---------|
| `codex_spawn` | Start worker, return `jobId` immediately |
| `codex_status` | Get job status |
| `codex_result` | Get final delegated message (default); use `view="full"` for status + stdout/stderr tails |
| `codex_events` | Poll incremental events (cursor-based) |
| `codex_wait_any` | Wait for first completion among jobs |
| `codex_cancel` | Cancel running job (SIGTERM default, SIGKILL when `force=true`) |

## Workflow Model: Coordinator + Delegated Jobs

The main agent coordinates delegated jobs as background processes and produces a single consolidated outcome.

1) **Dispatch**: coordinator spawns N delegated jobs via `codex_spawn` (non-blocking)
2) **Monitor**: coordinator polls progress via `codex_events` and/or uses `codex_wait_any`
3) **React**: coordinator cancels jobs or adjusts strategy based on intermediate results
4) **Collect**: coordinator gathers final delegated messages via `codex_result()` (default is final message)
5) **Decide**: coordinator synthesizes results, then either finishes or spawns follow-up delegated jobs

Delegated jobs:
- Execute independently as `codex exec --json` processes
- Edit files in the shared worktree (no commits)
- Report completion/failure to MCP server
- Cannot spawn sub-agents or call orchestration tools

## Concurrency Model (Locked Defaults)

The coordinator MUST apply bounded concurrency to reduce conflict risk.

- `K_read = 8` for analysis-only jobs (prompt-enforced “do not modify files”, even though sandbox is `workspace-write`)
- `K_write = 3` for edit jobs (allowed to modify files under `workspace-write`)

Notes:
- Server-side cap remains authoritative (`CODEX_MCP_MAX_JOBS`, default `32`) in `mcp/codex-mcp-server/src/jobs/job_manager.ts`.
- Under upstream tool serialization, the coordinator should prefer **short tool calls** (`codex_spawn`, `codex_events`, `codex_status`) and avoid long waits.

## Shared Worktree Coordination (Dirty OK, Git-assisted)

This protocol supports dirty worktrees and overlapping intent by treating the current workspace state as the baseline and using git as an optional recovery/inspection tool.

### Pre-Spawn Phase (Coordinator)
1) Record baseline state (dirty is OK):
   - `git status --porcelain`
   - `git diff --name-only`
2) Optionally record baseline HEAD (useful for recovery):
   - `git rev-parse HEAD`

### Execution Phase (Coordinator + Delegated Jobs)
1) Spawn N delegated jobs with `sandbox: "workspace-write"`.
2) For “analysis-only” jobs, enforce read-only behavior via prompt (not via sandbox).
3) Delegated jobs may edit files freely (no commits).
4) Coordinator monitors progress via `codex_events` and/or `codex_wait_any`.

### Post-Completion Phase (Coordinator)
For each completed job:
1) Fetch the final delegated message:
   - `codex_result({ jobId })`
2) Record delegated-job-reported `modifiedFiles` from the final message (required by the Delegated Prompt Contract).
3) Optionally extract file touches from `codex_events` (when `file_change` events are present).

**Important**: Per-job `git status` snapshots are NOT reliable for attributing changes under concurrent writers. Use delegated self-reporting (`modifiedFiles`) as best-effort attribution and git only as an overall “what changed” view.

### Overlap Handling (Coordinator)
If the same file appears in multiple jobs’ `modifiedFiles`, this indicates **order-dependent edits**, not an automatic failure.

Recommended behaviors:
- Avoid concurrently spawning two edit jobs likely to touch the same file. If you need a follow-up edit on the same file, **chain** it: spawn the next job after the previous job finishes so it can “continue from what’s currently in the workspace”.
- If overlapping edits happened concurrently and results look inconsistent, prefer a **reconcile job** that re-reads the current file and produces a coherent final version, rather than rolling back.

Optional recovery (when needed):
- Use git to roll back to the recorded baseline HEAD if a cancellation/crash leaves the workspace inconsistent.

### Failure Recovery (Coordinator)
- On delegated job failure: capture `codex_result` (full view for stderr tail if needed), inspect `git diff --name-only`, and decide whether to keep changes, reconcile, or roll back.
- On crash mid-write: prefer a reconcile job first; roll back only if the workspace is clearly inconsistent.

## Cancellation Semantics (Locked)

- Default cancellation is best-effort (SIGTERM): `codex_cancel({ jobId, force: false })`
- Forced kill is allowed when explicitly required (SIGKILL): `codex_cancel({ jobId, force: true })`

After any cancellation of an edit job, the coordinator MUST treat the workspace as potentially inconsistent and run recovery checks (`git status --porcelain`, `git diff --name-only`) before continuing.

## Delegated Prompt Contract

Every delegated job prompt should include the following contract.

Role: Sub-agent (delegated job).

Constraints:
- Complete the assigned task only.
- Do NOT spawn sub-agents or call orchestration tools.
- Do NOT make git commits; do not rewrite git history (no `reset`, `checkout`, `stash`).
- Work on top of the current workspace state (dirty is OK); do not require a clean baseline.
- Report files modified in the final response (`modifiedFiles` list).

Output format:
- Summary of actions taken
- List of files modified
- Any issues encountered

Enforcement model:
1) Prompt instructions (required; soft enforcement)
2) Coordinator monitoring for violations (recommended): treat unexpected orchestration attempts as a job failure and re-run safely

## Anti-Patterns

- Relying on “parallel tool calls” as a concurrency guarantee.
- Allowing workers to spawn additional workers (recursive sub-agents).
- Allowing workers to create commits/branches (violates shared worktree model).
- Continuing after cancellation without git-based integrity checks.

## Troubleshooting

### “Unknown jobId” / “job not found”
Cause: job state is in-memory in the MCP server process; server restart loses jobs.  
Action: treat as lost; re-spawn work if still needed.

### Overlaps are frequent
Actions:
- Reduce `K_write` from `3` to `1` for the affected run.
- Increase task decomposition granularity (smaller edits per worker).
- Strengthen worker prompts to reduce scope creep and file overlap.

### Dirty working tree prevents safe orchestration
This workflow assumes dirty working trees are OK. If you still need a clean baseline (e.g., for reproducibility), do it as a deliberate coordinator-only step (stash/commit/worktree), not as a hard requirement for spawning.

# Codex Async Sub-Agents Orchestration (MCP Jobs, Option A)

Status: Active  
Scope: Operational + design guidance for async “sub-agent” (boss–worker) orchestration using `codex-mcp-server` job primitives.  
Owners: Tooling / DX  
Last updated: 2026-01-08

## Context and Goal

### Repositories
- **Current worktree**: `/data/Qwen3-VL`
- **Local MCP server (editable)**: `/data/Qwen3-VL/mcp/codex-mcp-server`
- **Upstream Codex (read-only, no modifications)**: `/data/Qwen3-VL/references/codex`

### Objective
Finalize the async sub-agent mechanism design and document how the main agent orchestrates worker sub-agents using job semantics.

**Scope**: Design refinement + spec + documentation updates. No implementation changes are required to adopt the workflow at the spec level.

## Hard Constraints (Locked)

1) **No upstream Codex modifications** — `/references/codex` is read-only
2) **No recursive sub-agents** — workers report to boss only; workers cannot spawn sub-agents
3) **Shared worktree model** — all workers edit the same workspace; no per-worker commits
4) **Git is required** — conflict detection and rollback MUST use git
5) **A2 coordination strategy** — optimistic concurrency with git-based recovery (no lock tools)

## Document Map (Avoid Redundancy)

This repository intentionally keeps **one canonical runbook** and treats other documents as thin entrypoints:

- Canonical runbook (this file): `docs/reference/CODEX_SUBAGENTS_ORCHESTRATION.md`
  - Owns: architecture model, A2 protocol, worker prompt contract, and default concurrency limits.
- Quick reference: `docs/reference/CODEX_SUBAGENTS_ORCHESTRATION_QUICKREF.md`
  - Owns: short checklist and tool list only; MUST NOT re-specify full protocols.
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

Layer 0: Main Agent (Boss)
    - Single Codex session
    - Calls MCP tools to orchestrate workers
    - Performs git checkpoint/rollback
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
| `codex_result` | Get final result + stdout/stderr tails |
| `codex_events` | Poll incremental events (cursor-based) |
| `codex_wait_any` | Wait for first completion among jobs |
| `codex_cancel` | Cancel running job (SIGTERM default, SIGKILL when `force=true`) |

## Workflow Model: Boss–Worker Pattern

The boss main agent orchestrates workers as background jobs and produces a single consolidated outcome.

1) **Dispatch**: Boss spawns N workers via `codex_spawn` (non-blocking)
2) **Monitor**: Boss polls progress via `codex_events` and/or uses `codex_wait_any`
3) **React**: Boss cancels workers or adjusts strategy based on intermediate results
4) **Collect**: Boss gathers final results via `codex_result`
5) **Decide**: Boss synthesizes results, then either finishes or spawns follow-up workers

Workers:
- Execute independently as `codex exec --json` processes
- Edit files in the shared worktree (no commits)
- Report completion/failure to MCP server
- Cannot spawn sub-agents or call orchestration tools

## Concurrency Model (Locked Defaults)

The boss MUST apply bounded concurrency to reduce conflict risk.

- `K_read = 8` for read-only workers (`sandbox="read-only"`)
- `K_write = 3` for write-enabled workers (`sandbox="workspace-write"`)

Notes:
- Server-side cap remains authoritative (`CODEX_MCP_MAX_JOBS`, default `32`) in `mcp/codex-mcp-server/src/jobs/job_manager.ts`.
- Under upstream tool serialization, the boss should prefer **short tool calls** (`codex_spawn`, `codex_events`, `codex_status`) and avoid long waits.

## A2 Coordination Protocol (Optimistic + Git)

This protocol allows parallel edits in a shared worktree with git-based recovery.

### Pre-Spawn Phase (Boss)
1) Ensure a clean working tree, OR stash/rollback to a clean baseline.
2) Record baseline HEAD:
   - `git rev-parse HEAD`
3) Record baseline working tree state:
   - `git status --porcelain`

### Execution Phase (Boss + Workers)
1) Spawn N workers with `sandbox: "workspace-write"` (for edit tasks).
2) Workers edit files freely (no commits).
3) Boss monitors progress via `codex_events` and/or `codex_wait_any`.

### Post-Completion Phase (Boss)
For each completed job:
1) Fetch `codex_result` for `{ jobId }`.
2) Record worker-reported `modifiedFiles` from the worker’s final message (required by the Worker Prompt Contract).
3) Optionally extract file touches from `codex_events` (when `file_change` events are present).

**Important**: Per-job `git status` snapshots are NOT reliable for attributing changes under concurrent writers. Git snapshots remain authoritative for rollback and for the final overall “what changed” view, but per-job attribution should be treated as best-effort.

### Conflict Detection (Boss)
1) Compare `modifiedFiles` across all jobs.
2) If the same file appears in multiple jobs’ `modifiedFiles`, a conflict is detected.

Recommended additional safety check:
- Compare the union of all `modifiedFiles` to the actual working tree diff:
  - `git diff --name-only`
  - Any diff-only file not reported by workers should be treated as a protocol violation or missed attribution.

### Resolution (Boss)
- **No conflict**: accept all changes; optionally commit (boss-only).
- **Conflict detected**:
  1) Roll back to baseline.
     - Prefer: `git restore --source <baseline-head> --worktree --staged .`
     - Or (older syntax): `git checkout -- .`
  2) Re-run conflicting tasks sequentially (reduce `K_write` temporarily to `1`), OR manually resolve and continue.

### Failure Recovery (Boss)
- On worker failure: capture `codex_result` (stderr tail), inspect `git diff --name-only`, then roll back to baseline.
- On worker crash mid-write: roll back to baseline (do not attempt partial recovery unless a narrower scope is proven safe).

## Cancellation Semantics (Locked)

- Default cancellation is best-effort (SIGTERM): `codex_cancel({ jobId, force: false })`
- Forced kill is allowed when explicitly required (SIGKILL): `codex_cancel({ jobId, force: true })`

After any cancellation of a write-enabled worker, the boss MUST treat the workspace as potentially inconsistent and run git-based recovery checks (`git status --porcelain`, `git diff --name-only`) before continuing.

## Worker Prompt Contract

Every worker task prompt should include the following contract.

Role: Sub-agent worker.

Constraints:
- Complete the assigned task only.
- Do NOT spawn sub-agents or call orchestration tools.
- Do NOT make git commits; only edit files.
- Report files modified in the final response.

Output format:
- Summary of actions taken
- List of files modified
- Any issues encountered

Enforcement model:
1) Prompt instructions (required; soft enforcement)
2) Boss monitoring for violations (recommended): treat unexpected orchestration attempts as a worker failure and re-run safely

## Anti-Patterns

- Relying on “parallel tool calls” as a concurrency guarantee.
- Allowing workers to spawn additional workers (recursive sub-agents).
- Allowing workers to create commits/branches (violates shared worktree model).
- Continuing after cancellation without git-based integrity checks.

## Troubleshooting

### “Unknown jobId” / “job not found”
Cause: job state is in-memory in the MCP server process; server restart loses jobs.  
Action: treat as lost; re-spawn work if still needed.

### Conflicts are frequent
Actions:
- Reduce `K_write` from `3` to `1` for the affected run.
- Increase task decomposition granularity (smaller edits per worker).
- Strengthen worker prompts to reduce scope creep and file overlap.

### Dirty working tree prevents safe orchestration
Actions:
- Stash or roll back to a clean baseline before spawning write-enabled workers.
- Use `git status --porcelain` as a preflight gate.

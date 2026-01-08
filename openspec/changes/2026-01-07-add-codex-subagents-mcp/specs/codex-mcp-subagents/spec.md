# Spec Delta: Async Codex Sub-Agents via MCP Jobs (Option A)

This change refines the Codex “sub-agent” mechanism to use **async job semantics** exposed by the local `codex-mcp-server`.

**Scope**: Design refinement + spec updates. No implementation code changes are implied by this delta.

## Context and Goal

### Repositories
- **Current worktree**: `/data/Qwen3-VL`
- **Local MCP server (editable)**: `/data/Qwen3-VL/mcp/codex-mcp-server`
- **Upstream Codex (read-only, no modifications)**: `/data/Qwen3-VL/references/codex`

### Objective
Finalize the async sub-agent mechanism design and update the OpenSpec + documentation.

## Hard Constraints (Locked)

1) **No upstream Codex modifications** — `/references/codex` is read-only
2) **No recursive sub-agents** — workers report to boss only; workers cannot spawn sub-agents
3) **Shared worktree model** — all workers edit the same workspace; no per-worker commits
4) **Git is required** — conflict detection and rollback MUST use git
5) **Coordination strategy A2** — optimistic concurrency with git-based recovery

## Design Decisions (Locked)

| Parameter | Decision | Rationale |
|-----------|----------|-----------|
| Sub-agent MCP access | **Prompt-only: workers MUST NOT call orchestration tools** | Prevents recursive spawning without hard enforcement |
| Coordination strategy | **A2 (Optimistic + Git)** | Simple; no new lock tools needed |
| Per-worker commits | **No** | Boss controls git; workers only edit files |
| Git requirement | **Required** | Essential for conflict detection and rollback |
| Cancellation default | **Best-effort (SIGTERM)** | Allows graceful cleanup; SIGKILL on `force=true` |
| Event buffer | **Unbounded** | External periodic cleanup |
| `K_read / K_write` | **8 / 2** | Balance parallelism vs conflict risk |

## ADDED Requirements

### Requirement: Async Sub-Agent Job Tools
The MCP server MUST expose an async “sub-agent job” API that supports background-worker semantics.

The MCP server MUST expose the following tools (tool names are the MCP tool names, not fully-qualified `mcp__...` identifiers):
- `codex_spawn` → starts a worker job and returns `{ jobId }` immediately
- `codex_status` → returns job status for `{ jobId }`
- `codex_result` → returns final/partial result for `{ jobId }`
- `codex_cancel` → cancels a running `{ jobId }` (SIGTERM default; SIGKILL when `force=true`)
- `codex_events` → returns normalized incremental events for `{ jobId }` (cursor-based)
- `codex_wait_any` → waits until any job completes (helper)

#### Scenario: Boss spawns N workers non-blocking
- GIVEN a boss main agent orchestrating N independent subtasks
- WHEN the boss calls `codex_spawn` N times
- THEN each call returns a `{ jobId }` without waiting for the worker process to complete
- AND the boss can continue orchestration while workers run concurrently

### Requirement: Concurrency Is Job-Based (Not Tool-Call Parallelism)
The system MUST achieve concurrency via **background worker jobs** rather than relying on “parallel tool invocation implies parallel execution”.

#### Scenario: Tool calls may serialize while jobs overlap
- GIVEN a boss main agent that can only execute MCP tool calls sequentially
- WHEN the boss spawns two workers via `codex_spawn`
- THEN the worker processes run concurrently after being spawned
- AND the boss can observe overlap via `codex_status` / `codex_events` without relying on parallel tool calls

### Requirement: Boss–Worker Workflow Model
The system MUST define a boss–worker orchestration model:
- The boss main agent orchestrates via MCP job tools and returns the final consolidated result.
- Workers execute as `codex exec --json` processes and report progress/results via job events.

Boss responsibilities MUST include:
1) Dispatch workers (`codex_spawn`)
2) Monitor (`codex_events`, `codex_status`, and/or `codex_wait_any`)
3) React (cancel, retry, re-scope)
4) Collect (`codex_result`)
5) Decide and synthesize a single final outcome

Worker responsibilities MUST include:
- Execute the assigned task only
- Edit files in the shared worktree when permitted by `sandbox`
- NEVER make git commits
- Report modified files in the final response

#### Scenario: Boss orchestrates and synthesizes one final result
- GIVEN a user request that is decomposed into N worker tasks
- WHEN the boss spawns N workers with `codex_spawn`
- AND the boss monitors progress via `codex_events` and/or `codex_wait_any`
- THEN the boss collects each final output via `codex_result`
- AND the boss produces one consolidated response after the required workers complete

### Requirement: No Recursive Sub-Agents
Workers MUST NOT spawn sub-agents and MUST NOT call orchestration tools.

This MUST be enforced by worker prompt constraints.

#### Scenario: Worker refuses recursive spawning
- GIVEN a worker process executing a task under the Worker Prompt Contract
- WHEN the worker determines that spawning another worker would be helpful
- THEN the worker MUST NOT spawn a sub-agent
- AND the worker reports the limitation and recommendations to the boss

### Requirement: Shared Worktree Model (No Per-Worker Commits)
All workers MUST operate in the same shared workspace directory.

- Workers MAY edit files when spawned with `sandbox="workspace-write"`.
- Workers MUST NOT create per-worker commits or branches.
- Git operations for checkpointing, rollback, and final commit (if desired) MUST be performed by the boss.

#### Scenario: Workers edit without committing
- GIVEN a clean git working tree at the start of an orchestration run
- WHEN multiple workers are spawned with `sandbox="workspace-write"`
- THEN workers may modify files in the shared workspace
- AND no worker creates git commits
- AND the boss remains the only actor permitted to checkpoint or rollback via git

### Requirement: A2 Coordination Protocol (Optimistic + Git)
The system MUST use git-based recovery for optimistic parallel writes.

The boss MUST implement the following phases:

**Pre-spawn**
1) Ensure a clean working tree, or stash/rollback to a known clean baseline
2) Record baseline HEAD (`git rev-parse HEAD`)
3) Record baseline working tree state (`git status --porcelain`)

**Execution**
1) Spawn N workers (writes use `sandbox="workspace-write"`)
2) Monitor workers via `codex_wait_any` and/or polling

**Post-completion (per job)**
1) Collect `codex_result`
2) Determine `JobResult.modifiedFiles` using the worker’s final report and/or job event stream

**Conflict detection**
- If the same file appears in multiple `JobResult.modifiedFiles`, a conflict MUST be flagged.

**Resolution**
- On conflict, the boss MUST either:
  - rollback and re-run conflicting work sequentially, OR
  - manually resolve and continue

#### Scenario: No-conflict parallel edits are accepted
- GIVEN two write-enabled workers running concurrently
- WHEN each worker modifies a disjoint set of files
- THEN the boss detects no overlap in `modifiedFiles`
- AND the boss accepts the combined working tree changes

#### Scenario: Conflict triggers rollback and sequential re-run
- GIVEN two write-enabled workers running concurrently
- WHEN both workers modify the same file
- THEN the boss detects a conflict via overlapping `modifiedFiles`
- AND the boss rolls back to the baseline using git-based recovery
- AND the boss re-runs the conflicting tasks sequentially or resolves manually

### Requirement: Bounded Concurrency (`K_read / K_write`)
The boss MUST enforce bounded concurrency across worker jobs:
- Default `K_read = 8` (read-only workers)
- Default `K_write = 2` (write-enabled workers)

The boss MUST NOT exceed the MCP server’s configured maximum concurrent jobs.

#### Scenario: Boss throttles worker spawning
- GIVEN a queue of 20 read-only worker tasks and 5 write-enabled worker tasks
- WHEN orchestration begins
- THEN no more than 8 read-only workers are running at once
- AND no more than 2 write-enabled workers are running at once
- AND remaining tasks remain queued until capacity is available

### Requirement: Cancellation Semantics
The system MUST support cancellation of running workers:
- Default cancellation MUST be best-effort (SIGTERM).
- Forced cancellation MUST be available when explicitly requested (`force=true`, SIGKILL).

#### Scenario: Best-effort cancellation
- GIVEN a running worker job
- WHEN the boss calls `codex_cancel` with `force=false`
- THEN the worker receives a best-effort termination request
- AND the boss MUST perform git-based integrity checks and recovery if required

#### Scenario: Forced cancellation
- GIVEN a running worker job that does not terminate on best-effort cancellation
- WHEN the boss calls `codex_cancel` with `force=true`
- THEN the worker process is force-killed
- AND the boss MUST treat the workspace as potentially inconsistent and apply git-based recovery

### Requirement: Event Streaming (Cursor-Based) and Buffering
The `codex_events` tool MUST return incremental events via an opaque cursor mechanism.

Each event MUST include:
- `type`: `"message" | "progress" | "tool_call" | "tool_result" | "error" | "final"`
- `content`: event-specific payload
- `timestamp`: ISO 8601 string

Event buffering MUST be unbounded (no fixed cap), with cleanup handled operationally.

#### Scenario: Cursor-based polling
- GIVEN a running worker job
- WHEN the boss calls `codex_events` with a cursor
- THEN the response contains `events`, `nextCursor`, and a completion indicator
- AND the boss can resume polling using `nextCursor` without re-reading old events

### Requirement: Orchestration Instruction Document
The system MUST provide an instruction document describing the boss–worker model and A2 coordination protocol.

#### Scenario: Boss follows documented orchestration patterns
- GIVEN a boss main agent with access to the job-based MCP tools
- WHEN the orchestration instructions are available as a canonical reference
- THEN the boss can apply the documented dispatch/monitor/react/collect/decide workflow
- AND the boss can apply the documented git-based conflict detection and rollback procedures

Implementation note: the canonical instruction document is `docs/reference/CODEX_SUBAGENTS_ORCHESTRATION.md`.

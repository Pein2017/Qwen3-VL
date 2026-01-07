# Spec Delta: Codex Subagent Orchestration

## ADDED Requirements

### Requirement: Orchestration Instruction Document
The system MUST provide a comprehensive instruction document that teaches the Codex main agent how to orchestrate subagents using the existing Codex MCP tool `mcp__codex-cli-wrapper__codex`.

#### Scenario: Main agent learns orchestration patterns
- GIVEN a Codex main agent with access to the `codex` MCP tool
- WHEN the orchestration instructions are included in the system prompt or profile
- THEN the main agent understands when and how to spawn parallel subagents
- AND the main agent follows documented patterns and avoids anti-patterns

Implementation note: the canonical instruction document is `docs/reference/CODEX_SUBAGENTS_ORCHESTRATION.md`.

### Requirement: Parallel Tool Invocation Support
The system MUST support the main agent making multiple MCP tool calls (for example `mcp__codex-cli-wrapper__codex`) in a single message for parallel execution.

#### Scenario: Fan-out to N independent subtasks
- WHEN the main agent issues N `mcp__codex-cli-wrapper__codex` tool calls in one message
- THEN the MCP server processes all N calls concurrently
- THEN each call spawns a separate `codex exec` process
- THEN results return as N separate tool responses

### Requirement: Shared Workspace Coordination
The system MUST support subagents operating in a shared working directory with documented coordination patterns.

#### Scenario: Multiple subagents in same workspace
- WHEN multiple subagents are spawned with the same `workingDirectory`
- THEN all subagents can read from the shared workspace
- THEN write operations follow documented serialization patterns
- THEN the main agent is responsible for conflict avoidance

### Requirement: Sandbox-Based Permission Control
The system MUST leverage the existing `sandbox` parameter for subagent permission control.

#### Scenario: Read-only exploration subagents
- WHEN subagents are spawned with `sandbox="read-only"`
- THEN subagents can explore and analyze but cannot modify files
- THEN parallel execution is safe without conflict risk

#### Scenario: Write-enabled subagents
- WHEN subagents are spawned with `sandbox="workspace-write"`
- THEN subagents can modify files in the workspace
- THEN main agent ensures non-overlapping file targets

### Requirement: Result Synthesis Guidance
The instruction document MUST include guidance on how the main agent should synthesize results from multiple subagents.

#### Scenario: Combining subagent outputs
- WHEN all subagents complete and return results
- THEN the main agent collects all responses
- THEN the main agent validates for errors or conflicts
- THEN the main agent synthesizes a unified response for the user

### Requirement: Async Subagent Job Tools
The system MUST provide an async “subagent job” MCP API to enable reactive orchestration without blocking on the slowest subagent.

The MCP server MUST expose the following tools (tool names are the MCP tool names, not the fully-qualified `mcp__...` identifiers):

- `codex_spawn` → starts a job and returns `{ jobId }` immediately
- `codex_status` → returns job status for `{ jobId }`
- `codex_result` → returns final/partial result for `{ jobId }`
- `codex_cancel` → cancels a running `{ jobId }`
- `codex_events` → returns normalized incremental events for `{ jobId }`
- `codex_wait_any` → (optional helper) waits until any job completes

#### Scenario: Reactive orchestration (don’t wait for slowest)
- GIVEN a main agent orchestrating multiple independent subtasks
- WHEN the main agent calls `codex_spawn` multiple times
- THEN the main agent receives `jobId` values immediately
- AND the main agent can continue working while jobs run
- WHEN one job finishes earlier than others
- THEN the main agent can poll `codex_status` / `codex_events` and react (e.g., spawn follow-up jobs)

### Requirement: Async Job Implementation (codex exec --json)
Async subagent jobs MUST be implemented by spawning `codex exec --json` processes and parsing JSONL output into normalized events.

#### Scenario: Structured events via JSONL
- WHEN `codex_spawn` starts a subagent
- THEN the MCP server runs `codex exec --json ...`
- THEN the MCP server parses JSONL thread events
- THEN `codex_events` can return incremental updates without waiting for process completion

### Requirement: Normalized Event Format
The `codex_events` tool MUST return events normalized into a stable shape for easy orchestration.

Each event MUST include:
- `type`: `"message" | "progress" | "tool_call" | "tool_result" | "error" | "final"`
- `content`: an event-specific payload
- `timestamp`: ISO 8601 string

#### Scenario: Main agent consumes normalized events
- WHEN a main agent polls `codex_events` with a `cursor`
- THEN the response contains `events` and `nextCursor`
- THEN the main agent can drive orchestration logic using only `type` + `content`

### Requirement: In-Memory Job State
Async job state MUST be in-memory only (ephemeral). If the MCP server restarts, jobs are lost.

#### Scenario: Server restarts during a job
- GIVEN a running `jobId`
- WHEN the MCP server restarts
- THEN the job cannot be resumed and MUST be treated as lost
- AND the main agent should re-spawn the job if needed

### Requirement: Concurrency Safety Valve
The MCP server MUST enforce a maximum number of concurrently running async subagent jobs.

- Default limit: 32 running jobs
- Configurable via `CODEX_MCP_MAX_JOBS` environment variable

#### Scenario: Spawn limit reached
- GIVEN the server already has 32 running jobs
- WHEN a caller invokes `codex_spawn`
- THEN the MCP server rejects the request with an error
- AND includes guidance to raise `CODEX_MCP_MAX_JOBS` when appropriate

## OPTIONAL Enhancements

### Enhancement: Git Snapshot Parameter
The `codex` MCP tool MAY add an optional `includeGitSnapshot` parameter.

#### Scenario: Capture repository state after execution
- WHEN `includeGitSnapshot=true` is passed to the `codex` tool
- THEN the tool appends `git status --porcelain` output to response metadata
- THEN the tool appends `git diff` output to response metadata
- THEN the main agent can review cumulative changes

## REMOVED from Original Spec

The following items from the original proposal are explicitly NOT required:

- ~~Per-subagent diff attribution~~ → Not reliably possible in shared workspace

Note: the original proposal intentionally avoided adding new tools. This change now explicitly ADDS async job tools to enable reactive orchestration and avoid client-side tool-call timeouts.

## Clarifications (Implementation Guidance)

### Verified: Parallel Execution Support
**Status**: CONFIRMED via smoke test in Codex CLI (single main agent session).

Codex CLI supports multiple MCP tool calls in a single assistant response. These calls execute concurrently by the MCP server; each call spawns a separate `codex exec` process.

### Tool Naming Convention
**Verified tool identifiers (this repo config)**:
- `mcp__codex-cli-wrapper__codex` - main subagent tool
- `mcp__codex-cli-wrapper__ping` - connectivity test
- `mcp__codex-cli-wrapper__codex_spawn` - async job spawn
- `mcp__codex-cli-wrapper__codex_status` - async job status
- `mcp__codex-cli-wrapper__codex_result` - async job result
- `mcp__codex-cli-wrapper__codex_cancel` - async job cancel
- `mcp__codex-cli-wrapper__codex_events` - async job events
- `mcp__codex-cli-wrapper__codex_wait_any` - async helper

Server config in `~/.codex/config.toml`:
```toml
[mcp_servers.codex-cli-wrapper]
command = "node"
args = ["/path/to/mcp/codex-mcp-server/dist/index.js"]
```

### Verified Architecture (Clean 3-Layer Model)

- Main agent CAN call MCP tools directly in a single Codex session (no shell wrapper needed).
- Multiple MCP calls in one assistant message DO execute in parallel.
- For async orchestration, the intended subprocess layer is: the MCP server spawning `codex exec --json` subagent processes and exposing them as `{ jobId }` jobs.

### Q1: Write Lock Mode
**Decision**: YES, enforce via instructions (not MCP code).

- Parallel read-only subagents: ALLOWED (unlimited concurrency)
- Parallel write subagents to DIFFERENT files: ALLOWED
- Parallel write subagents to SAME file: FORBIDDEN (serialize via phased execution)

The main agent instructions enforce this pattern, not the MCP server.

### Q2: Default Sandbox Settings
**Decision**: Safer defaults, escalate only when needed.

| Subagent Type | Default Sandbox | When to Escalate |
|---------------|-----------------|------------------|
| Exploration | `read-only` | Never |
| Analysis/Review | `read-only` | Never |
| File editing | `workspace-write` | Only for edit tasks |
| System commands | `workspace-write` | Only when explicitly needed |
| Full access | NEVER default | Only on explicit user request |

### Q3: Orchestration Trigger
**Decision**: Proactive for qualifying tasks.

The main agent SHOULD proactively use subagent orchestration when:
- Task involves 3+ independent areas/modules
- Task is naturally parallelizable (e.g., "analyze X, Y, and Z")
- Large codebase exploration is needed
- Multiple specialist perspectives benefit the task

The main agent SHOULD NOT require explicit keywords like "parallelize" or "subagents" from the user. Good orchestration is transparent to the user.

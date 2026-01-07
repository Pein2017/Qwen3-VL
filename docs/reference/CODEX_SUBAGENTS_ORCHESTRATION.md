# Codex Subagents Orchestration (MCP)

Status: Active  
Scope: Operational guidance for orchestrating parallel Codex “subagents” using the Codex MCP server.  
Owners: Tooling / DX  
Last updated: 2026-01-07

## Goal

Enable a single Codex “main agent” session to delegate multiple independent subtasks to multiple “subagents” concurrently, then synthesize results.

This mirrors Claude Code’s subagent workflow: the power comes from *instructions and orchestration patterns*, not from a new concurrency primitive.

## Verified Architecture (3 Layers)

Layer 0: Main agent (single Codex session)
    ↓ (one assistant message containing N tool calls)
Layer 1: Parallel MCP tool calls
    - `mcp__codex-cli-wrapper__codex` × N
    ↓
Layer 2: Subagent processes
    - N separate `codex exec ...` processes spawned by the MCP server

Notes:
- The main agent calls MCP tools directly in the same Codex session.
- Each MCP call spawns exactly one Codex CLI subprocess for that subtask.
- No shell wrapper pattern is required in normal Codex usage.

## Tool Naming (Verified)

The exact callable tool identifiers depend on the MCP server name configured in `~/.codex/config.toml`.

In this repository’s configuration, the server is named `codex-cli-wrapper`, and the callable tool identifiers are:
- `mcp__codex-cli-wrapper__codex` (spawn a subagent run)
- `mcp__codex-cli-wrapper__ping` (connectivity test)
- `mcp__codex-cli-wrapper__codex_spawn` (spawn async subagent job; returns `jobId` immediately)
- `mcp__codex-cli-wrapper__codex_status` (poll job status)
- `mcp__codex-cli-wrapper__codex_result` (get final/partial result, including stdout/stderr tails)
- `mcp__codex-cli-wrapper__codex_events` (poll normalized incremental events)
- `mcp__codex-cli-wrapper__codex_wait_any` (wait for first completion among jobs)
- `mcp__codex-cli-wrapper__codex_cancel` (cancel running job)

If the server name changes, the tool prefix changes accordingly (pattern: `mcp__<server-name>__<tool>`).

Note: Some Codex environments may accept underscore variants (e.g., `mcp__codex_cli_wrapper__codex`) as aliases. Prefer the canonical identifiers from the MCP tool listing in the active Codex session.

## Core Principle

```
ONE MESSAGE → MULTIPLE TOOL CALLS → PARALLEL EXECUTION
```

Parallelism is achieved by issuing multiple MCP tool calls in a single assistant turn.

## When to Use Subagents

Use subagents when:
- The user request spans 3+ independent areas (directories/modules/concerns).
- Multiple “specialist perspectives” help (security, performance, tests, style).
- Broad exploration is needed before a write phase.

Avoid subagents when:
- The task is small and sequential.
- A single-file edit is needed (do it in the main agent).
- Tasks are tightly dependent (chain sequentially instead).

## Sandbox Rules (Safety Baseline)

| Task Type | Sandbox |
|----------|---------|
| Exploration / analysis / review | `read-only` |
| Editing files | `workspace-write` |
| Running tests | `workspace-write` |

Never default subagents to `danger-full-access`. Only use it on explicit user request with clear justification.

### Default subagent sandbox (optional)

To avoid repeating `sandbox="workspace-write"` on every subagent call, configure the MCP server with:
- `CODEX_MCP_DEFAULT_SANDBOX=workspace-write`

When configured, the MCP `codex` tool uses this value when `sandbox` is omitted.

## Inheriting Model / Reasoning Defaults (Config-Driven)

To keep main agent and subagents consistent:
- Omit `model` unless an override is explicitly required.
- Omit `reasoningEffort` unless an override is explicitly required.

This allows `~/.codex/config.toml` (and any configured profile) to determine model and reasoning settings uniformly across main agent and subagents.

## Write Coordination (Instruction-Based “Write Lock”)

Because subagents share the same worktree/branch:
- Parallel read-only subagents are safe.
- Parallel writes to *different files* can be acceptable.
- Parallel writes to the *same file* are forbidden (serialize those edits).

This is enforced by main-agent instructions (not by the MCP server).

## Subagent Prompt Header (Recommended)

For each subagent call, prepend a strict header to reduce risk:

```
Role: Subagent worker.
Constraints:
- Do not call any MCP tools.
- Do not spawn further subagents.
- Stay within the assigned scope only.
- If asked to edit, only touch the explicitly named files.
Output format:
- Findings (bullets)
- Proposed changes (bullets)
- If edits were made: list edited files
```

## Orchestration Patterns

### Pattern A: Parallel exploration (recommended default)

Issue multiple `mcp__codex-cli-wrapper__codex` calls with:
- `sandbox="read-only"`
- shared `workingDirectory` (repo root)
- small, non-overlapping scopes (e.g., `src/auth`, `src/api`, `tests/`)

Then synthesize results in the main agent.

### Pattern B: Phased editing (safe writes)

Phase 1 (parallel exploration, read-only): gather facts.

Phase 2 (writes, workspace-write): ensure each subagent targets different files, or serialize same-file edits.

Phase 3 (validation): run tests/lint as a dedicated subtask (often parallelizable with other read-only checks).

### Pattern C: Specialist delegation

Fan out “review lenses”:
- Security review (read-only)
- Performance review (read-only)
- Style/maintainability review (read-only)
- Tests/coverage audit (read-only)

Synthesize into one coherent report.

### Pattern D: Reactive async orchestration (job-based, “first-completed-wins”)

Use async jobs when the main agent must:
- keep working while subagents run
- react as soon as any subagent finishes (without waiting for the slowest)

This pattern uses:
- `codex_spawn` for immediate fan-out
- `codex_wait_any` for first-completed detection
- `codex_result` for final output
- `codex_events` for incremental progress (optional)
- `codex_cancel` to stop no-longer-needed work (optional)

Important constraint: job state is **in-memory** in the MCP server process. If the MCP server restarts, all jobs are lost.

#### Minimal workflow: spawn → wait_any → result

```
# Step 1: Spawn N subagents (prefer parallel tool calls in a single assistant response)
mcp__codex-cli-wrapper__codex_spawn({ prompt: "task A", sandbox: "workspace-write", workingDirectory: "/data/Qwen3-VL" }) -> jobIdA
mcp__codex-cli-wrapper__codex_spawn({ prompt: "task B", sandbox: "workspace-write", workingDirectory: "/data/Qwen3-VL" }) -> jobIdB
mcp__codex-cli-wrapper__codex_spawn({ prompt: "task C", sandbox: "workspace-write", workingDirectory: "/data/Qwen3-VL" }) -> jobIdC

# Step 2: Wait for first completion
mcp__codex-cli-wrapper__codex_wait_any({ jobIds: [jobIdA, jobIdB, jobIdC], timeoutMs: 60000 })
  -> { completedJobId }

# Step 3: Inspect result and adapt
mcp__codex-cli-wrapper__codex_result({ jobId: completedJobId })
  -> { status, exitCode, finalMessage, stdoutTail, stderrTail }

# Step 4: Continue for remaining jobs (or cancel)
mcp__codex-cli-wrapper__codex_wait_any({ jobIds: [jobIdA, jobIdC], timeoutMs: 60000 })
mcp__codex-cli-wrapper__codex_cancel({ jobId: jobIdC })
```

#### Optional: incremental progress via events

```
mcp__codex-cli-wrapper__codex_events({ jobId: jobIdA, cursor: "0", maxEvents: 200 })
  -> { events, nextCursor, done }
```

Persist `nextCursor` and pass it back on subsequent polls.

## Example Prompts (Main Agent)

These prompts are designed for the human user to send to the main agent.

### Example 1: Parallel exploration + synthesis (read-only)

“Analyze `src/stage_a`, `src/stage_b`, and `scripts/` in parallel. Use read-only subagents for exploration only. Return a combined summary with the top 10 findings and next steps.”

### Example 2: Reactive async orchestration (first-completed-wins)

“Spawn three async subagents via `codex_spawn` to:
1) locate where Stage-B loads guidance rules
2) locate where Stage-A writes summary JSONL
3) locate scripts that run Stage-A then Stage-B

Use `codex_wait_any` to react to whichever finishes first, and then decide if a follow-up subagent is needed. Use `codex_result` for completed jobs, and cancel any job that becomes unnecessary.”

### Example 3: Safe phased writes (avoid same-file conflicts)

“Phase 1: spawn read-only subagents to propose changes to `docs/reference/CODEX_SUBAGENTS_ORCHESTRATION.md` and `docs/reference/README.md`. Phase 2: apply edits in the main agent (single writer).”

## Anti-Patterns

- Recursive orchestration: subagents calling MCP tools or spawning subagents.
- Parallel writes to the same file (race conditions, conflicts, nondeterminism).
- Using `danger-full-access` by default (should only happen on explicit request).
- Using async job IDs across MCP server restarts (job IDs become invalid).

## Troubleshooting

### “Unknown jobId” / “job not found”
Cause: the MCP server process restarted (or orchestration is occurring across multiple independent MCP client sessions).

Action:
- treat jobs as ephemeral; re-spawn if needed
- keep orchestration within a single long-lived Codex “main agent” session

### `codex_spawn` rejects with “Too many concurrent jobs”
Cause: concurrency cap reached.

Action:
- wait for existing jobs to complete, or cancel non-essential jobs
- raise the cap via `CODEX_MCP_MAX_JOBS` when appropriate

### Writes are blocked / sandbox denies
Cause: subagent sandbox mode is insufficient (e.g., `read-only`).

Action:
- use `sandbox="workspace-write"` for file creation/editing subagents
- ensure `CODEX_MCP_DEFAULT_SANDBOX=workspace-write` is set if consistent write capability is desired

## Result Synthesis Checklist

After all subagents return:
- Validate each output for errors and scope violations.
- Detect overlap/conflicts (multiple subagents claiming the same file).
- Combine results into a single coherent answer.
- Propose the next action (apply patch, run tests, etc.).

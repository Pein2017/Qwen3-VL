---
name: codex-subagent-orchestrator
description: Orchestrate async Codex worker jobs via MCP job tools (codex_spawn/codex_wait_any/codex_events/codex_result/codex_cancel) using a boss–worker pattern with A2 optimistic concurrency and git rollback in a shared worktree.
---

# Codex Sub-Agent Orchestrator (Job-Based, Option 1)

This skill provides a job-based orchestration playbook so a single Codex “boss” agent can delegate work to multiple “worker” sub-agents using MCP **async job primitives**.

**Option selection**: Boss logic lives in the main agent via skill/prompt instructions (MCP server remains a primitives layer).

**Hard constraints (locked)**:
- No upstream Codex modifications (reference-only in `/references/codex`).
- No recursive sub-agents: workers MUST NOT spawn workers.
- Shared worktree model: workers edit the same workspace; workers MUST NOT create git commits.
- Git is REQUIRED for conflict detection and rollback.
- Coordination is A2: optimistic concurrency + git-based recovery.

## Tool Naming (Verified)

In this repository, the MCP server name is `codex-cli-wrapper`, and callable tool identifiers preserve the hyphenated server name.

Job-based tool identifiers:
- `mcp__codex-cli-wrapper__codex_spawn` (async worker spawn; returns `jobId` immediately)
- `mcp__codex-cli-wrapper__codex_status` (poll job status)
- `mcp__codex-cli-wrapper__codex_events` (poll normalized incremental events; cursor-based)
- `mcp__codex-cli-wrapper__codex_wait_any` (wait for first completion among `jobIds`)
- `mcp__codex-cli-wrapper__codex_result` (final/partial job result)
- `mcp__codex-cli-wrapper__codex_cancel` (cancel running job; SIGTERM default, SIGKILL on `force=true`)

If the MCP server is renamed in `~/.codex/config.toml`, the tool prefix changes accordingly.

Note: Some Codex environments may accept underscore variants as aliases (e.g., `mcp__codex_cli_wrapper__codex_spawn`). Prefer the canonical identifiers from the MCP tool listing in the active Codex session.

## Core Idea (Job Semantics)

- A “worker” is an independent `codex exec --json` process spawned by the MCP server.
- Concurrency comes from background worker jobs that continue running after `codex_spawn` returns.
- The boss MUST NOT rely on “parallel tool calls” as a concurrency guarantee; upstream tool execution may serialize.
- All workers operate in a shared worktree by default; safety relies on A2 optimistic concurrency + git rollback.

The canonical orchestration guide lives at:
- `docs/reference/CODEX_SUBAGENTS_ORCHESTRATION.md`

Read it when orchestration is needed, and follow it strictly.

## When This Skill Should Be Applied (Proactive Trigger)

Proactively use subagent orchestration when:
- The request spans 3+ independent areas (modules, directories, concerns).
- The request benefits from multiple specialist perspectives (security/perf/style/test).
- The request requires broad codebase exploration before editing.

Avoid subagents for:
- trivial reads (single file, single grep)
- tasks with tight sequential dependencies
- tasks that require a single cohesive edit in one file

## Concurrency Defaults (Locked)

- Default `K_read = 8` for exploration/review workers (`sandbox="read-only"`).
- Default `K_write = 2` for edit workers (`sandbox="workspace-write"`).

If conflict frequency is high, temporarily reduce `K_write` to `1`.

## Default Permissions (Sandbox)

To avoid repeating `sandbox` on every `codex_spawn`, configure MCP server environment:

- `CODEX_MCP_DEFAULT_SANDBOX=workspace-write`

With this set, worker spawns may omit `sandbox` and still run with `workspace-write` consistently.

Safety notes:
- Never default to `danger-full-access` for subagents; only use it on explicit user request and with clear justification.
- Avoid `fullAuto` when explicitly setting `sandbox`; `fullAuto` is a convenience alias that implies `sandbox="workspace-write"` in Codex CLI.

## Inheriting Model / Reasoning Defaults (Config-Driven)

To ensure the main agent and subagents behave identically, prefer inheriting from `~/.codex/config.toml`:
- Do not set `model` in subagent calls unless an override is explicitly required.
- Do not set `reasoningEffort` in subagent calls unless an override is explicitly required.

## A2 Coordination Protocol (Optimistic + Git Rollback)

This skill uses A2 coordination (no lock tools):
- Workers may edit freely (within prompt-scoped intent).
- The boss detects conflicts after completion.
- Git provides rollback to a known baseline.

Boss MUST follow this protocol for any write-enabled workers:

**Pre-spawn**
1) Ensure a clean working tree (or stash/rollback to a clean baseline).
2) Record baseline HEAD: `git rev-parse HEAD`.
3) Record baseline status: `git status --porcelain` (must be empty after baseline preparation).

**Execution**
1) Spawn workers via `codex_spawn` (read-only and write workers separately).
2) Monitor via `codex_wait_any` and/or polling `codex_events`.
3) Collect via `codex_result` when complete.

**Post-completion**
1) Require each worker to report `modifiedFiles` in its final response.
2) Detect conflicts by overlap in `modifiedFiles` across jobs.
3) If conflict detected:
   - Roll back to baseline (git restore / checkout), then re-run conflicting tasks sequentially.

Important: Per-worker `git status` snapshots are not reliable under concurrent writers; attribution MUST come from worker self-report (and optionally job event streams).

## Tool Call Template (Subagent Prompt Header)

For every subagent call, prepend a strict header to the subagent prompt to avoid recursion and reduce risk:

- State that this run is a subagent.
- Forbid spawning additional subagents or calling MCP tools.
- Require a concise, structured output.
- Constrain scope (paths, files, or a single concern).

Suggested header:

```
Role: Subagent worker.
Constraints:
- Complete the assigned task only.
- Do NOT spawn sub-agents or call orchestration tools.
- Do NOT create git commits or branches.
- Track and report modified files in the final response.
Output format:
- Summary of actions taken
- Modified files (one per line)
- Any issues encountered
```

## Orchestration Patterns

### Pattern A: Parallel exploration (job-based)

Spawn multiple read-only workers with:
- `sandbox="read-only"`
- shared `workingDirectory` (repo root)
- disjoint investigation questions (avoid duplicate work)

Then synthesize results in the main agent.

### Pattern B: Parallel writes + A2 recovery (shared worktree)

Phase 1: parallel exploration (read-only; up to `K_read`).

Phase 2: spawn write workers (up to `K_write`) with strict Worker Prompt Contract.

Phase 3: after all workers complete, run conflict detection (overlapping `modifiedFiles`), then:
- No conflicts → proceed
- Conflicts → rollback and re-run sequentially (temporary `K_write=1`) or manually resolve

### Pattern C: Reactive orchestration (“first-completed-wins”)

Use the async job tools when orchestration must be reactive (do not wait for the slowest job):

1) spawn N jobs immediately (`codex_spawn`)
2) wait for the first completion (`codex_wait_any`)
3) read results (`codex_result`) and adapt (spawn follow-ups or cancel others)
4) poll incremental progress when needed (`codex_events`)

Important constraints:
- Jobs are **in-memory** on the MCP server. If the MCP server restarts, job state is lost.
- All orchestration calls MUST be made within a single long-lived “main agent” Codex session so the MCP server process remains available.

Minimal pattern:

```
# Step 1: Spawn N workers (sequential tool calls are acceptable; jobs run concurrently once spawned)
mcp__codex-cli-wrapper__codex_spawn({ prompt: "...task A...", sandbox: "workspace-write", workingDirectory: "/data/Qwen3-VL" }) -> jobIdA
mcp__codex-cli-wrapper__codex_spawn({ prompt: "...task B...", sandbox: "workspace-write", workingDirectory: "/data/Qwen3-VL" }) -> jobIdB
mcp__codex-cli-wrapper__codex_spawn({ prompt: "...task C...", sandbox: "workspace-write", workingDirectory: "/data/Qwen3-VL" }) -> jobIdC

# Step 2: React to first completion
mcp__codex-cli-wrapper__codex_wait_any({ jobIds: [jobIdA, jobIdB, jobIdC], timeoutMs: 60000 }) -> { completedJobId }

# Step 3: Inspect result, adapt
mcp__codex-cli-wrapper__codex_result({ jobId: completedJobId }) -> { exitCode, finalMessage, stdoutTail, stderrTail }

# Step 4: Continue waiting, or cancel no-longer-needed work
mcp__codex-cli-wrapper__codex_wait_any({ jobIds: [remaining...], timeoutMs: 60000 })
mcp__codex-cli-wrapper__codex_cancel({ jobId: jobIdC })
```

`codex_events` usage (incremental progress):
- Use `cursor="0"` for first call.
- Persist returned `nextCursor` and pass it back on the next poll.
- Treat `done=true` as terminal (then call `codex_result`).

## Result Synthesis Checklist

After tool calls return:
- Validate each subagent output for errors and scope violations.
- Detect overlap/conflicts via `modifiedFiles` overlap (A2 protocol).
- Combine into a single coherent answer and propose the next action (apply patch, run tests, etc.).

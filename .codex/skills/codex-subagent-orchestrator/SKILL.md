---
name: codex-subagent-orchestrator
description: Orchestrate async Codex worker jobs via MCP job tools (codex_spawn/codex_wait_any/codex_events/codex_result/codex_cancel) using a boss–worker pattern with A2 optimistic concurrency and git rollback in a shared worktree.
---

# Codex Sub-Agent Orchestrator (Job-Based, Option 1)

This skill enables job-based orchestration so a single Codex “boss” agent can delegate work to multiple “worker” sub-agents using MCP **async job primitives**.

**Option selection**: Boss logic lives in the main agent via skill/prompt instructions (MCP server remains a primitives layer).

**Hard constraints (locked)**:
- No upstream Codex modifications (reference-only in `/references/codex`).
- No recursive sub-agents: workers MUST NOT spawn workers.
- Shared worktree model: workers edit the same workspace; workers MUST NOT create git commits.
- Git is REQUIRED for conflict detection and rollback.
- Coordination is A2: optimistic concurrency + git-based recovery.

## Canonical References (Avoid Redundancy)

This skill intentionally stays short and defers protocol details to canonical docs:

- Canonical runbook: `docs/reference/CODEX_SUBAGENTS_ORCHESTRATION.md`
- Quick checklist: `docs/reference/CODEX_SUBAGENTS_ORCHESTRATION_QUICKREF.md`
- Normative spec delta: `openspec/changes/2026-01-07-add-codex-subagents-mcp/specs/codex-mcp-subagents/spec.md`

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

## Concurrency Defaults

Do not re-specify numeric defaults in this skill (to avoid drift).

- Use the canonical defaults defined in `docs/reference/CODEX_SUBAGENTS_ORCHESTRATION.md`.
- If conflicts are frequent, temporarily reduce write concurrency (often `K_write=1`).

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

## Boss Algorithm (A2, Shared Worktree)

Follow the full A2 protocol in `docs/reference/CODEX_SUBAGENTS_ORCHESTRATION.md`. Minimum boss loop:

1) Preflight: ensure a clean baseline (`git status --porcelain`), record baseline HEAD (`git rev-parse HEAD`).
2) Spawn: use `codex_spawn` for workers (read vs write). Keep within the canonical `K_read` / `K_write`.
3) Monitor: prefer `codex_wait_any` for completions; use `codex_events` for progress when needed.
4) Collect: call `codex_result` for completed jobs; extract worker-reported `modifiedFiles`.
5) Detect conflict: overlap in `modifiedFiles` across jobs ⇒ conflict.
6) Recover: roll back to baseline and re-run conflicting work safely (usually reduce write concurrency).

## Tool Call Template (Subagent Prompt Header)

Do not maintain a separate worker contract template in this skill.

- Use the **Worker Prompt Contract** from `docs/reference/CODEX_SUBAGENTS_ORCHESTRATION.md`.
- Require the worker to list `modifiedFiles` explicitly (one per line) to enable A2 conflict detection.

## Reactive Orchestration (“first-completed-wins”)

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

- Validate worker outputs for errors/scope violations.
- Detect conflicts via overlapping `modifiedFiles` (A2 protocol).
- On conflict: roll back to baseline and re-run safely.
- Combine into one coherent final answer (single consolidated user response).

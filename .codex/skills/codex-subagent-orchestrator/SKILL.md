---
name: codex-subagent-orchestrator
description: Orchestrate async Codex sub-agent jobs via MCP job tools (codex_spawn/codex_wait_any/codex_events/codex_result/codex_cancel) in a shared worktree with workspace-write defaults and last-writer-wins tolerance; safe when auto-loaded inside delegated jobs.
---

# Async Sub-Agent Orchestration (Job-Based, Option 1)

This skill describes job-based orchestration where one Codex agent coordinates multiple concurrent delegated jobs (each running `codex exec --json`) via MCP **async job primitives**.

**Option selection**: Coordination logic lives in the main agent via skill/prompt instructions (the MCP server remains a primitives layer).

## Role Safety (Auto-Load Friendly)

This skill may be auto-loaded in either context:

- **Coordinator context (main agent)**: responsible for spawning/monitoring jobs and synthesizing one final user outcome.
- **Delegated-task context (sub-agent job)**: running inside a spawned job; responsible only for completing the assigned subtask and reporting results.

**Default rule**: If the current prompt does not explicitly request orchestration (spawn/monitor/cancel jobs), assume **delegated-task context** and DO NOT call `codex_spawn` / `codex_wait_any` / `codex_cancel`.

**Tone / relationship**: The coordinator and delegated sub-agents should interact as teammates. The distinction is task assignment (coordinate vs execute), not capability.

**Hard constraints (locked)**:
- No upstream Codex modifications (reference-only in `/references/codex`).
- No recursive sub-agents: delegated sub-agents MUST NOT spawn additional sub-agents (prompt discipline only; no hard tool blocking assumed).
- Shared worktree model: delegated sub-agents edit the same workspace; delegated sub-agents MUST NOT create git commits.
- Git is available for recovery and for “what changed?” inspection (recommended but not a hard gate for spawning).
- Coordination uses optimistic concurrency in a shared worktree; when overlapping edits happen, prefer *chaining* follow-up tasks rather than hard rollback.

## Canonical References (Avoid Redundancy)

This skill intentionally stays short and defers protocol details to canonical docs:

- Canonical runbook: `docs/reference/CODEX_SUBAGENTS_ORCHESTRATION.md`
- Quick checklist: `docs/reference/CODEX_SUBAGENTS_ORCHESTRATION_QUICKREF.md`
- Toy drills (safe scratch-only): `docs/reference/CODEX_SUBAGENTS_TOY_DRILLS.md`
- Normative spec delta: `openspec/changes/2026-01-07-add-codex-subagents-mcp/specs/codex-mcp-subagents/spec.md`

## Tool Naming (Verified)

In this repository, the MCP server name is `codex-cli-wrapper`, and callable tool identifiers preserve the hyphenated server name.

Job-based tool identifiers:
- `mcp__codex-cli-wrapper__codex_spawn` (async sub-agent spawn; returns `jobId` immediately)
- `mcp__codex-cli-wrapper__codex_status` (poll job status)
- `mcp__codex-cli-wrapper__codex_events` (poll normalized incremental events; cursor-based)
- `mcp__codex-cli-wrapper__codex_wait_any` (wait for first completion among `jobIds`)
- `mcp__codex-cli-wrapper__codex_result` (job result; default returns the final message as plain text; use `view:"full"` for status + tails)
- `mcp__codex-cli-wrapper__codex_cancel` (cancel running job; SIGTERM default, SIGKILL on `force=true`)

If the MCP server is renamed in `~/.codex/config.toml`, the tool prefix changes accordingly.

Note: Some Codex environments may accept underscore variants as aliases (e.g., `mcp__codex_cli_wrapper__codex_spawn`). Prefer the canonical identifiers from the MCP tool listing in the active Codex session.

## Core Idea (Job Semantics)

- A delegated sub-agent is an independent `codex exec --json` process spawned by the MCP server.
- Concurrency comes from background jobs that continue running after `codex_spawn` returns.
- The coordinator MUST NOT rely on “parallel tool calls” as a concurrency guarantee; upstream tool execution may serialize.
- All jobs operate in a shared worktree by default; safety relies on A2 optimistic concurrency + git rollback.

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

With this set, sub-agent job spawns may omit `sandbox` and still run with `workspace-write` consistently.

Safety notes:
- Never default to `danger-full-access` for subagents; only use it on explicit user request and with clear justification.
- Avoid `fullAuto` when explicitly setting `sandbox`; `fullAuto` is a convenience alias that implies `sandbox="workspace-write"` in Codex CLI.

## Inheriting Model / Reasoning Defaults (Config-Driven)

To ensure the coordinating agent and delegated sub-agents behave identically, prefer inheriting from `~/.codex/config.toml`:
- Do not set `model` in subagent calls unless an override is explicitly required.
- Do not set `reasoningEffort` in subagent calls unless an override is explicitly required.

## Coordinator Algorithm (A2, Shared Worktree)

Follow the full coordination protocol in `docs/reference/CODEX_SUBAGENTS_ORCHESTRATION.md`. Minimum coordinator loop:

1) Preflight: record baseline state (dirty is OK): `git status --porcelain`, optionally baseline HEAD (`git rev-parse HEAD`).
2) Spawn: use `codex_spawn` for delegated jobs (analysis vs edit). Keep within the canonical `K_read` / `K_write`.
3) Monitor: prefer `codex_wait_any` for completions; use `codex_events` for progress when needed.
4) Collect: call `codex_result` for completed jobs; extract delegated-agent-reported `modifiedFiles`.
5) Detect overlap: overlap in `modifiedFiles` across jobs ⇒ order-dependent edits.
6) Reconcile: prefer a follow-up “reconcile” job (or a sequential re-run) that continues from the current workspace state.

## Coordinator Waiting Pattern (Don’t Busy-Poll)

Long-running jobs (e.g., large code reviews) may take 10+ minutes. The coordinator should avoid “status spam” and avoid long blocking waits that prevent useful parallel coordinator work.

Recommended pattern:

1) **Warm-up handshake (fast)**: after spawning jobs, call `codex_events` once per job with `cursor="0"` to confirm that each job has started producing events (plan/ack).
2) **Short completion checks**: use `codex_wait_any` with a short `timeoutMs` (including `timeoutMs: 0` for a non-blocking check) to detect completions without stalling the coordinator.
   - Treat `{ timedOut: true, completedJobId: null }` as “no completion yet” (not an error).
3) **Backoff on quiet jobs**: if a job is producing no new events, poll it less frequently (e.g., 5s → 15s → 30s), rather than tight loops.
4) **Do something small while waiting** (examples):
   - Prepare the final synthesis skeleton (sections, checklists, acceptance criteria).
   - Identify likely conflict hotspots (files that multiple subtasks may touch) and adjust task decomposition.
   - Pre-collect repository context that does not require waiting (search symbols, map modules, read docs).

Rule of thumb: coordination calls should usually be short and bounded; the background concurrency comes from spawned jobs, not from long-running orchestration tool calls.

## Prompt Envelope (Coordinator → Delegated Sub-Agent)

Use a consistent prompt envelope to keep cross-agent “conversation” coherent. Example template:

```
Context: Delegated sub-agent task (spawned job).

Shared objective (1–2 sentences):
<why this subtask exists / what the user ultimately needs>

Subtask label:
<short stable identifier; include in the final response>

Task:
<specific assignment; keep scope narrow and testable>

Constraints:
- Do not spawn additional sub-agents or attempt orchestration.
- Do not create git commits/branches; do not `git checkout`, `git reset`, or `git stash`.
- Prefer minimal edits; avoid drive-by refactors.
- Assume the workspace may be dirty and may be modified by other delegated jobs; re-read any file you touch and integrate with current content.
- If scope ambiguity exists, make a reasonable assumption and state it explicitly in the final response.

Output format:
- Summary (what was done)
- Findings / decisions (bullet list)
- modifiedFiles (one path per line; required)
- Follow-ups (optional recommendations)
```

Do not copy/paste the full A2 protocol into delegated prompts; keep the delegated prompt focused on the subtask.

## Delegated Task Mode (When Auto-Loaded Inside a Job)

If the current process is a spawned job, treat the current role as a delegated teammate and:

- Complete the assigned subtask only; avoid expanding scope.
- Do not spawn sub-agents or call orchestration tools.
- Do not perform git write operations (commit/checkout/reset/stash); edits are file-based only.
- Provide an early acknowledgement + short plan in the first assistant message (improves progress visibility via `codex_events`).
- If you are about to run a long tool command, emit a short assistant message first (so cancellation still leaves a `lastAgentMessage`).
- In the final response, list `modifiedFiles` explicitly (one per line) so the coordinator can apply A2 conflict detection.

## Reactive Orchestration (“first-completed-wins”)

Use the async job tools when orchestration must be reactive (do not wait for the slowest job):

1) spawn N jobs immediately (`codex_spawn`)
2) wait for the first completion (`codex_wait_any`)
3) read results (`codex_result`) and adapt (spawn follow-ups or cancel others)
4) poll incremental progress when needed (`codex_events`)

Important constraints:
- Jobs are **in-memory** on the MCP server. If the MCP server restarts, job state is lost.
- All orchestration calls MUST be made within a single long-lived coordinating Codex session so the MCP server process remains available.

Minimal pattern:

```
# Step 1: Spawn N delegated jobs (sequential tool calls are acceptable; jobs run concurrently once spawned)
mcp__codex-cli-wrapper__codex_spawn({ prompt: "...task A...", sandbox: "workspace-write", workingDirectory: "/data/Qwen3-VL" }) -> jobIdA
mcp__codex-cli-wrapper__codex_spawn({ prompt: "...task B...", sandbox: "workspace-write", workingDirectory: "/data/Qwen3-VL" }) -> jobIdB
mcp__codex-cli-wrapper__codex_spawn({ prompt: "...task C...", sandbox: "workspace-write", workingDirectory: "/data/Qwen3-VL" }) -> jobIdC

# Step 2: React to first completion
mcp__codex-cli-wrapper__codex_wait_any({ jobIds: [jobIdA, jobIdB, jobIdC], timeoutMs: 60000 }) -> { completedJobId, timedOut }

# Step 3: Inspect result, adapt (default returns final message to avoid process noise)
mcp__codex-cli-wrapper__codex_result({ jobId: completedJobId }) -> "<final delegated message>"

# Step 4: Continue waiting, or cancel no-longer-needed work
mcp__codex-cli-wrapper__codex_wait_any({ jobIds: [remaining...], timeoutMs: 60000 })
mcp__codex-cli-wrapper__codex_cancel({ jobId: jobIdC })
```

`codex_events` usage (incremental progress):
- Use `cursor="0"` for first call.
- Persist returned `nextCursor` and pass it back on the next poll.
- Treat `done=true` as terminal (then call `codex_result`).

## Result Synthesis Checklist

- Validate delegated outputs for errors/scope violations.
- Detect conflicts via overlapping `modifiedFiles` (A2 protocol).
- On conflict: roll back to baseline and re-run safely.
- Combine into one coherent final answer (single consolidated user response).

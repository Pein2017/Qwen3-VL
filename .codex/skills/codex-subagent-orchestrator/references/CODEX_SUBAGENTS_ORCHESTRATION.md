# Codex Async Sub-Agents Orchestration (MCP Jobs, Option A)

Status: Active  
Scope: Operational + design guidance for async sub-agent orchestration (coordinator + delegated jobs) using MCP job tools.  
Owners: Tooling / DX  
Last updated: 2026-01-09

## Objective

Provide a reliable orchestration protocol for a **main agent (coordinator)** running multiple concurrent **delegated jobs (sub-agents)** via MCP job semantics:
- spawn (`codex_spawn`)
- monitor (`codex_events`, `codex_status`, `codex_wait_any`)
- collect (`codex_result`)
- cancel (`codex_cancel`)

This runbook is meant to be **portable**. Repositories may keep thin `docs/**` entrypoints pointing here.

## Hard Constraints (Workflow Assumptions)

1) **No recursive sub-agents** — delegated jobs MUST NOT spawn additional delegated jobs.  
2) **Shared worktree model** — all delegated jobs edit the same workspace filesystem; no per-job branching/commits.  
3) **Dirty worktrees are allowed** — coordination MUST work on top of an existing (dirty) workspace state.  
4) **Git is recommended** — use git for recovery and for “what changed?” inspection when needed.

## Document Map (Avoid Redundancy)

This skill intentionally keeps one canonical runbook:

- Canonical runbook (this file): `references/CODEX_SUBAGENTS_ORCHESTRATION.md`
  - Owns: architecture model, coordination protocol, delegated prompt contract, and default concurrency guidance.
- Quick reference: `references/CODEX_SUBAGENTS_ORCHESTRATION_QUICKREF.md`
  - Owns: short checklist and tool list only; MUST NOT re-specify full protocols.
- Toy drills: `references/CODEX_SUBAGENTS_TOY_DRILLS.md`
  - Owns: copy/paste stability drills + operator feedback checklist; MUST remain toy-only.

Repositories may also have:
- Repo stubs (optional): `docs/reference/CODEX_SUBAGENTS_*.md` (thin pointers to these files)
- Normative specs (optional): repo-specific spec systems (e.g., OpenSpec) describing requirements/scenarios

## Why Job Semantics (Not “Parallel Tool Calls”)

Even if the agent runtime supports “parallel tool calls”, the runtime may still serialize tool execution internally.

**Implication**: “multiple tool calls in one assistant message” MUST NOT be treated as a concurrency primitive.  
**Requirement**: real concurrency MUST come from background jobs that keep running after `codex_spawn` returns.

## Architecture

### Layer Model

Layer 0: Main Agent (Coordinator)
    - One Codex session
    - Calls MCP tools to coordinate delegated jobs
    - May use git for inspection/recovery (optional)
    - Synthesizes final result for the user
         │ MCP tool calls (codex_spawn, codex_events, codex_wait_any, …)
         ▼
Layer 1: MCP Job Manager (codex-mcp-server, or equivalent)
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

### Optional Grounded References (Repo-Dependent)

Some repos vendor the MCP server and/or upstream Codex sources; if present, these locations are helpful:
- MCP server: `mcp/codex-mcp-server/src/**`
- Upstream Codex: `references/codex/**`

This runbook does not require those to exist.

## Existing MCP Tools (Job API)

These tools are the only required interface for orchestration:

| Tool | Purpose |
|------|---------|
| `codex_spawn` | Start worker, return `jobId` immediately |
| `codex_status` | Get job status |
| `codex_result` | Get final delegated message (default); use `view="full"` for status + stdout/stderr tails |
| `codex_events` | Poll incremental events (cursor-based) |
| `codex_wait_any` | Wait for first completion among jobs |
| `codex_cancel` | Cancel running job (SIGTERM default, SIGKILL when `force=true`) |

**Tool identifiers** vary by MCP server name (prefix). Always prefer the tool list from your active Codex session.

## Workflow Model: Coordinator + Delegated Jobs

The main agent coordinates delegated jobs as background processes and produces a single consolidated outcome:

1) **Dispatch**: coordinator spawns N delegated jobs via `codex_spawn` (non-blocking)
2) **Monitor**: coordinator polls progress via `codex_events` and/or uses `codex_wait_any`
3) **React**: coordinator cancels jobs or adjusts strategy based on intermediate results
4) **Collect**: coordinator gathers final delegated messages via `codex_result()` (default is final message)
5) **Decide**: coordinator synthesizes results, then either finishes or spawns follow-up delegated jobs

Delegated jobs:
- execute independently as `codex exec --json` processes
- may edit files in the shared worktree (no commits)
- report completion/failure to MCP server
- MUST NOT spawn sub-agents or call orchestration tools

## Concurrency Model (Suggested Defaults)

Use bounded concurrency to reduce conflict risk:

- `K_read = 8` for analysis-only jobs (prompt-enforced “do not modify files”)
- `K_write = 3` for edit jobs (allowed to modify files under `workspace-write`)

If conflicts are frequent, temporarily reduce write concurrency (often `K_write=1`) and chain edits.

## Shared Worktree Coordination (Dirty OK, Git-assisted)

This protocol supports dirty worktrees and overlapping intent by treating the current workspace state as the baseline and using git as an optional recovery/inspection tool.

### Pre-Spawn Phase (Coordinator)

1) Record baseline state (dirty is OK):
   - `git status --porcelain`
   - `git diff --name-only`
2) Optionally record baseline HEAD (useful for recovery):
   - `git rev-parse HEAD`

### Execution Phase (Coordinator + Delegated Jobs)

1) Spawn N delegated jobs with `sandbox: "workspace-write"` (recommended).
2) For “analysis-only” jobs, enforce read-only behavior via prompt (not via sandbox).
3) Delegated jobs may edit files freely (no commits).
4) Coordinator monitors progress via `codex_events` and/or `codex_wait_any`.

### Post-Completion Phase (Coordinator)

For each completed job:
1) Fetch the final delegated message:
   - `codex_result({ jobId })`
2) Record delegated-job-reported `modifiedFiles` from the final message (required by the Delegated Prompt Contract).
3) Optionally extract file touches from `codex_events` (when file-change events are present).

Important limitation:
- Per-job `git status` snapshots are NOT reliable for attributing changes under concurrent writers.
- Use delegated self-reporting (`modifiedFiles`) as best-effort attribution and git only as an overall “what changed” view.

### Overlap Handling (Coordinator)

If the same file appears in multiple jobs’ `modifiedFiles`, this indicates **order-dependent edits**, not an automatic failure.

Recommended behaviors:
- Avoid concurrently spawning two edit jobs likely to touch the same file.
- If you need a follow-up edit on the same file, **chain** it: spawn the next job after the previous job finishes so it can “continue from what’s currently in the workspace”.
- If overlapping edits happened concurrently and results look inconsistent, prefer a **reconcile job** that re-reads the current file and produces a coherent final version, rather than rolling back.

Optional recovery (when needed):
- Use git to roll back to a recorded baseline HEAD if a cancellation/crash leaves the workspace inconsistent.

### Failure Recovery (Coordinator)

- On delegated job failure: capture `codex_result` (use `view="full"` for stderr tail if needed), inspect `git diff --name-only`, and decide whether to keep changes, reconcile, or roll back.
- On crash mid-write: prefer a reconcile job first; roll back only if the workspace is clearly inconsistent.

## Cancellation Semantics

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

Collaboration hygiene:
- **Refresh-before-write** (edit jobs): immediately before editing, re-read the exact symbol/region you will change and integrate with current content. If assumptions no longer hold, stop and report.
- **Scope declaration**: explicitly state which files/symbols you intend to touch. If you need to expand scope, request it (or return a scope-expansion request to the coordinator).

Quality rules (high-trust defaults):
- **Evidence hygiene**: keep “observed” separate from “hypothesized”.
- **Recommendations must exist**: if recommending a script/path, verify it exists before implying it does.
- **modifiedFiles discipline**: if you did not write files, output `modifiedFiles` as exactly `(none)`; do not list pre-existing dirty/untracked files.

Output format (recommended):
- Summary (what was done)
- Observations (confirmed facts)
- Evidence (file paths + small snippets / commands run)
- Hypotheses (explicitly labeled; plausible but not confirmed)
- Tests / next checks (how to validate or falsify hypotheses)
- Scope (intended vs actually touched)
- Refresh-before-write (yes/no, and what changed if applicable)
- modifiedFiles (one path per line; required)
- Follow-ups (optional recommendations)

Enforcement model:
1) Prompt instructions (required; soft enforcement)
2) Coordinator monitoring for violations (recommended): treat unexpected orchestration attempts as a job failure and re-run safely

## Anti-Patterns

- Relying on “parallel tool calls” as a concurrency guarantee.
- Allowing workers to spawn additional workers (recursive sub-agents).
- Allowing workers to create commits/branches (violates shared worktree model).
- Continuing after cancellation without git-based integrity checks.


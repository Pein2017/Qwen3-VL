# Codex Async Sub-Agents Orchestration (Quick Reference)

Status: Active  
Scope: Fast checklist for orchestrating Codex async sub-agents (coordinator + delegated jobs) via MCP job tools.  
Owners: Tooling / DX  
Last updated: 2026-01-09

## Canonical (Portable) Quick Reference

The full quickref lives inside the skill package:

- `.codex/skills/codex-subagent-orchestrator/references/CODEX_SUBAGENTS_ORCHESTRATION_QUICKREF.md`

This file stays as a stable repo entrypoint.

## Minimal Checklist

1) Preflight: record baseline state (dirty OK): `git status --porcelain`, `git diff --name-only`.
2) Spawn: use `codex_spawn` for delegated jobs; keep bounded concurrency for analysis vs edits.
3) Monitor: `codex_wait_any` for completion; use `codex_events` for progress if needed.
4) Collect: `codex_result()` for each job; require `modifiedFiles` (or `(none)`).
5) Overlap: overlap in `modifiedFiles` ⇒ order-dependent edits; prefer chaining or a reconcile job (not rollback by default).
6) Recover: on cancellation/crash, run `git status --porcelain` and decide whether to reconcile or roll back.

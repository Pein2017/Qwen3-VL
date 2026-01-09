# Codex Async Sub-Agents Orchestration (MCP Jobs, Option A)

Status: Active  
Scope: Repo-facing entrypoint for sub-agent orchestration docs.  
Owners: Tooling / DX  
Last updated: 2026-01-09

## Canonical (Portable) References

The full, portable runbook lives inside the skill package:

- Canonical runbook: `.codex/skills/codex-subagent-orchestrator/references/CODEX_SUBAGENTS_ORCHESTRATION.md`
- Quick checklist: `.codex/skills/codex-subagent-orchestrator/references/CODEX_SUBAGENTS_ORCHESTRATION_QUICKREF.md`
- Toy drills (safe scratch-only): `.codex/skills/codex-subagent-orchestrator/references/CODEX_SUBAGENTS_TOY_DRILLS.md`

This `docs/reference/...` file stays as a **stable repo entrypoint** to avoid scattering links across the doc tree.

## Repo-Specific Pointers (Qwen3-VL)

- Local MCP server (editable): `mcp/codex-mcp-server`
- Upstream Codex reference (read-only, no modifications): `references/codex`
- OpenSpec (normative requirements): `openspec/changes/2026-01-07-add-codex-subagents-mcp/specs/codex-mcp-subagents/spec.md`
- Skill entrypoint: `.codex/skills/codex-subagent-orchestrator/SKILL.md`

## Minimal Workflow Summary (for scanning)

Use job semantics (not “parallel tool calls”):

1) **Dispatch**: spawn N delegated jobs via `codex_spawn` (non-blocking).
2) **Monitor**: poll with `codex_events` / `codex_status`, and detect completions via `codex_wait_any`.
3) **Collect**: pull final messages via `codex_result`.
4) **React**: cancel with `codex_cancel` when needed.
5) **Reconcile**: if edits overlap, prefer a follow-up “reconcile job” that continues from current workspace state.

Shared worktree model (dirty is OK):
- delegated jobs may edit files, but MUST NOT create commits/branches or rewrite git history
- the user may edit files while jobs run
- use bounded concurrency and scope separation to keep edits mostly linear/incremental

Delegated jobs MUST:
- declare scope (files/symbols intended to touch)
- refresh-before-write (re-read the exact symbol/region right before editing)
- report `modifiedFiles` (or `(none)` for read-only analysis jobs)

For the full protocol + prompt templates, use the canonical runbook in `.codex/skills/.../references/`.

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

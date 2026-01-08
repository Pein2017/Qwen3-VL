# Tasks: Async Codex Sub-Agents (Option A)

## Primary Tasks

- [x] Update OpenSpec delta spec for Option A (job-based, A2 coordination)
- [x] Update canonical orchestration runbook in `docs/reference/`
- [x] Update quick reference card in `docs/reference/`
- [x] Update boss orchestration skill in `.codex/skills/`
- [ ] Add orchestration instructions to Codex profile/system prompt configuration (operational wiring)

## Async Sub-Agent MCP Job Tools (Already Implemented)

- [x] Add `codex_spawn` tool definition and schema
- [x] Add `codex_status` tool definition and schema
- [x] Add `codex_result` tool definition and schema
- [x] Add `codex_cancel` tool definition and schema
- [x] Add `codex_events` tool definition and schema (normalized event stream)
- [x] Add `codex_wait_any` tool definition and schema (optional helper)
- [x] Implement in-memory job manager with concurrency cap (`CODEX_MCP_MAX_JOBS`, default 32)
- [x] Spawn subagents via `codex exec --json` and parse JSONL into normalized events
- [x] Add unit tests covering spawn/status/result/events lifecycle
- [x] Update `mcp/codex-mcp-server/docs/api-reference.md` for async subagent tools

## Validation Tasks

- [x] Smoke test: `codex_spawn` returns a `jobId` immediately in a live Codex session ✓ CONFIRMED
- [x] Smoke test: `codex_events` can be polled and returns normalized events ✓ CONFIRMED
- [x] Smoke test: `codex_wait_any` returns the first completed job (reactive orchestration) ✓ CONFIRMED
- [ ] Acceptance test (primary): disjoint-file edits with 2+ write-enabled workers (no conflict)
- [ ] Acceptance test (secondary): same-file edits trigger conflict detection + git rollback + safe re-run
- [x] Run `openspec validate 2026-01-07-add-codex-subagents-mcp --strict`

## Documentation Tasks

- [x] Update `docs/reference/CODEX_SUBAGENTS_ORCHESTRATION.md` to be canonical
- [x] Update `docs/reference/CODEX_SUBAGENTS_ORCHESTRATION_QUICKREF.md` to be a thin checklist
- [x] Update `.codex/skills/codex-subagent-orchestrator/SKILL.md` to reference the canonical docs (avoid redundancy)

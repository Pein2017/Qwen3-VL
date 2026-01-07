# Tasks: Codex Subagent Orchestration

## Primary Tasks

- [x] Create orchestration instruction document at `openspec/changes/2026-01-07-add-codex-subagents-mcp/orchestration-instructions.md`
- [ ] Add orchestration instructions to Codex profile/system prompt configuration
- [x] Document parallel tool invocation examples in `mcp/codex-mcp-server/README.md`
- [x] Create example prompts demonstrating orchestration patterns (main agent + async subagents)

## Async Subagent MCP Tools (Option B)

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

## Optional Enhancement Tasks

- [ ] Add `includeGitSnapshot` parameter to `codex` tool schema in `src/tools/definitions.ts`
- [ ] Implement git snapshot collection in `src/tools/handlers.ts`
- [ ] Add tests for git snapshot functionality

## Validation Tasks

- [x] Smoke test: Main agent spawns 2 parallel read-only subagents ✓ CONFIRMED
- [x] Verify parallel MCP tool calls execute concurrently ✓ CONFIRMED
- [x] Smoke test: `codex_spawn` returns a `jobId` immediately in a live Codex session ✓ CONFIRMED
- [x] Smoke test: `codex_events` can be polled and returns normalized events ✓ CONFIRMED
- [x] Smoke test: `codex_wait_any` returns the first completed job (reactive orchestration) ✓ CONFIRMED
- [ ] Smoke test: Main agent spawns sequential write subagents to different files
- [ ] Smoke test: Main agent synthesizes results from multiple subagents
- [x] Run `openspec validate 2026-01-07-add-codex-subagents-mcp --strict`

## Documentation Tasks

- [x] Update `mcp/codex-mcp-server/README.md` with subagent orchestration section
- [x] Add anti-patterns and troubleshooting guide
- [x] Create quick-reference card for orchestration patterns

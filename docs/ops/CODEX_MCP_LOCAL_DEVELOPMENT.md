# Codex MCP (Local Developer Setup)

Status: Active
Scope: Local development setup for Codex CLI MCP integration in this repo.
Owners: Ops + Tooling
Last updated: 2026-01-08
Related:
- [ops/README.md](README.md)
- [reference/CODEX_SUBAGENTS_ORCHESTRATION.md](../reference/CODEX_SUBAGENTS_ORCHESTRATION.md)

This document describes how to run and develop Codex MCP servers locally with **Codex CLI** (not Claude Code).
It focuses on developer workflows: submodules, local builds, config wiring, and fast verification.

## Goals / Non-goals

Goals:
- Configure Codex CLI to load MCP servers from a working copy (including `mcp/codex-mcp-server`).
- Support local development of the MCP server (build/test, update fork, bump submodule pointer).
- Provide a repeatable workflow for subagent orchestration primitives (async job tools are documented elsewhere).

Non-goals:
- Installing MCP for Claude Code, VS Code extensions, or third-party wrappers.
- General host-machine setup guidance beyond what is needed for local development.

## Repository Layout (Git Submodule)

`Qwen3-VL` includes `mcp/codex-mcp-server` as a **git submodule**.

Key implications:
- The parent repo (`Qwen3-VL`) stores a **pointer** to a specific submodule commit (gitlink).
- The submodule repo (`codex-mcp-server`) stores the full source history and has its own remotes.

### Development in Parallel (Shared Worktree)

This layout supports “team-style” development in a single filesystem checkout:
- All tools (including spawned Codex subagents) operate in the same working directory tree under `/data/Qwen3-VL`.
- Git history is still separate: commits that touch `mcp/codex-mcp-server/` must be created in the submodule repo, then the parent repo must bump the submodule pointer.

### Initialize After Clone

```bash
cd /data/Qwen3-VL
git submodule update --init --recursive
```

### Inspect the Pinned Version

```bash
cd /data/Qwen3-VL
git submodule status
```

## Prerequisites

- Codex CLI installed and functional: `codex --version`
- Node.js + npm available for building the MCP server: `node --version`, `npm --version`

## Build `codex-mcp-server` (dist/)

The Codex MCP server is launched by Codex as a stdio process. The stable entrypoint is `dist/index.js`.

```bash
cd /data/Qwen3-VL/mcp/codex-mcp-server
npm ci
npm run build
npm test  # optional but recommended
```

## Configure Codex CLI to Load the MCP Server

This repo uses a project-local Codex config at `.codex/config.toml`.

### Minimal Config (dist/ entrypoint)

```toml
[mcp_servers.codex-cli-wrapper]
command = "node"
args = ["/data/Qwen3-VL/mcp/codex-mcp-server/dist/index.js"]

# All *_timeout_sec values are seconds (not milliseconds).
startup_timeout_sec = 40
tool_timeout_sec = 120

# Optional defaults for the MCP server process.
env = { CODEX_MCP_DEFAULT_SANDBOX = "workspace-write", CODEX_MCP_MAX_JOBS = "32" }
```

### Alternative: Register via `codex mcp add` (global config)

This writes an entry to `~/.codex/config.toml`.

```bash
codex mcp add codex-cli-wrapper \
  --env CODEX_MCP_DEFAULT_SANDBOX=workspace-write \
  --env CODEX_MCP_MAX_JOBS=32 \
  -- node /data/Qwen3-VL/mcp/codex-mcp-server/dist/index.js
```

### Tool Naming (hyphens preserved)

The MCP tool prefix preserves the server name, including hyphens:
- Server name: `codex-cli-wrapper`
- Tool prefix: `mcp__codex-cli-wrapper__...`

Example tool identifiers:
- `mcp__codex-cli-wrapper__ping`
- `mcp__codex-cli-wrapper__codex`
- `mcp__codex-cli-wrapper__codex_spawn` (async job tools)

## Verify Health & Tool Registration

List configured MCP servers:

```bash
codex mcp list
```

Inspect a specific server config:

```bash
codex mcp get codex-cli-wrapper --json
```

For subagent orchestration patterns and smoke tests, refer to:
- [reference/CODEX_SUBAGENTS_ORCHESTRATION.md](../reference/CODEX_SUBAGENTS_ORCHESTRATION.md)

## Developing `codex-mcp-server` Locally

### Mode 1: dist/ build (recommended integration target)

Workflow:
1) Edit TypeScript sources under `mcp/codex-mcp-server/src/`.
2) Rebuild `dist/`.
3) Restart the Codex session to reload MCP tools (server state is in-memory).

```bash
cd /data/Qwen3-VL/mcp/codex-mcp-server
npm run build
npm test
```

### Mode 2: tsx dev (fast iteration)

This runs TypeScript directly via `tsx`.

```toml
[mcp_servers.codex-cli-wrapper]
command = "npm"
args = ["--prefix", "/data/Qwen3-VL/mcp/codex-mcp-server", "run", "dev"]
tool_timeout_sec = 120
env = { CODEX_MCP_DEFAULT_SANDBOX = "workspace-write" }
```

Note: `dist/` remains the canonical entrypoint for stable integration.

## Submodule Update Workflow (Two Repos)

Changes under `mcp/codex-mcp-server/` require two commits if the parent repo should pick up the new version.

### 1) Commit in the submodule repo

```bash
cd /data/Qwen3-VL/mcp/codex-mcp-server
git status
git add -A
git commit -m "feat: ..."
git push origin main
```

### 2) Bump the submodule pointer in `Qwen3-VL`

```bash
cd /data/Qwen3-VL
git add mcp/codex-mcp-server
git commit -m "chore(mcp): bump codex-mcp-server submodule"
git push origin main
```

### Sync the fork with official upstream (submodule repo)

```bash
cd /data/Qwen3-VL/mcp/codex-mcp-server
git fetch upstream
git checkout main
git merge upstream/main
git push origin main
```

## Troubleshooting

### "Unknown jobId" when polling async jobs

Cause:
- Async jobs are stored in-memory by the MCP server process.
- If Codex restarts (or the MCP server restarts), job state is lost and polling returns “unknown jobId”.

Mitigation:
- Keep reactive orchestration within a single long-lived Codex session.

### Timeout confusion (`tool_timeout_sec`)

`tool_timeout_sec` is seconds (Duration seconds), not milliseconds.
Use `tool_timeout_sec = 120` for a 2-minute timeout.

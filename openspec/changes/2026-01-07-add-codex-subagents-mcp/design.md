# Design: Codex Subagent Orchestration

## Philosophy

Claude Code's subagent power comes from **instructions, not infrastructure**. The `Task` tool is essentially parallel Claude invocations guided by well-crafted system prompts. We adopt the same philosophy for Codex.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Codex Main Agent                         │
│  (with orchestration instructions in system prompt)         │
└─────────────────────┬───────────────────────────────────────┘
                      │ Parallel MCP tool calls
          ┌───────────┼───────────┐
          ▼           ▼           ▼
    ┌──────────────────────┐ ┌──────────────────────┐ ┌──────────────────────┐
    │ mcp__codex-cli-wrapper│ │ mcp__codex-cli-wrapper│ │ mcp__codex-cli-wrapper│
    │ __codex tool call #1  │ │ __codex tool call #2  │ │ __codex tool call #3  │
    └──────────┬───────────┘ └──────────┬───────────┘ └──────────┬───────────┘
         │            │            │
         ▼            ▼            ▼
    ┌──────────┐ ┌──────────┐ ┌──────────┐
    │ codex    │ │ codex    │ │ codex    │
    │ exec #1  │ │ exec #2  │ │ exec #3  │
    └────┬─────┘ └────┬─────┘ └────┬─────┘
         │            │            │
         └────────────┼────────────┘
                      ▼
              Shared Working Directory
```

## Core Mechanism

### How Parallel Execution Works

MCP protocol supports multiple tool calls in a single message. When the main agent issues N `mcp__codex-cli-wrapper__codex` tool calls simultaneously:

1. MCP server receives N concurrent requests
2. Each spawns a separate `codex exec` process
3. Results return as N separate tool responses
4. Main agent synthesizes all results

In this repository configuration, the callable subagent tool identifier is `mcp__codex-cli-wrapper__codex`.

### Existing Tool Capabilities

The `mcp__codex-cli-wrapper__codex` MCP tool already provides:

| Parameter | Purpose for Orchestration |
|-----------|---------------------------|
| `prompt` | Subtask-specific instruction |
| `workingDirectory` | Shared workspace path |
| `sandbox` | Permission control (`read-only` for exploration) |
| `sessionId` | Optional continuity for multi-turn subtasks |
| `model` | Can vary per subtask if needed |

## Orchestration Patterns

### Pattern 1: Parallel Exploration (Read-Only)

**Use case**: Gather information from multiple areas simultaneously.

```
Main agent receives: "Understand the authentication system"

Orchestration:
├── Subagent A: "Analyze src/auth/ - list all auth-related functions"
├── Subagent B: "Analyze src/middleware/ - find auth middleware"
└── Subagent C: "Analyze tests/auth/ - summarize test coverage"

All use: sandbox="read-only", same workingDirectory
```

### Pattern 2: Divide and Conquer (Sequential Write)

**Use case**: Multiple edits to different files.

```
Main agent receives: "Add logging to all API endpoints"

Phase 1 - Explore (parallel, read-only):
├── Subagent A: "List all files in src/api/"
└── Subagent B: "Find the logging utility location"

Phase 2 - Edit (sequential or parallel to different files):
├── Subagent C: "Add logging to src/api/users.ts"
├── Subagent D: "Add logging to src/api/orders.ts"
└── Subagent E: "Add logging to src/api/products.ts"

Write subagents use: sandbox="workspace-write"
```

### Pattern 3: Specialist Delegation

**Use case**: Different expertise for different aspects.

```
Main agent receives: "Review this PR for quality"

Orchestration:
├── Subagent A: "Review for security vulnerabilities"
├── Subagent B: "Review for performance issues"
├── Subagent C: "Review for code style and best practices"
└── Subagent D: "Verify test coverage is adequate"

Main agent synthesizes into unified review.
```

## Anti-Patterns to Avoid

### Anti-Pattern 1: Parallel Writes to Same File
```
❌ BAD:
├── Subagent A: "Add function foo() to utils.ts"
└── Subagent B: "Add function bar() to utils.ts"
    ^ Race condition - one will overwrite the other

✅ GOOD:
├── Subagent A: "Add function foo() to utils.ts"
└── (wait for completion)
└── Subagent B: "Add function bar() to utils.ts"
```

### Anti-Pattern 2: Overly Broad Subtask Prompts
```
❌ BAD: "Analyze the codebase" (too vague, duplicates work)

✅ GOOD: "Analyze src/auth/ for security patterns" (scoped)
```

### Anti-Pattern 3: Unnecessary Subagent Spawning
```
❌ BAD: Spawn subagent for "read package.json" (trivial task)

✅ GOOD: Main agent reads simple files directly
```

## Decision Framework

```
Should I spawn subagents?
│
├─ Is the task parallelizable?
│  └─ NO → Do it sequentially
│
├─ Are subtasks independent?
│  └─ NO → Chain them sequentially
│
├─ Will subtasks write to same files?
│  └─ YES → Serialize writes or redesign
│
├─ Is overhead worth it? (each subagent = API call)
│  └─ NO → Do it in main agent
│
└─ YES to all → Spawn parallel subagents
```

## Minimal MCP Enhancement (Optional)

Add `includeGitSnapshot` parameter to existing `codex` tool:

```typescript
// When true, append git status/diff to response
if (args.includeGitSnapshot) {
  const status = await exec('git status --porcelain');
  const diff = await exec('git diff');
  response._meta.gitSnapshot = { status, diff };
}
```

This helps main agent see cumulative changes after parallel edits.

## Result Synthesis

After subagents complete, main agent must:

1. **Collect**: Gather all subagent responses
2. **Validate**: Check for errors or conflicts
3. **Synthesize**: Combine into coherent response
4. **Report**: Present unified result to user

Example synthesis prompt pattern:
```
I spawned 3 subagents to analyze different modules.

Results:
- Auth module: [subagent A response]
- API module: [subagent B response]
- DB module: [subagent C response]

Synthesized findings: [main agent combines insights]
```

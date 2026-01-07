# Proposal: Codex Subagent Orchestration via Instructions

## Summary
Enable Codex CLI to orchestrate multiple subagent runs in parallel by leveraging the existing `codex` MCP tool with enhanced main-agent instructions. This approach mirrors Claude Code's native subagent workflow without requiring significant new MCP server code.

## Motivation
Codex CLI can consume MCP tools, but lacks guidance on how to use the existing `codex` tool for parallel task delegation. Claude Code achieves powerful subagent orchestration primarily through **well-crafted system instructions**, not complex orchestration infrastructure.

Key insight: The existing `codex-mcp-server` already supports:
- Multiple parallel tool invocations (MCP protocol feature)
- Shared working directory (`workingDirectory` parameter)
- Sandbox control (`sandbox` parameter)
- Session continuity (`sessionId` parameter)

What's missing is **instruction-based guidance** that teaches the main agent when and how to orchestrate subagents effectively.

## Scope
1. **Primary**: Create comprehensive orchestration instructions for Codex main agent
2. **Secondary**: Add minimal MCP enhancements (optional `includeGitSnapshot` parameter)
3. **Documentation**: Best practices, patterns, and anti-patterns for subagent orchestration

## Approach Comparison

| Aspect | Heavy MCP Approach | Instruction-Based Approach (Chosen) |
|--------|-------------------|-------------------------------------|
| New MCP tools | `spawnSubagents` tool | None (use existing `codex` tool) |
| Complexity | High | Low |
| Flexibility | Fixed patterns | Agent-driven, adaptive |
| Maintenance | More code | More documentation |
| Claude Code parity | Partial | High (same philosophy) |

## Non-Goals
- No new `spawnSubagents` MCP tool (existing `codex` tool suffices)
- No complex concurrency management in MCP server (agent handles this)
- No automatic conflict resolution (document as caveat)

## Risks
- Main agent may not follow orchestration patterns consistently
- Parallel writes to same files can cause conflicts (mitigated by instructions)
- Token usage increases with multiple subagent calls

## Success Criteria
- Codex main agent can successfully orchestrate 2+ parallel subagent tasks
- Clear documentation enables consistent orchestration patterns
- Workflow mirrors Claude Code's Task tool behavior

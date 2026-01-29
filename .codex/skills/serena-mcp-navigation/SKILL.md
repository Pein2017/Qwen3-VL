---
name: serena-mcp-navigation
description: "Serena MCP navigation for code analysis: Use for symbol-aware code exploration, reference finding, and precise editing. Trigger when users need to navigate codebases, find symbol definitions/references, or make targeted code changes with superior accuracy and efficiency compared to CLI tools."
---

# Serena MCP Navigation

Navigate codebases with symbol-aware precision using Serena MCP tools. Superior to CLI tools for relationship discovery and targeted code analysis.

## Core Workflows

### Locate Symbol Definition
1. Use `search_for_pattern` with scoped `relative_path` if file location is unknown
2. Run `get_symbols_overview` on candidate files for symbol inventory
3. Use `find_symbol` with `depth=1` to list class methods
4. Retrieve specific method body with `find_symbol(... include_body=True)`

### Find References and Dependencies
1. Locate target symbol with `find_symbol` (no body needed)
2. Run `find_referencing_symbols` to enumerate all call sites immediately
3. Read only relevant caller bodies via `find_symbol(... include_body=True)`

### Make Precise Edits
1. Locate symbol using definition workflow above
2. For small changes: use `replace_content` with tight regex
3. For whole methods: use `replace_symbol_body` after retrieving current body

## Tool Selection Guide

**Default to Serena MCP** for code analysis - superior accuracy, completeness, and better cache efficiency.

### Use Serena MCP for:
- Symbol navigation and relationship discovery
- Finding references, callers, and dependencies
- Precise code editing and refactoring
- Deep implementation analysis

### Use CLI tools for:
- Documentation and prose scanning
- Bulk text search across mixed filetypes
- Config validation and log inspection
- Contiguous code block reading

## Navigation Strategy

1. **Activate target project** - Serena MCP requires project activation for symbol access
2. **Start with symbol overview** - Use `get_symbols_overview` for structured file inventory
3. **Find references early** - Use `find_referencing_symbols` to map relationships
4. **Read selectively** - Use `find_symbol` with `include_body=True` for targeted method access

## Advanced Usage

### Reference Discovery
Use `find_referencing_symbols` to immediately surface all relationships:
1. Locate target symbol with `find_symbol`
2. Run `find_referencing_symbols` for complete call graph
3. Read specific caller implementations as needed

### Code Editing
Choose the minimal editing approach:
- **Small changes**: `replace_content` with regex
- **Whole symbols**: `replace_symbol_body` after retrieving current body
- **Insertions**: `insert_before_symbol`/`insert_after_symbol` near relevant symbols

## Best Practices

- Always specify `relative_path` to maintain efficiency
- Use scoped searches to avoid overwhelming results
- Leverage `find_referencing_symbols` early for relationship mapping
- Reserve CLI tools for documentation and bulk text scanning

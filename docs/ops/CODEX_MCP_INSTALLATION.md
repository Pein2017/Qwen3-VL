# Codex MCP Installation Guide

Status: Active
Scope: Installation and configuration steps for Codex MCP used by local tooling.
Owners: Ops
Last updated: 2026-01-02
Related: [ops/README.md](README.md)

This guide provides step-by-step instructions for installing and configuring the Codex MCP server for Claude Code on any host machine.

## Prerequisites

Before installation, ensure you have:

1. **Claude Code** (v2.0.56 or later)
2. **Codex CLI** (v0.61.0 or later)
3. **UV tool** (Python package installer)

Placeholders used in this guide:
- `<UVX_BIN_DIR>` — directory that contains the `uvx` executable
- `<UVX_ENV_FILE>` — uv-generated env file to source (if used)
- `<UVX_PATH>` — full path to `uvx` (if not on PATH)
- `<SHELL_RC>` — shell startup file (e.g., `.bashrc`, `.zshrc`)
- `<CLAUDE_CONFIG_PATH>` — Claude config file location
- `<CLAUDE_SETTINGS_PATH>` — Claude settings file location

### Check Prerequisites

```bash
# Check Claude Code version
claude --version

# Check Codex CLI version
codex --version

# Check if UV is installed
uv --version
```

## Installation Steps

### Step 1: Install UV Tool (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After installation, ensure `uvx` is available on PATH:

```bash
# Add to your shell configuration (replace <UVX_BIN_DIR> with the directory that contains uvx)
export PATH="<UVX_BIN_DIR>:$PATH"

# Or source the uv-provided env file (replace <UVX_ENV_FILE> with the actual path)
source <UVX_ENV_FILE>
```

Verify installation:

```bash
which uvx
# Should output: <UVX_PATH>
```

### Step 2: Remove Existing Codex MCP (if installed)

If you have an official Codex MCP installed, remove it first:

```bash
claude mcp remove codex
```

### Step 3: Install Codex MCP

Use the `uvx` executable on PATH (or replace `<UVX_PATH>` with the output of `which uvx`):

```bash
claude mcp add codex -s user --transport stdio -- uvx --from git+https://github.com/GuDaStudio/codexmcp.git codexmcp
```

### Step 4: Verify Installation

```bash
claude mcp list
```

Expected output:
```
Checking MCP server health...

codex: uvx --from git+https://github.com/GuDaStudio/codexmcp.git codexmcp - ✓ Connected
```

### Step 5: Restart Claude Code

Exit and restart your Claude Code session to ensure the MCP tools are fully loaded.

### Step 6: Verify MCP Tools Are Available

After restarting, run:

```bash
/mcp
```

You should see the Codex MCP server listed with status "✓ Connected".

## Configuration Files

### MCP Server Configuration Location

The MCP server configuration is stored in the Claude configuration file:
```
<CLAUDE_CONFIG_PATH>
```

Example configuration entry:
```json
{
  "mcpServers": {
    "codex": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/GuDaStudio/codexmcp.git",
        "codexmcp"
      ],
      "transport": "stdio"
    }
  }
}
```

### Permissions Configuration

To grant Claude Code automatic permissions to use Codex MCP tools, edit:
```
<CLAUDE_SETTINGS_PATH>
```

Add the tool to the allow list:
```json
{
  "permissions": {
    "allow": [
      "mcp__codex__codex"
    ],
    "deny": []
  }
}
```

Or use wildcard to allow all MCP tools:
```json
{
  "permissions": {
    "allow": [
      "*"
    ],
    "deny": []
  }
}
```

## Testing the Installation

After installation and restart, test the MCP connection:

1. Start a new Claude Code conversation
2. Ask Claude to interact with Codex
3. Claude should be able to use the `mcp__codex__codex` tool

Example test command you can ask Claude to run:
```
Can you send a test greeting to Codex using the MCP tool?
```

## Troubleshooting

### Issue: "Failed to connect" Error

**Symptom:** `claude mcp list` shows "✗ Failed to connect"

**Solution:** The issue is usually PATH-related. Make sure you're using the **full path** to `uvx`:

1. Find your uvx path:
   ```bash
   which uvx
   ```

2. Remove and re-add with full path:
   ```bash
   claude mcp remove codex
   claude mcp add codex -s user --transport stdio -- $(which uvx) --from git+https://github.com/GuDaStudio/codexmcp.git codexmcp
   ```

### Issue: Tool Not Available in Session

**Symptom:** `mcp__codex__codex` tool not found

**Solution:** Restart Claude Code session to load the MCP tools.

### Issue: uvx Command Not Found

**Symptom:** Shell can't find `uvx` command

**Solution:** Add UV to PATH permanently:

```bash
# Add to shell config (replace <UVX_BIN_DIR> and <SHELL_RC> as needed)
echo 'export PATH="<UVX_BIN_DIR>:$PATH"' >> <SHELL_RC>

# Reload shell configuration
source <SHELL_RC>
```

### Issue: Connection Works but Tools Don't Load

**Symptom:** `claude mcp list` shows connected, but tools aren't available

**Solution:**
1. Completely exit Claude Code (not just end conversation)
2. Restart Claude Code
3. The tools should be available in new conversations

## Platform-Specific Notes

### Windows (WSL Recommended)

Windows users are strongly recommended to run this in WSL (Windows Subsystem for Linux). Follow the same Linux instructions within WSL.

### macOS

Replace `<UVX_PATH>` placeholders with the local `uvx` path if it is not available on PATH.

### Linux

Works on most distributions. For Alpine or musl-based distributions, you may need additional dependencies.

## Manual Configuration

If you prefer to manually edit the configuration file:

1. Open `<CLAUDE_CONFIG_PATH>`
2. Add the MCP server configuration:

```json
{
  "mcpServers": {
    "codex": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/GuDaStudio/codexmcp.git",
        "codexmcp"
      ],
      "transport": "stdio"
    }
  }
}
```

3. Save and restart Claude Code

## Uninstallation

To remove Codex MCP:

```bash
claude mcp remove codex
```

This removes the configuration from `<CLAUDE_CONFIG_PATH>`.

## Additional Resources

- Codex MCP Repository: https://github.com/GuDaStudio/codexmcp
- Claude Code Documentation: https://docs.claude.com/
- UV Documentation: https://github.com/astral-sh/uv

## Quick Reference Card

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to PATH (replace <UVX_BIN_DIR>)
export PATH="<UVX_BIN_DIR>:$PATH"

# Install Codex MCP (use uvx from PATH or replace <UVX_PATH>)
claude mcp add codex -s user --transport stdio -- uvx --from git+https://github.com/GuDaStudio/codexmcp.git codexmcp

# Verify
claude mcp list

# Restart Claude Code
# Exit and start again

# Test in new session
/mcp
```

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify all prerequisites are met
3. Ensure `uvx` is on PATH (or substitute the full path)
4. Check Claude Code and Codex CLI versions
5. Restart Claude Code after configuration changes

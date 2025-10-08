#!/usr/bin/env python3
"""Fix syntax errors in MCP server imports."""

import os
import re
from pathlib import Path


def fix_file(filepath):
    """Fix syntax errors in a single file."""
    with open(filepath) as f:
        content = f.read()

    # Fix double comma in imports
    content = re.sub(r"MCPToolSpec,,", "MCPToolSpec,", content)

    with open(filepath, "w") as f:
        f.write(content)


def main():
    """Fix all files."""
    servers_dir = Path("DeepResearch/src/mcp_servers/vendored")

    for server_file in servers_dir.glob("*.py"):
        fix_file(server_file)
        print(f"Fixed {server_file}")


if __name__ == "__main__":
    main()

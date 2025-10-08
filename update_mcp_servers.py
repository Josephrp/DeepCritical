#!/usr/bin/env python3
"""
Script to update all MCP servers to use the new @mcp_tool() decorator pattern.

This script updates all vendored MCP servers to use the new Pydantic AI integration patterns.
"""

import os
import re
from pathlib import Path


def update_server_file(filepath):
    """Update a single MCP server file to use new patterns."""
    print(f"Updating {filepath}")

    with open(filepath) as f:
        content = f.read()

    # Update imports
    import_pattern = r"(from \.\.\.datatypes\.mcp import \(\s*[^)]*?)(\s*\))"

    def update_imports(match):
        imports = match.group(1)
        if "MCPAgentIntegration" not in imports:
            imports += ",\n    MCPAgentIntegration"
        return imports + match.group(2)

    content = re.sub(import_pattern, update_imports, content, flags=re.DOTALL)

    # Update constructor type annotation
    content = re.sub(
        r"config: Optional\[MCPServerConfig\] = None",
        "config: MCPServerConfig | None = None",
        content,
    )

    # Remove MCPToolSpec decorators and replace with @mcp_tool()
    # Find all @mcp_tool(MCPToolSpec( patterns
    mcp_tool_pattern = r"@mcp_tool\(MCPToolSpec\(\s*[^}]*?\s*\)\)\s*\n"
    content = re.sub(mcp_tool_pattern, "@mcp_tool()\n", content, flags=re.DOTALL)

    # Update type annotations
    content = re.sub(r"Optional\[([^\]]+)\]", r"\1 | None", content)
    content = re.sub(r"Dict\[str, Any\]", "dict[str, Any]", content)
    content = re.sub(r"List\[str\]", "list[str]", content)

    with open(filepath, "w") as f:
        f.write(content)

    print(f"Updated {filepath}")


def main():
    """Update all MCP server files."""
    servers_dir = Path("DeepResearch/src/mcp_servers/vendored")

    # List of servers that still need updating (excluding the 5 already updated)
    servers_to_update = [
        "busco_server.py",
        "cutadapt_server.py",
        "fastp_server.py",
        "featurecounts_server.py",
        "hisat2_server.py",
        "homer_server.py",
        "htseq_server.py",
        "kallisto_server.py",
        "multiqc_server.py",
        "picard_server.py",
        "salmon_server.py",
        "star_server.py",
        "stringtie_server.py",
        "tophat_server.py",
        "trimgalore_server.py",
    ]

    for server_file in servers_to_update:
        filepath = servers_dir / server_file
        if filepath.exists():
            try:
                update_server_file(filepath)
            except Exception as e:
                print(f"Error updating {filepath}: {e}")
        else:
            print(f"File not found: {filepath}")


if __name__ == "__main__":
    main()

"""
Standalone runner for the Deeptools MCP Server.

This script can be used to run the Deeptools MCP server either as a FastMCP server
or as a standalone MCP server with Pydantic AI integration.
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add the parent directory to the path so we can import the server
sys.path.insert(0, str(Path(__file__).parent))

from deeptools_server import DeeptoolsServer  # type: ignore[import]


def main():
    parser = argparse.ArgumentParser(description="Run Deeptools MCP Server")
    parser.add_argument(
        "--mode",
        choices=["fastmcp", "mcp", "test"],
        default="fastmcp",
        help="Server mode: fastmcp (FastMCP server), mcp (MCP with Pydantic AI), test (test mode)",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port for HTTP server mode"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host for HTTP server mode")
    parser.add_argument(
        "--no-fastmcp", action="store_true", help="Disable FastMCP integration"
    )

    args = parser.parse_args()

    # Create server instance
    enable_fastmcp = not args.no_fastmcp
    server = DeeptoolsServer(enable_fastmcp=enable_fastmcp)

    print(f"Starting Deeptools MCP Server in {args.mode} mode...")
    print(f"Server info: {server.get_server_info()}")

    if args.mode == "fastmcp":
        if not enable_fastmcp:
            print("Error: FastMCP mode requires FastMCP to be enabled")
            sys.exit(1)
        print("Running FastMCP server...")
        server.run_fastmcp_server()

    elif args.mode == "mcp":
        print("Running MCP server with Pydantic AI integration...")
        # For MCP mode, you would typically integrate with an MCP client
        # This is a placeholder for the actual MCP integration
        print("MCP mode not yet implemented - use FastMCP mode instead")

    elif args.mode == "test":
        print("Running in test mode...")
        # Test some basic functionality
        tools = server.list_tools()
        print(f"Available tools: {tools}")

        info = server.get_server_info()
        print(f"Server info: {info}")

        # Test a mock operation
        result = server.run(
            {
                "operation": "compute_gc_bias",
                "bamfile": "/tmp/test.bam",
                "effective_genome_size": 3000000000,
                "genome": "/tmp/test.2bit",
                "fragment_length": 200,
            }
        )
        print(f"Test result: {result}")


if __name__ == "__main__":
    main()

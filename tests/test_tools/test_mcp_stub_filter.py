"""MCP server stub vs implemented listing."""

from __future__ import annotations

import pytest

from DeepResearch.src.datatypes.mcp import (
    MCPServerConfig,
    MCPServerStatus,
    MCPServerType,
)
from DeepResearch.src.tools.mcp_server_tools import (
    MCP_STUB_SERVER_NAMES,
    MCPServerManager,
)


def test_stub_names_disjoint_from_implemented_list() -> None:
    mgr = MCPServerManager()
    impl = set(mgr.list_implemented())
    for name in MCP_STUB_SERVER_NAMES:
        assert name not in impl


@pytest.mark.asyncio
async def test_deploy_stub_fails_fast() -> None:
    mgr = MCPServerManager()
    dep = await mgr.deploy_server(
        "bwa",
        MCPServerConfig(
            server_name="bwa",
            server_type=MCPServerType.CUSTOM,
        ),
    )
    assert dep.status == MCPServerStatus.FAILED
    assert "not implemented" in (dep.error_message or "").lower()

"""Deep agent task_tool execution via subagent registry."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from DeepResearch.src.datatypes.deep_agent_state import DeepAgentState
from DeepResearch.src.datatypes.deep_agent_tools import TaskRequestModel
from DeepResearch.src.tools.deep_agent_tools import task_tool


@pytest.mark.asyncio
async def test_task_tool_runs_registered_subagent() -> None:
    state = DeepAgentState(session_id="sess1")
    sub = MagicMock()
    sub.run = AsyncMock(return_value=SimpleNamespace(output={"answer": "yes"}))
    state.shared_state["subagent_registry"] = {"research": sub}

    ctx = MagicMock()
    ctx.state = state

    req = TaskRequestModel(
        description="Is water wet?",
        subagent_type="research",
        parameters={},
    )
    resp = await task_tool(req, ctx)
    assert resp.success
    assert resp.result is not None
    assert resp.result.get("answer") == "yes"
    sub.run.assert_awaited_once_with("Is water wet?")


@pytest.mark.asyncio
async def test_task_tool_missing_subagent() -> None:
    state = DeepAgentState(session_id="sess2")
    ctx = MagicMock()
    ctx.state = state
    req = TaskRequestModel(
        description="Hello",
        subagent_type="missing",
        parameters={},
    )
    resp = await task_tool(req, ctx)
    assert resp.success is False
    assert "No subagent" in resp.message

"""Tests for SubAgentMiddleware real agent wiring."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from DeepResearch.src.datatypes.deep_agent_types import (
    CustomSubAgent,
    ModelConfig,
    ModelProvider,
    SubAgent,
    TaskRequest,
)
from DeepResearch.src.datatypes.middleware import MiddlewareConfig, SubAgentMiddleware


@pytest.mark.asyncio
async def test_subagent_middleware_initializes_pydantic_agents() -> None:
    sub = SubAgent(
        name="worker",
        description="worker desc",
        prompt="You are a worker.",
        model=ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model_name="claude-sonnet-4-0",
        ),
    )
    with patch("pydantic_ai.Agent") as pa:
        pa.return_value = MagicMock()
        mw = SubAgentMiddleware(subagents=[sub])
        await mw._initialize_subagents()
        assert "worker" in mw._agent_registry
        pa.assert_called_once()
        call_kw = pa.call_args.kwargs
        assert call_kw["system_prompt"] == "You are a worker."
        assert call_kw["tools"] == []


def test_custom_subagent_raises() -> None:
    custom = CustomSubAgent(
        name="c",
        description="d",
        graph_config={},
        entry_point="start",
    )
    mw = SubAgentMiddleware(subagents=[custom])
    with pytest.raises(RuntimeError, match="CustomSubAgent graph execution"):
        mw._create_subagent(custom)


@pytest.mark.asyncio
async def test_execute_subagent_task_runs_agent() -> None:
    sub = SubAgent(
        name="worker",
        description="worker desc",
        prompt="You are a worker.",
        model=None,
    )
    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(
        return_value=SimpleNamespace(output={"answer": 42}),
    )
    with patch("pydantic_ai.Agent", return_value=mock_agent):
        mw = SubAgentMiddleware(subagents=[sub])
        await mw._initialize_subagents()

    task = TaskRequest(
        task_id="t1",
        description="What is 6*7?",
        subagent_type="worker",
        parameters={},
    )
    result = await mw.execute_subagent_task("worker", task, MagicMock())
    assert result.success
    assert result.result is not None
    assert result.result.get("answer") == 42
    mock_agent.run.assert_awaited_once_with("What is 6*7?")


@pytest.mark.asyncio
async def test_strict_subagent_init_reraises() -> None:
    bad = SubAgent(
        name="worker",
        description="worker desc",
        prompt="Hi",
        model=None,
    )
    cfg = MiddlewareConfig(strict_subagent_init=True)
    with patch("pydantic_ai.Agent", side_effect=OSError("no model")):
        with pytest.raises(OSError, match="no model"):
            await SubAgentMiddleware(
                subagents=[bad], config=cfg
            )._initialize_subagents()

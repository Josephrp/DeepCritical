"""Regression tests for primary workflow orchestrator failure modes."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from omegaconf import OmegaConf
from pydantic_ai.exceptions import AgentRunError

from DeepResearch.src.agents.workflow_orchestrator import PrimaryWorkflowOrchestrator
from DeepResearch.src.datatypes.llm_models import DEFAULT_PYDANTIC_AI_MODEL
from DeepResearch.src.datatypes.workflow_orchestration import (
    JudgeConfig,
    WorkflowConfig,
    WorkflowOrchestrationConfig,
    WorkflowType,
)


@pytest.fixture(autouse=True)
def _anthropic_api_key_for_agent_init(monkeypatch: pytest.MonkeyPatch) -> None:
    """Agent(model=...) validates provider env at construction time."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used-for-network")


def _minimal_config(**kwargs) -> WorkflowOrchestrationConfig:
    return WorkflowOrchestrationConfig(
        primary_workflow=WorkflowConfig(
            workflow_type=WorkflowType.PRIMARY_REACT,
            name="main",
            parameters={"model_name": DEFAULT_PYDANTIC_AI_MODEL},
        ),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_execute_primary_workflow_failure_on_agent_run_error() -> None:
    orch = PrimaryWorkflowOrchestrator(_minimal_config())
    orch.primary_agent.run = AsyncMock(side_effect=AgentRunError("model unavailable"))

    res = await orch.execute_primary_workflow("hello", OmegaConf.create({}))
    assert res["success"] is False
    assert "model unavailable" in (res.get("error") or "")
    assert res["execution_metadata"]["failure_kind"] == "agent_run_error"


@pytest.mark.asyncio
async def test_hypothesis_quality_judge_alias_when_custom_judges() -> None:
    orch = PrimaryWorkflowOrchestrator(
        _minimal_config(
            judges=[
                JudgeConfig(
                    judge_id="custom_rubric",
                    name="Custom",
                    evaluation_criteria=["x"],
                )
            ]
        )
    )
    assert "hypothesis_quality_judge" in orch.judge_registry
    assert (
        orch.judge_registry["hypothesis_quality_judge"].judge_id
        == "hypothesis_quality_judge"
    )

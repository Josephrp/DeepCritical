"""Integration-style tests: orchestrator ↔ MultiAgentCoordinator wiring (mocked)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from DeepResearch.src.agents.multi_agent_coordinator import MultiAgentCoordinator
from DeepResearch.src.agents.workflow_orchestrator import PrimaryWorkflowOrchestrator
from DeepResearch.src.datatypes.llm_models import DEFAULT_PYDANTIC_AI_MODEL
from DeepResearch.src.datatypes.multi_agent import (
    CoordinationResult,
    CoordinationStrategy,
)
from DeepResearch.src.datatypes.workflow_orchestration import (
    AgentConfig,
    AgentRole,
    MultiAgentCoordinationRequest,
    MultiAgentSystemConfig,
    WorkflowConfig,
    WorkflowOrchestrationConfig,
    WorkflowType,
)


@pytest.fixture(autouse=True)
def _anthropic_api_key_for_agent_init(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used-for-network")


def _orch_with_system() -> PrimaryWorkflowOrchestrator:
    agents = [
        AgentConfig(
            agent_id="agent_1",
            role=AgentRole.EXECUTOR,
            model_name=DEFAULT_PYDANTIC_AI_MODEL,
        )
    ]
    system = MultiAgentSystemConfig(
        system_id="sys_integration",
        name="Integration test system",
        agents=agents,
        coordination_strategy="collaborative",
        enabled=True,
    )
    return PrimaryWorkflowOrchestrator(
        WorkflowOrchestrationConfig(
            primary_workflow=WorkflowConfig(
                workflow_type=WorkflowType.PRIMARY_REACT,
                name="main",
                parameters={"model_name": DEFAULT_PYDANTIC_AI_MODEL},
            ),
            multi_agent_systems=[system],
        )
    )


@pytest.mark.asyncio
async def test_coordinate_maps_coordinator_final_result() -> None:
    fake = CoordinationResult(
        coordination_id="coord-1",
        system_id="sys_integration",
        strategy=CoordinationStrategy.COLLABORATIVE,
        success=True,
        total_rounds=2,
        final_result={"answer": "synthesized", "source": "mocked_coordinator"},
        agent_results={"agent_1": {"rounds": 2}},
        consensus_score=0.9,
        error_message=None,
    )
    orch = _orch_with_system()
    with patch.object(
        MultiAgentCoordinator,
        "coordinate",
        new_callable=AsyncMock,
        return_value=fake,
    ):
        req = MultiAgentCoordinationRequest(
            system_id="sys_integration",
            task_description="Solve X",
            input_data={"question": "Solve X"},
            coordination_strategy="collaborative",
            max_rounds=3,
        )
        res = await orch._coordinate_multi_agent_system(req)
    assert res.success is True
    assert res.final_result.get("answer") == "synthesized"
    assert res.coordination_rounds == 2
    assert res.consensus_score == 0.9
    assert res.error_message is None


@pytest.mark.asyncio
async def test_coordinate_propagates_coordinator_error_message() -> None:
    fake = CoordinationResult(
        coordination_id="coord-2",
        system_id="sys_integration",
        strategy=CoordinationStrategy.COLLABORATIVE,
        success=False,
        total_rounds=0,
        final_result={},
        agent_results={},
        consensus_score=0.0,
        error_message="coordination aborted: test",
    )
    orch = _orch_with_system()
    with patch.object(
        MultiAgentCoordinator,
        "coordinate",
        new_callable=AsyncMock,
        return_value=fake,
    ):
        req = MultiAgentCoordinationRequest(
            system_id="sys_integration",
            task_description="Task",
            input_data={},
            max_rounds=1,
        )
        res = await orch._coordinate_multi_agent_system(req)
    assert res.success is False
    assert res.final_result == {}
    assert res.error_message == "coordination aborted: test"

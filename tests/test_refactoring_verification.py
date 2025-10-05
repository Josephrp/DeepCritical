"""
Verification tests for the refactoring of agent_orchestrator.py.

This module tests that the refactoring to move prompts and types to their
respective directories was successful and all imports work correctly.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_refactoring_verification():
    """Test that all refactored components work correctly."""

    # Test datatypes imports
    print("Testing datatypes imports...")
    from DeepResearch.src.datatypes.workflow_orchestration import (
        OrchestratorDependencies,
        NestedLoopRequest,
        SubgraphSpawnRequest,
        BreakConditionCheck,
        OrchestrationResult,
    )

    assert OrchestratorDependencies is not None
    assert NestedLoopRequest is not None
    assert SubgraphSpawnRequest is not None
    assert BreakConditionCheck is not None
    assert OrchestrationResult is not None
    print("+ Workflow orchestration types import successfully")

    # Test main datatypes package
    print("Testing main datatypes package...")
    from DeepResearch.src.datatypes import (
        OrchestratorDependencies as OD1,
        NestedLoopRequest as NLR1,
        SubgraphSpawnRequest as SSR1,
        BreakConditionCheck as BCC1,
        OrchestrationResult as OR1,
    )

    assert OD1 is not None
    assert NLR1 is not None
    assert SSR1 is not None
    assert BCC1 is not None
    assert OR1 is not None
    print("+ All types available from main datatypes package")

    # Test prompts
    print("Testing prompts...")
    from DeepResearch.src.prompts.orchestrator import (
        ORCHESTRATOR_SYSTEM_PROMPT,
        ORCHESTRATOR_INSTRUCTIONS,
        OrchestratorPrompts,
    )
    from DeepResearch.src.prompts.workflow_orchestrator import (
        WORKFLOW_ORCHESTRATOR_SYSTEM_PROMPT,
        WORKFLOW_ORCHESTRATOR_INSTRUCTIONS,
        WorkflowOrchestratorPrompts,
    )

    assert ORCHESTRATOR_SYSTEM_PROMPT is not None
    assert ORCHESTRATOR_INSTRUCTIONS is not None
    assert OrchestratorPrompts is not None
    assert isinstance(ORCHESTRATOR_SYSTEM_PROMPT, str)
    assert isinstance(ORCHESTRATOR_INSTRUCTIONS, list)
    print("+ Orchestrator prompts import successfully")
    assert WORKFLOW_ORCHESTRATOR_SYSTEM_PROMPT is not None
    assert WORKFLOW_ORCHESTRATOR_INSTRUCTIONS is not None
    assert WorkflowOrchestratorPrompts is not None
    assert isinstance(WORKFLOW_ORCHESTRATOR_SYSTEM_PROMPT, str)
    assert isinstance(WORKFLOW_ORCHESTRATOR_INSTRUCTIONS, list)
    print("+ Workflow orchestrator prompts import successfully")

    # Test agent orchestrator
    print("Testing agent orchestrator...")
    from DeepResearch.src.agents.agent_orchestrator import AgentOrchestrator

    assert AgentOrchestrator is not None
    print("+ AgentOrchestrator imports successfully")

    print(
        "All refactoring tests passed! The refactoring is complete and working correctly."
    )
    return True


if __name__ == "__main__":
    test_refactoring_verification()

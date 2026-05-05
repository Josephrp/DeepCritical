from __future__ import annotations

import pytest

from DeepResearch.agents import create_agent
from DeepResearch.src.datatypes.agent_taxonomy import (
    DEEP_AGENT_TYPES,
    WORKFLOW_ONLY_FLOW_KEYS,
)
from DeepResearch.src.datatypes.agents import AgentType


def test_deep_agent_types_is_subset_of_enum() -> None:
    for t in DEEP_AGENT_TYPES:
        assert isinstance(t, AgentType)


def test_workflow_only_flow_keys() -> None:
    assert "literature_review" in WORKFLOW_ONLY_FLOW_KEYS
    assert AgentType.PARSER not in DEEP_AGENT_TYPES  # sanity


def test_create_agent_deep_requires_config_gate() -> None:
    with pytest.raises(ValueError, match="deep_agent"):
        create_agent(AgentType.DEEP_AGENT_GENERAL)

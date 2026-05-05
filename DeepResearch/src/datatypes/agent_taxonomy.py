"""Reference for overlapping agent and flow naming across DeepCritical.

- :class:`~DeepResearch.src.datatypes.agents.AgentType` names the
  ``DeepResearch.agents.BaseAgent`` family and keys in
  ``MultiAgentOrchestrator.agents`` (except types gated by config).
- :class:`~DeepResearch.src.datatypes.workflow_orchestration.AgentRole` names
  roles in primary REACT / nested orchestration (different string values).
- Hypothesis and literature-review stacks are driven by Hydra ``flows.*`` and
  dedicated apps; they are **not** ``AgentType`` enum members.
"""

from __future__ import annotations

from DeepResearch.src.datatypes.agents import AgentType

DEEP_AGENT_TYPES: frozenset[AgentType] = frozenset(
    {
        AgentType.DEEP_AGENT_PLANNING,
        AgentType.DEEP_AGENT_FILESYSTEM,
        AgentType.DEEP_AGENT_RESEARCH,
        AgentType.DEEP_AGENT_ORCHESTRATION,
        AgentType.DEEP_AGENT_GENERAL,
    }
)

# ``configs/config.yaml`` keys under ``flows`` without a matching ``AgentType``.
WORKFLOW_ONLY_FLOW_KEYS: frozenset[str] = frozenset(
    {
        "hypothesis_generation",
        "hypothesis_testing",
        "literature_review",
    }
)

# ``AgentType.ORCHESTRATOR`` is used by workflow-pattern / collaborative agents.
# It is **not** wired into ``MultiAgentOrchestrator._initialize_agents``. For
# nested REACT, use ``AgentRole.ORCHESTRATOR_AGENT`` in orchestration configs.

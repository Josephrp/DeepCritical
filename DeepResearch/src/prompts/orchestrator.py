from typing import Dict


STYLE = "concise"
MAX_STEPS = 3


ORCHESTRATOR_PROMPTS: Dict[str, str] = {
    "style": STYLE,
    "max_steps": str(MAX_STEPS),
    "orchestrate_workflow": "Orchestrate the following workflow: {workflow_description}",
    "coordinate_agents": "Coordinate multiple agents for the task: {task_description}",
}


class OrchestratorPrompts:
    """Prompt templates for orchestrator operations."""

    STYLE = STYLE
    MAX_STEPS = MAX_STEPS
    PROMPTS = ORCHESTRATOR_PROMPTS

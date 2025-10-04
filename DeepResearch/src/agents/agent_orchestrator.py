"""
Agent-based orchestrator for DeepCritical's enhanced REACT architecture.

This module implements orchestrators as agents that can spawn nested REACT loops,
manage subgraphs, and coordinate multi-statemachine workflows with configurable
break conditions and loss functions.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from dataclasses import dataclass, field
from omegaconf import DictConfig

from pydantic_ai import Agent, RunContext
from pydantic import BaseModel, Field

from ..datatypes.workflow_orchestration import (
    AgentOrchestratorConfig,
    NestedReactConfig,
    SubgraphConfig,
    BreakCondition,
    MultiStateMachineMode,
    SubgraphType,
    LossFunctionType,
    AgentRole,
)

if TYPE_CHECKING:
    pass


class OrchestratorDependencies(BaseModel):
    """Dependencies for the agent orchestrator."""

    config: Dict[str, Any] = Field(default_factory=dict)
    user_input: str = Field(..., description="User input/query")
    context: Dict[str, Any] = Field(default_factory=dict)
    available_subgraphs: List[str] = Field(default_factory=list)
    available_agents: List[str] = Field(default_factory=list)
    current_iteration: int = Field(0, description="Current iteration number")
    parent_loop_id: Optional[str] = Field(None, description="Parent loop ID if nested")


class NestedLoopRequest(BaseModel):
    """Request to spawn a nested REACT loop."""

    loop_id: str = Field(..., description="Loop identifier")
    parent_loop_id: Optional[str] = Field(None, description="Parent loop ID")
    max_iterations: int = Field(10, description="Maximum iterations")
    break_conditions: List[BreakCondition] = Field(
        default_factory=list, description="Break conditions"
    )
    state_machine_mode: MultiStateMachineMode = Field(
        MultiStateMachineMode.GROUP_CHAT, description="State machine mode"
    )
    subgraphs: List[SubgraphType] = Field(
        default_factory=list, description="Subgraphs to include"
    )
    agent_roles: List[AgentRole] = Field(
        default_factory=list, description="Agent roles"
    )
    tools: List[str] = Field(default_factory=list, description="Available tools")
    priority: int = Field(0, description="Execution priority")


class SubgraphSpawnRequest(BaseModel):
    """Request to spawn a subgraph."""

    subgraph_id: str = Field(..., description="Subgraph identifier")
    subgraph_type: SubgraphType = Field(..., description="Type of subgraph")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Subgraph parameters"
    )
    entry_node: str = Field(..., description="Entry node")
    max_execution_time: float = Field(300.0, description="Maximum execution time")
    tools: List[str] = Field(default_factory=list, description="Available tools")


class BreakConditionCheck(BaseModel):
    """Result of break condition evaluation."""

    condition_met: bool = Field(..., description="Whether the condition is met")
    condition_type: LossFunctionType = Field(..., description="Type of condition")
    current_value: float = Field(..., description="Current value")
    threshold: float = Field(..., description="Threshold value")
    should_break: bool = Field(..., description="Whether to break the loop")


class OrchestrationResult(BaseModel):
    """Result of orchestration execution."""

    success: bool = Field(..., description="Whether orchestration was successful")
    final_answer: str = Field(..., description="Final answer")
    nested_loops_spawned: List[str] = Field(
        default_factory=list, description="Nested loops spawned"
    )
    subgraphs_executed: List[str] = Field(
        default_factory=list, description="Subgraphs executed"
    )
    total_iterations: int = Field(..., description="Total iterations")
    break_reason: Optional[str] = Field(None, description="Reason for breaking")
    execution_metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Execution metadata"
    )


@dataclass
class AgentOrchestrator:
    """Agent-based orchestrator that can spawn nested REACT loops and manage subgraphs."""

    config: AgentOrchestratorConfig
    nested_loops: Dict[str, NestedReactConfig] = field(default_factory=dict)
    subgraphs: Dict[str, SubgraphConfig] = field(default_factory=dict)
    active_loops: Dict[str, Any] = field(default_factory=dict)
    execution_history: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        """Initialize the agent orchestrator."""
        self._create_orchestrator_agent()
        self._register_orchestrator_tools()

    def _create_orchestrator_agent(self):
        """Create the orchestrator agent."""
        self.orchestrator_agent = Agent(
            model_name=self.config.model_name,
            deps_type=OrchestratorDependencies,
            system_prompt=self._get_orchestrator_system_prompt(),
            instructions=self._get_orchestrator_instructions(),
        )

    def _get_orchestrator_system_prompt(self) -> str:
        """Get the system prompt for the orchestrator agent."""
        return f"""You are an advanced orchestrator agent responsible for managing nested REACT loops and subgraphs.

Your capabilities include:
1. Spawning nested REACT loops with different state machine modes
2. Managing subgraphs for specialized workflows (RAG, search, code, etc.)
3. Coordinating multi-agent systems with configurable strategies
4. Evaluating break conditions and loss functions
5. Making decisions about when to continue or terminate loops

You have access to various tools for:
- Spawning nested loops with specific configurations
- Executing subgraphs with different parameters
- Checking break conditions and loss functions
- Coordinating agent interactions
- Managing workflow execution

Your role is to analyze the user input and orchestrate the most appropriate combination of nested loops and subgraphs to achieve the desired outcome.

Current configuration:
- Max nested loops: {self.config.max_nested_loops}
- Coordination strategy: {self.config.coordination_strategy}
- Can spawn subgraphs: {self.config.can_spawn_subgraphs}
- Can spawn agents: {self.config.can_spawn_agents}"""

    def _get_orchestrator_instructions(self) -> List[str]:
        """Get instructions for the orchestrator agent."""
        return [
            "Analyze the user input to understand the complexity and requirements",
            "Determine if nested REACT loops are needed based on the task complexity",
            "Select appropriate state machine modes (group_chat, sequential, hierarchical, etc.)",
            "Choose relevant subgraphs (RAG, search, code, bioinformatics, etc.)",
            "Configure break conditions and loss functions appropriately",
            "Spawn nested loops and subgraphs as needed",
            "Monitor execution and evaluate break conditions",
            "Coordinate between different loops and subgraphs",
            "Synthesize results from multiple sources",
            "Make decisions about when to terminate or continue execution",
        ]

    def _register_orchestrator_tools(self):
        """Register tools for the orchestrator agent."""

        @self.orchestrator_agent.tool
        def spawn_nested_loop(
            ctx: RunContext[OrchestratorDependencies],
            loop_id: str,
            state_machine_mode: str,
            max_iterations: int = 10,
            subgraphs: List[str] = None,
            agent_roles: List[str] = None,
            tools: List[str] = None,
            priority: int = 0,
        ) -> Dict[str, Any]:
            """Spawn a nested REACT loop."""
            try:
                # Create nested loop configuration
                nested_config = NestedReactConfig(
                    loop_id=loop_id,
                    parent_loop_id=ctx.deps.parent_loop_id,
                    max_iterations=max_iterations,
                    state_machine_mode=MultiStateMachineMode(state_machine_mode),
                    subgraphs=[SubgraphType(sg) for sg in (subgraphs or [])],
                    agent_roles=[AgentRole(role) for role in (agent_roles or [])],
                    tools=tools or [],
                    priority=priority,
                )

                # Add to nested loops
                self.nested_loops[loop_id] = nested_config

                # Spawn the actual loop
                loop_result = self._spawn_nested_loop(nested_config, ctx.deps)

                return {
                    "success": True,
                    "loop_id": loop_id,
                    "result": loop_result,
                    "message": f"Nested loop {loop_id} spawned successfully",
                }

            except Exception as e:
                return {
                    "success": False,
                    "loop_id": loop_id,
                    "error": str(e),
                    "message": f"Failed to spawn nested loop {loop_id}",
                }

        @self.orchestrator_agent.tool
        def execute_subgraph(
            ctx: RunContext[OrchestratorDependencies],
            subgraph_id: str,
            subgraph_type: str,
            parameters: Dict[str, Any] = None,
            entry_node: str = "start",
            max_execution_time: float = 300.0,
            tools: List[str] = None,
        ) -> Dict[str, Any]:
            """Execute a subgraph."""
            try:
                # Create subgraph configuration
                subgraph_config = SubgraphConfig(
                    subgraph_id=subgraph_id,
                    subgraph_type=SubgraphType(subgraph_type),
                    state_machine_path=f"DeepResearch.src.statemachines.{subgraph_type}_workflow",
                    entry_node=entry_node,
                    exit_node="end",
                    parameters=parameters or {},
                    tools=tools or [],
                    max_execution_time=max_execution_time,
                )

                # Add to subgraphs
                self.subgraphs[subgraph_id] = subgraph_config

                # Execute the subgraph
                subgraph_result = self._execute_subgraph(subgraph_config, ctx.deps)

                return {
                    "success": True,
                    "subgraph_id": subgraph_id,
                    "result": subgraph_result,
                    "message": f"Subgraph {subgraph_id} executed successfully",
                }

            except Exception as e:
                return {
                    "success": False,
                    "subgraph_id": subgraph_id,
                    "error": str(e),
                    "message": f"Failed to execute subgraph {subgraph_id}",
                }

        @self.orchestrator_agent.tool
        def check_break_conditions(
            ctx: RunContext[OrchestratorDependencies],
            current_iteration: int,
            current_metrics: Dict[str, Any],
        ) -> Dict[str, Any]:
            """Check break conditions for the current loop."""
            try:
                break_results = []
                should_break = False
                break_reason = None

                for condition in self.config.break_conditions:
                    if not condition.enabled:
                        continue

                    result = self._evaluate_break_condition(
                        condition, current_iteration, current_metrics
                    )
                    break_results.append(result)

                    if result.should_break:
                        should_break = True
                        break_reason = (
                            f"Break condition met: {condition.condition_type.value}"
                        )
                        break

                return {
                    "should_break": should_break,
                    "break_reason": break_reason,
                    "break_results": [r.dict() for r in break_results],
                    "current_iteration": current_iteration,
                }

            except Exception as e:
                return {
                    "should_break": False,
                    "error": str(e),
                    "current_iteration": current_iteration,
                }

        @self.orchestrator_agent.tool
        def coordinate_agents(
            ctx: RunContext[OrchestratorDependencies],
            coordination_strategy: str,
            agent_roles: List[str],
            task_description: str,
        ) -> Dict[str, Any]:
            """Coordinate agents using the specified strategy."""
            try:
                # This would integrate with MultiAgentCoordinator
                coordination_result = self._coordinate_agents(
                    coordination_strategy, agent_roles, task_description, ctx.deps
                )

                return {
                    "success": True,
                    "coordination_strategy": coordination_strategy,
                    "result": coordination_result,
                    "message": f"Agent coordination completed using {coordination_strategy}",
                }

            except Exception as e:
                return {
                    "success": False,
                    "coordination_strategy": coordination_strategy,
                    "error": str(e),
                    "message": f"Agent coordination failed: {str(e)}",
                }

    async def execute_orchestration(
        self, user_input: str, config: DictConfig, max_iterations: Optional[int] = None
    ) -> OrchestrationResult:
        """Execute the orchestration with nested loops and subgraphs."""
        start_time = time.time()
        max_iterations = max_iterations or self.config.max_nested_loops

        # Create dependencies
        deps = OrchestratorDependencies(
            config=config,
            user_input=user_input,
            context={"execution_start": datetime.now().isoformat()},
            current_iteration=0,
        )

        try:
            # Execute the orchestrator agent
            result = await self.orchestrator_agent.run(user_input, deps=deps)

            # Process results and create final answer
            final_answer = self._synthesize_results(result, user_input)

            execution_time = time.time() - start_time

            return OrchestrationResult(
                success=True,
                final_answer=final_answer,
                nested_loops_spawned=list(self.nested_loops.keys()),
                subgraphs_executed=list(self.subgraphs.keys()),
                total_iterations=deps.current_iteration,
                execution_metadata={
                    "execution_time": execution_time,
                    "nested_loops_count": len(self.nested_loops),
                    "subgraphs_count": len(self.subgraphs),
                    "orchestrator_id": self.config.orchestrator_id,
                },
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return OrchestrationResult(
                success=False,
                final_answer=f"Orchestration failed: {str(e)}",
                total_iterations=deps.current_iteration,
                break_reason=f"Error: {str(e)}",
                execution_metadata={"execution_time": execution_time, "error": str(e)},
            )

    def _spawn_nested_loop(
        self, config: NestedReactConfig, deps: OrchestratorDependencies
    ) -> Dict[str, Any]:
        """Spawn a nested REACT loop."""
        # This would create and execute a nested REACT loop
        # For now, return a placeholder
        return {
            "loop_id": config.loop_id,
            "state_machine_mode": config.state_machine_mode.value,
            "status": "spawned",
            "subgraphs": [sg.value for sg in config.subgraphs],
            "agent_roles": [role.value for role in config.agent_roles],
        }

    def _execute_subgraph(
        self, config: SubgraphConfig, deps: OrchestratorDependencies
    ) -> Dict[str, Any]:
        """Execute a subgraph."""
        # This would execute the actual subgraph
        # For now, return a placeholder
        return {
            "subgraph_id": config.subgraph_id,
            "subgraph_type": config.subgraph_type.value,
            "status": "executed",
            "parameters": config.parameters,
            "execution_time": 0.0,
        }

    def _evaluate_break_condition(
        self,
        condition: BreakCondition,
        current_iteration: int,
        current_metrics: Dict[str, Any],
    ) -> BreakConditionCheck:
        """Evaluate a break condition."""
        current_value = 0.0

        if condition.condition_type == LossFunctionType.ITERATION_LIMIT:
            current_value = current_iteration
        elif condition.condition_type == LossFunctionType.CONFIDENCE_THRESHOLD:
            current_value = current_metrics.get("confidence", 0.0)
        elif condition.condition_type == LossFunctionType.QUALITY_SCORE:
            current_value = current_metrics.get("quality_score", 0.0)
        elif condition.condition_type == LossFunctionType.CONSENSUS_LEVEL:
            current_value = current_metrics.get("consensus_level", 0.0)
        elif condition.condition_type == LossFunctionType.TIME_LIMIT:
            current_value = current_metrics.get("execution_time", 0.0)

        # Evaluate the condition
        condition_met = False
        if condition.operator == ">=":
            condition_met = current_value >= condition.threshold
        elif condition.operator == "<=":
            condition_met = current_value <= condition.threshold
        elif condition.operator == "==":
            condition_met = current_value == condition.threshold
        elif condition.operator == "!=":
            condition_met = current_value != condition.threshold

        return BreakConditionCheck(
            condition_met=condition_met,
            condition_type=condition.condition_type,
            current_value=current_value,
            threshold=condition.threshold,
            should_break=condition_met,
        )

    def _coordinate_agents(
        self,
        coordination_strategy: str,
        agent_roles: List[str],
        task_description: str,
        deps: OrchestratorDependencies,
    ) -> Dict[str, Any]:
        """Coordinate agents using the specified strategy."""
        # This would integrate with MultiAgentCoordinator
        # For now, return a placeholder
        return {
            "coordination_strategy": coordination_strategy,
            "agent_roles": agent_roles,
            "task_description": task_description,
            "result": "placeholder_coordination_result",
        }

    def _synthesize_results(self, result: Any, user_input: str) -> str:
        """Synthesize results from orchestration."""
        # This would synthesize results from all nested loops and subgraphs
        # For now, return a basic synthesis
        return f"""# Orchestration Results

**Question:** {user_input}

**Orchestration Summary:**
- Nested Loops Spawned: {len(self.nested_loops)}
- Subgraphs Executed: {len(self.subgraphs)}
- Total Iterations: {len(self.execution_history)}

**Nested Loops:**
{chr(10).join([f"- {loop_id}: {config.state_machine_mode.value}" for loop_id, config in self.nested_loops.items()])}

**Subgraphs:**
{chr(10).join([f"- {subgraph_id}: {config.subgraph_type.value}" for subgraph_id, config in self.subgraphs.items()])}

**Final Result:**
{str(result) if result else "Orchestration completed successfully"}"""

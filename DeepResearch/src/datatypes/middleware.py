"""
Middleware data types for DeepCritical agent middleware system.

This module defines Pydantic models for middleware components including
planning, filesystem, subagent orchestration, summarization, and prompt caching.
"""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

# Import existing DeepCritical types
from .deep_agent_types import (
    CustomSubAgent,
    ModelConfig,
    ModelProvider,
    SubAgent,
    TaskRequest,
    TaskResult,
)
from .llm_models import DEFAULT_PYDANTIC_AI_MODEL

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable

    from pydantic_ai import Agent, RunContext

    from .deep_agent_state import DeepAgentState


class MiddlewareConfig(BaseModel):
    """Configuration for middleware components."""

    enabled: bool = Field(True, description="Whether middleware is enabled")
    priority: int = Field(
        0, description="Middleware priority (higher = earlier execution)"
    )
    timeout: float = Field(30.0, gt=0, description="Middleware timeout in seconds")
    retry_attempts: int = Field(3, ge=0, description="Number of retry attempts")
    retry_delay: float = Field(1.0, gt=0, description="Delay between retries")
    strict_subagent_init: bool = Field(
        False,
        description="If True, subagent construction failures propagate instead of logging",
    )

    model_config = ConfigDict(json_schema_extra={})


class MiddlewareResult(BaseModel):
    """Result from middleware execution."""

    success: bool = Field(..., description="Whether middleware succeeded")
    modified_state: bool = Field(False, description="Whether state was modified")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Middleware metadata"
    )
    error: str | None = Field(None, description="Error message if failed")
    execution_time: float = Field(0.0, description="Execution time in seconds")


class BaseMiddleware:
    """Base class for all middleware components."""

    def __init__(self, config: MiddlewareConfig | None = None):
        self.config = config or MiddlewareConfig()
        self.name = self.__class__.__name__

    async def process(
        self, agent: Agent | None, state: DeepAgentState, **kwargs
    ) -> MiddlewareResult:
        """Process the middleware logic.

        Args:
            agent: The agent instance (may be None if not available)
            state: The current agent state
            **kwargs: Additional middleware-specific arguments

        Returns:
            MiddlewareResult with execution status
        """
        start_time = time.time()
        try:
            if not self.config.enabled:
                return MiddlewareResult(
                    success=True,
                    modified_state=False,
                    metadata={"skipped": True, "reason": "disabled"},
                )

            result = await self._execute(agent, state, **kwargs)
            execution_time = time.time() - start_time

            return MiddlewareResult(
                success=True,
                modified_state=result.get("modified_state", False),
                metadata=result.get("metadata", {}),
                execution_time=execution_time,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return MiddlewareResult(
                success=False,
                modified_state=False,
                error=str(e),
                execution_time=execution_time,
            )

    async def _execute(
        self, agent: Agent | None, state: DeepAgentState, **kwargs
    ) -> dict[str, Any]:
        """Execute the middleware logic. Override in subclasses."""
        return {"modified_state": False, "metadata": {}}


class PlanningMiddleware(BaseMiddleware):
    """Middleware for planning operations and todo management."""

    def __init__(self, config: MiddlewareConfig | None = None):
        super().__init__(config)
        # Import here to avoid circular imports
        from DeepResearch.src.tools.deep_agent_tools import write_todos_tool

        self.tools = [write_todos_tool]

    async def _execute(
        self, agent: Agent | None, state: DeepAgentState, **kwargs
    ) -> dict[str, Any]:
        """Execute planning middleware logic."""
        # Register planning tools with the agent
        if agent is not None:
            for tool in self.tools:
                if hasattr(agent, "add_tool"):
                    add_tool_method = getattr(agent, "add_tool", None)
                    if add_tool_method is not None and callable(add_tool_method):
                        add_tool_method(tool)

        # Add planning context to system prompt
        planning_state = state.get_planning_state()
        if planning_state.todos:
            todo_summary = f"Current todos: {len(planning_state.todos)} total, {len(planning_state.get_pending_todos())} pending, {len(planning_state.get_in_progress_todos())} in progress"
            state.shared_state["planning_summary"] = todo_summary

        return {
            "modified_state": True,
            "metadata": {
                "todos_count": len(planning_state.todos),
            },
        }


class FilesystemMiddleware(BaseMiddleware):
    """Middleware for filesystem operations."""

    def __init__(self, config: MiddlewareConfig | None = None):
        super().__init__(config)
        # Import here to avoid circular imports
        from DeepResearch.src.tools.deep_agent_tools import (
            edit_file_tool,
            list_files_tool,
            read_file_tool,
            write_file_tool,
        )

        self.tools = [list_files_tool, read_file_tool, write_file_tool, edit_file_tool]

    async def _execute(
        self, agent: Agent | None, state: DeepAgentState, **kwargs
    ) -> dict[str, Any]:
        """Execute filesystem middleware logic."""
        # Register filesystem tools with the agent
        if agent is not None:
            for tool in self.tools:
                if hasattr(agent, "add_tool"):
                    add_tool_method = getattr(agent, "add_tool", None)
                    if add_tool_method is not None and callable(add_tool_method):
                        add_tool_method(tool)

        # Add filesystem context to system prompt
        filesystem_state = state.get_filesystem_state()
        if filesystem_state.files:
            file_summary = (
                f"Available files: {len(filesystem_state.files)} files in filesystem"
            )
            state.shared_state["filesystem_summary"] = file_summary

        return {
            "modified_state": True,
            "metadata": {
                "files_count": len(filesystem_state.files),
            },
        }


class SubAgentMiddleware(BaseMiddleware):
    """Middleware for subagent orchestration."""

    def __init__(
        self,
        subagents: list[SubAgent | CustomSubAgent] | None = None,
        default_tools: list[Callable] | None = None,
        config: MiddlewareConfig | None = None,
    ):
        super().__init__(config)
        self.subagents = subagents or []
        self.default_tools = default_tools or []
        # Import here to avoid circular imports
        from DeepResearch.src.tools.deep_agent_tools import task_tool

        self.tools = [task_tool]
        self._agent_registry: dict[str, Agent] = {}

    @staticmethod
    def _pydantic_ai_model_id(mc: ModelConfig | None) -> str:
        if mc is None:
            return DEFAULT_PYDANTIC_AI_MODEL
        prefix_by_provider: dict[ModelProvider, str] = {
            ModelProvider.ANTHROPIC: "anthropic",
            ModelProvider.OPENAI: "openai",
            ModelProvider.VLLM: "openai",
            ModelProvider.HUGGINGFACE: "openai",
            ModelProvider.CUSTOM: "openai",
        }
        prefix = prefix_by_provider.get(mc.provider, "openai")
        return f"{prefix}:{mc.model_name}"

    async def _execute(
        self, agent: Agent | None, state: DeepAgentState, **kwargs
    ) -> dict[str, Any]:
        """Execute subagent middleware logic."""
        # Register task tool with the agent
        if agent is not None:
            for tool in self.tools:
                if hasattr(agent, "add_tool"):
                    add_tool_method = getattr(agent, "add_tool", None)
                    if add_tool_method is not None and callable(add_tool_method):
                        add_tool_method(tool)

        # Initialize subagents if not already done
        if not self._agent_registry:
            await self._initialize_subagents()

        state.shared_state["subagent_registry"] = self._agent_registry

        # Add subagent context to system prompt
        subagent_descriptions = [
            f"- {sa.name}: {sa.description}" for sa in self.subagents
        ]
        if subagent_descriptions:
            state.shared_state["available_subagents"] = subagent_descriptions

        return {
            "modified_state": True,
            "metadata": {
                "subagents_available": len(self.subagents),
                "agent_registry_size": len(self._agent_registry),
            },
        }

    async def _initialize_subagents(self) -> None:
        """Initialize subagent registry."""
        strict = bool(
            self.config and getattr(self.config, "strict_subagent_init", False)
        )
        for subagent in self.subagents:
            try:
                agent = self._create_subagent(subagent)
                self._agent_registry[subagent.name] = agent
            except Exception:
                logger.exception(
                    "Failed to initialize subagent %r", getattr(subagent, "name", "?")
                )
                if strict:
                    raise

    def _create_subagent(self, subagent: SubAgent | CustomSubAgent) -> Agent:
        """Create a pydantic-ai Agent for a configured subagent."""
        if isinstance(subagent, CustomSubAgent):
            msg = (
                "CustomSubAgent graph execution is not implemented; "
                "use SubAgent with model and prompt instead."
            )
            raise RuntimeError(msg)

        from pydantic_ai import Agent as PydanticAgent

        model_id = self._pydantic_ai_model_id(subagent.model)
        tools = list(self.default_tools) if self.default_tools else []
        return PydanticAgent(model_id, system_prompt=subagent.prompt, tools=tools)

    async def execute_subagent_task(
        self, subagent_name: str, task: TaskRequest, context: DeepAgentState
    ) -> TaskResult:
        """Execute a task with a specific subagent."""
        if subagent_name not in self._agent_registry:
            return TaskResult(
                task_id=task.task_id,
                success=False,
                error=f"Subagent {subagent_name} not found",
                execution_time=0.0,
                subagent_used=subagent_name,
            )

        sub = self._agent_registry[subagent_name]
        start_time = time.time()
        try:
            run_out = await sub.run(task.description)
            payload = getattr(run_out, "output", run_out)
            base: dict[str, Any] = (
                dict(payload) if isinstance(payload, dict) else {"output": payload}
            )
            result_data = {
                **base,
                "task_id": task.task_id,
                "description": task.description,
                "subagent_type": subagent_name,
            }

            execution_time = time.time() - start_time

            return TaskResult(
                task_id=task.task_id,
                success=True,
                result=result_data,
                execution_time=execution_time,
                subagent_used=subagent_name,
                metadata={"middleware": "SubAgentMiddleware"},
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return TaskResult(
                task_id=task.task_id,
                success=False,
                error=str(e),
                execution_time=execution_time,
                subagent_used=subagent_name,
            )


class SummarizationMiddleware(BaseMiddleware):
    """Middleware for conversation summarization."""

    def __init__(
        self,
        max_tokens_before_summary: int = 120000,
        messages_to_keep: int = 20,
        config: MiddlewareConfig | None = None,
    ):
        super().__init__(config)
        self.max_tokens_before_summary = max_tokens_before_summary
        self.messages_to_keep = messages_to_keep

    async def _execute(
        self, agent: Agent | None, state: DeepAgentState, **kwargs
    ) -> dict[str, Any]:
        """Execute summarization middleware logic."""
        # Check if conversation history needs summarization
        conversation_history = state.conversation_history

        if len(conversation_history) > self.messages_to_keep:
            # Estimate token count (rough approximation)
            total_tokens = sum(
                len(str(msg.get("content", ""))) // 4  # Rough token estimation
                for msg in conversation_history
            )

            if total_tokens > self.max_tokens_before_summary:
                # Summarize older messages
                messages_to_summarize = conversation_history[: -self.messages_to_keep]
                recent_messages = conversation_history[-self.messages_to_keep :]

                # Create summary (simplified implementation)
                summary = {
                    "role": "system",
                    "content": f"Previous conversation summarized ({len(messages_to_summarize)} messages)",
                    "timestamp": time.time(),
                }

                # Update conversation history
                state.conversation_history = [summary, *recent_messages]

                return {
                    "modified_state": True,
                    "metadata": {
                        "messages_summarized": len(messages_to_summarize),
                        "messages_kept": len(recent_messages),
                        "total_tokens_before": total_tokens,
                    },
                }

        return {
            "modified_state": False,
            "metadata": {
                "messages_count": len(conversation_history),
                "summarization_needed": False,
            },
        }


class PromptCachingMiddleware(BaseMiddleware):
    """Middleware for prompt caching."""

    def __init__(
        self,
        ttl: str = "5m",
        unsupported_model_behavior: str = "ignore",
        config: MiddlewareConfig | None = None,
    ):
        super().__init__(config)
        self.ttl = ttl
        self.unsupported_model_behavior = unsupported_model_behavior
        self._cache: dict[str, Any] = {}

    async def _execute(
        self, agent: Agent | None, state: DeepAgentState, **kwargs
    ) -> dict[str, Any]:
        """Execute prompt caching middleware logic."""
        # This is a simplified implementation
        # In practice, you would implement proper prompt caching

        cache_key = self._generate_cache_key(state)

        if cache_key in self._cache:
            # Use cached result
            self._cache[cache_key]
            return {
                "modified_state": False,
                "metadata": {"cache_hit": True, "cache_key": cache_key},
            }
        # Cache miss - will be handled by the agent execution
        return {
            "modified_state": False,
            "metadata": {"cache_hit": False, "cache_key": cache_key},
        }

    def _generate_cache_key(self, state: DeepAgentState) -> str:
        """Generate a cache key for the current context."""
        # Simplified cache key generation
        # In practice, this would be more sophisticated
        return f"prompt_cache_{hash(str(state.conversation_history[-5:]))}"


class MiddlewarePipeline:
    """Pipeline for managing multiple middleware components."""

    def __init__(self, middleware: list[BaseMiddleware] | None = None):
        self.middleware = middleware or []
        # Sort by priority (higher priority first)
        self.middleware.sort(key=lambda m: m.config.priority, reverse=True)

    def add_middleware(self, middleware: BaseMiddleware) -> None:
        """Add middleware to the pipeline."""
        self.middleware.append(middleware)
        # Re-sort by priority
        self.middleware.sort(key=lambda m: m.config.priority, reverse=True)

    async def process(
        self, agent: Agent | None, state: DeepAgentState, **kwargs
    ) -> list[MiddlewareResult]:
        """Process all middleware in the pipeline.

        Args:
            agent: The agent instance (may be None if not available)
            state: The current agent state
            **kwargs: Additional middleware-specific arguments

        Returns:
            List of MiddlewareResult from each middleware in the pipeline
        """
        results = []

        for middleware in self.middleware:
            try:
                result = await middleware.process(agent, state, **kwargs)
                results.append(result)

                # If middleware failed and is critical, stop processing
                if not result.success and middleware.config.priority > 0:
                    break

            except Exception as e:
                results.append(
                    MiddlewareResult(
                        success=False,
                        error=f"Middleware {middleware.name} failed: {e!s}",
                    )
                )

        return results


# Factory functions for creating middleware
def create_planning_middleware(
    config: MiddlewareConfig | None = None,
) -> PlanningMiddleware:
    """Create a planning middleware instance."""
    return PlanningMiddleware(config)


def create_filesystem_middleware(
    config: MiddlewareConfig | None = None,
) -> FilesystemMiddleware:
    """Create a filesystem middleware instance."""
    return FilesystemMiddleware(config)


def create_subagent_middleware(
    subagents: list[SubAgent | CustomSubAgent] | None = None,
    default_tools: list[Callable] | None = None,
    config: MiddlewareConfig | None = None,
) -> SubAgentMiddleware:
    """Create a subagent middleware instance."""
    return SubAgentMiddleware(subagents, default_tools, config)


def create_summarization_middleware(
    max_tokens_before_summary: int = 120000,
    messages_to_keep: int = 20,
    config: MiddlewareConfig | None = None,
) -> SummarizationMiddleware:
    """Create a summarization middleware instance."""
    return SummarizationMiddleware(max_tokens_before_summary, messages_to_keep, config)


def create_prompt_caching_middleware(
    ttl: str = "5m",
    unsupported_model_behavior: str = "ignore",
    config: MiddlewareConfig | None = None,
) -> PromptCachingMiddleware:
    """Create a prompt caching middleware instance."""
    return PromptCachingMiddleware(ttl, unsupported_model_behavior, config)


def create_default_middleware_pipeline(
    subagents: list[SubAgent | CustomSubAgent] | None = None,
    default_tools: list[Callable] | None = None,
) -> MiddlewarePipeline:
    """Create a default middleware pipeline with common middleware."""
    # This DeepAgent middleware system is currently experimental and intentionally
    # disabled by default. Enable explicitly to opt in.
    if os.getenv("DEEPC_ENABLE_DEEP_AGENT_MIDDLEWARE", "0") != "1":
        return MiddlewarePipeline()

    pipeline = MiddlewarePipeline()

    # Add middleware in order of priority
    pipeline.add_middleware(create_planning_middleware())
    pipeline.add_middleware(create_filesystem_middleware())
    pipeline.add_middleware(create_subagent_middleware(subagents, default_tools))
    pipeline.add_middleware(create_summarization_middleware())
    pipeline.add_middleware(create_prompt_caching_middleware())

    return pipeline


# Export all middleware components
__all__ = [
    # Base classes
    "BaseMiddleware",
    "FilesystemMiddleware",
    # Configuration and results
    "MiddlewareConfig",
    "MiddlewarePipeline",
    "MiddlewareResult",
    # Middleware implementations
    "PlanningMiddleware",
    "PromptCachingMiddleware",
    "SubAgentMiddleware",
    "SummarizationMiddleware",
    "create_default_middleware_pipeline",
    "create_filesystem_middleware",
    # Factory functions
    "create_planning_middleware",
    "create_prompt_caching_middleware",
    "create_subagent_middleware",
    "create_summarization_middleware",
]

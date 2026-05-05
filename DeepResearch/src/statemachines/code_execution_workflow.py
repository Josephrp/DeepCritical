"""
Code Execution Workflow.

This module originally used Pydantic Graph, but the surrounding project now
type-checks against a specific `pydantic_graph` API. To keep the code-execution
workflow type-safe and runnable without tightly coupling to graph internals,
we implement a lightweight explicit state-machine loop instead.
"""

import inspect
from typing import Any, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field

# Optional import for pydantic_graph
try:
    from pydantic_graph import BaseNode, End, Graph
except ImportError:
    # Create placeholder classes for when pydantic_graph is not available
    from typing import Generic, TypeVar

    T = TypeVar("T")

    class Graph:
        def __init__(self, *args, **kwargs):
            pass

        async def run(self, state: Any) -> Any:
            """Stub method for when pydantic_graph is not available."""

    class BaseNode(Generic[T]):
        def __init__(self, *args, **kwargs):
            pass

    class End:
        def __init__(self, *args, **kwargs):
            pass


from DeepResearch.src.datatypes.agent_framework_content import TextContent
from DeepResearch.src.datatypes.agent_framework_types import AgentRunResponse
from DeepResearch.src.datatypes.coding_base import CodeBlock
from DeepResearch.src.utils.execution_status import ExecutionStatus


class _Node(Protocol):
    def run(self, state: "CodeExecutionWorkflowState") -> Any: ...


class CodeExecutionWorkflowState(BaseModel):
    """State for the code execution workflow."""

    user_query: str = Field(
        ..., description="Natural language description of desired operation"
    )
    code_type: str | None = Field(
        None, description="Type of code to generate (bash/python/auto)"
    )
    force_code_type: bool = Field(
        False, description="Whether to force the specified code type"
    )

    # Generation results
    detected_code_type: str | None = Field(None, description="Auto-detected code type")
    generated_code: str | None = Field(None, description="Generated code content")
    code_block: CodeBlock | None = Field(None, description="Generated code block")

    # Execution results
    execution_success: bool = Field(False, description="Whether execution succeeded")
    execution_output: str | None = Field(None, description="Execution output")
    execution_error: str | None = Field(None, description="Execution error message")
    execution_exit_code: int = Field(0, description="Execution exit code")
    execution_executor: str | None = Field(None, description="Executor used")

    # Error analysis and improvement
    error_analysis: dict[str, Any] | None = Field(
        None, description="Error analysis results"
    )
    improvement_attempts: int = Field(
        0, description="Number of improvement attempts made"
    )
    max_improvement_attempts: int = Field(
        3, description="Maximum improvement attempts allowed"
    )
    improved_code: str | None = Field(
        None, description="Improved code after error analysis"
    )
    improvement_history: list[dict[str, Any]] = Field(
        default_factory=list, description="History of improvements"
    )

    # Configuration
    use_docker: bool = Field(True, description="Use Docker for execution")
    use_jupyter: bool = Field(False, description="Use Jupyter for execution")
    jupyter_config: dict[str, Any] = Field(
        default_factory=dict, description="Jupyter configuration"
    )
    max_retries: int = Field(3, description="Maximum execution retries")
    timeout: float = Field(60.0, description="Execution timeout")
    enable_improvement: bool = Field(
        True, description="Enable automatic code improvement on errors"
    )

    # Final response
    final_response: AgentRunResponse | None = Field(
        None, description="Final response to user"
    )

    # Status and metadata
    status: ExecutionStatus = Field(
        ExecutionStatus.PENDING, description="Workflow status"
    )
    errors: list[str] = Field(
        default_factory=list, description="Any errors encountered"
    )
    generation_time: float = Field(0.0, description="Code generation time")
    execution_time: float = Field(0.0, description="Code execution time")
    improvement_time: float = Field(0.0, description="Code improvement time")
    total_time: float = Field(0.0, description="Total processing time")

    model_config = ConfigDict(json_schema_extra={})


class InitializeCodeExecution:
    """Initialize the code execution workflow."""

    def run(self, state: CodeExecutionWorkflowState) -> Any:
        """Initialize workflow parameters and validate inputs."""
        try:
            # Validate user query
            if not state.user_query or not state.user_query.strip():
                state.errors.append("User query cannot be empty")
                state.status = ExecutionStatus.FAILED
                return None

            # Set default configuration
            if state.code_type not in [None, "bash", "python", "auto"]:
                state.errors.append(f"Invalid code type: {state.code_type}")
                state.status = ExecutionStatus.FAILED
                return None

            # Normalize code_type
            if state.code_type == "auto":
                state.code_type = None

            state.status = ExecutionStatus.RUNNING
            return GenerateCode()

        except Exception as e:
            state.errors.append(f"Initialization failed: {e!s}")
            state.status = ExecutionStatus.FAILED
            return None


class GenerateCode:
    """Generate code from natural language description."""

    async def run(self, state: CodeExecutionWorkflowState) -> Any:
        """Generate code using the CodeGenerationAgent."""
        try:
            import time

            start_time = time.time()

            # Import the generation agent
            from DeepResearch.src.agents.code_generation_agent import (
                CodeGenerationAgent,
            )

            # Initialize generation agent
            generation_agent = CodeGenerationAgent(
                max_retries=state.max_retries, timeout=state.timeout
            )

            # Generate code
            detected_type, generated_code = await generation_agent.generate_code(
                state.user_query, state.code_type
            )

            # Create code block
            code_block = generation_agent.create_code_block(
                generated_code, detected_type
            )

            # Update state
            state.detected_code_type = detected_type
            state.generated_code = generated_code
            state.code_block = code_block
            state.generation_time = time.time() - start_time

            return ExecuteCode()

        except Exception as e:
            state.errors.append(f"Code generation failed: {e!s}")
            state.status = ExecutionStatus.FAILED
            return None


class ExecuteCode:
    """Execute the generated code."""

    async def run(self, state: CodeExecutionWorkflowState) -> Any:
        """Execute code using the CodeExecutionAgent."""
        try:
            import time

            start_time = time.time()

            # Get the current code to execute (original or improved)
            current_code = state.improved_code or state.generated_code
            if not current_code:
                state.errors.append("No code to execute")
                state.status = ExecutionStatus.FAILED
                return None

            # Create code block if needed
            if not state.code_block:
                state.code_block = CodeBlock(
                    code=current_code, language=state.detected_code_type or "python"
                )

            # Import the execution agent
            from DeepResearch.src.agents.code_generation_agent import CodeExecutionAgent

            # Initialize execution agent
            execution_agent = CodeExecutionAgent(
                use_docker=state.use_docker,
                use_jupyter=state.use_jupyter,
                jupyter_config=state.jupyter_config,
                max_retries=state.max_retries,
                timeout=state.timeout,
            )

            # Execute code
            execution_result = await execution_agent.execute_code_block(
                state.code_block
            )

            # Update state
            state.execution_success = execution_result["success"]
            state.execution_output = execution_result.get("output")
            state.execution_error = execution_result.get("error")
            state.execution_exit_code = execution_result.get("exit_code", 1)
            state.execution_executor = execution_result.get("executor")
            state.execution_time = time.time() - start_time

            # Check if execution succeeded or if we should try improvement
            if state.execution_success:
                return FormatResponse()
            if (
                state.enable_improvement
                and state.improvement_attempts < state.max_improvement_attempts
            ):
                return AnalyzeError()
            return FormatResponse()

        except Exception as e:
            state.errors.append(f"Code execution failed: {e!s}")
            state.status = ExecutionStatus.FAILED
            return None


class AnalyzeError:
    """Analyze execution errors to understand what went wrong."""

    async def run(self, state: CodeExecutionWorkflowState) -> Any:
        """Analyze the execution error using the CodeImprovementAgent."""
        try:
            import time

            start_time = time.time()

            if not state.execution_error:
                # No error to analyze, should not happen but handle gracefully
                return FormatResponse()

            # Get the current code that failed
            current_code = state.improved_code or state.generated_code
            if not current_code:
                state.errors.append("No code to analyze")
                return FormatResponse()

            # Import the improvement agent
            from DeepResearch.src.agents.code_improvement_agent import (
                CodeImprovementAgent,
            )

            # Initialize improvement agent
            improvement_agent = CodeImprovementAgent()

            # Analyze the error
            error_analysis = await improvement_agent.analyze_error(
                code=current_code,
                error_message=state.execution_error,
                language=state.detected_code_type or "python",
                context={
                    "working_directory": "unknown",  # Could be enhanced with actual working directory
                    "environment": state.execution_executor or "unknown",
                    "timeout": state.timeout,
                    "attempt": state.improvement_attempts + 1,
                },
            )

            # Update state
            state.error_analysis = error_analysis
            state.improvement_time += time.time() - start_time

            return ImproveCode()

        except Exception as e:
            state.errors.append(f"Error analysis failed: {e!s}")
            # Continue to improvement anyway
            return ImproveCode()


class ImproveCode:
    """Improve the code based on error analysis."""

    async def run(self, state: CodeExecutionWorkflowState) -> Any:
        """Improve the code using the CodeImprovementAgent."""
        try:
            import time

            start_time = time.time()

            # Get the current code to improve
            current_code = state.improved_code or state.generated_code
            if not current_code:
                state.errors.append("No code to improve")
                return FormatResponse()

            error_message = state.execution_error or "Unknown error"

            # Import the improvement agent
            from DeepResearch.src.agents.code_improvement_agent import (
                CodeImprovementAgent,
            )

            # Initialize improvement agent
            improvement_agent = CodeImprovementAgent()

            # Improve the code
            improvement_result = await improvement_agent.improve_code(
                original_code=current_code,
                error_message=error_message,
                language=state.detected_code_type or "python",
                context={
                    "working_directory": "unknown",
                    "environment": state.execution_executor or "unknown",
                    "timeout": state.timeout,
                    "attempt": state.improvement_attempts + 1,
                },
                improvement_focus="fix_errors",
            )

            # Update state
            state.improvement_attempts += 1
            state.improved_code = improvement_result["improved_code"]

            # Record improvement history
            state.improvement_history.append(
                {
                    "attempt": state.improvement_attempts,
                    "original_code": improvement_result["original_code"],
                    "error_message": error_message,
                    "improved_code": improvement_result["improved_code"],
                    "explanation": improvement_result["explanation"],
                    "analysis": state.error_analysis,
                }
            )

            # Update the code block with improved code
            state.code_block = improvement_agent.create_improved_code_block(
                improvement_result
            )

            state.improvement_time += time.time() - start_time

            # Execute the improved code
            return ExecuteCode()

        except Exception as e:
            state.errors.append(f"Code improvement failed: {e!s}")
            # Continue to formatting even if improvement fails
            return FormatResponse()


class FormatResponse:
    """Format the final response to the user."""

    def run(self, state: CodeExecutionWorkflowState) -> Any:
        """Format the execution results into a user-friendly response."""
        try:
            import time

            from DeepResearch.src.datatypes.agent_framework_types import (
                ChatMessage,
                Role,
            )

            # Calculate total time
            state.total_time = (
                state.generation_time + state.execution_time + state.improvement_time
            )

            # Create response messages
            messages = []

            # Code generation message
            code_type_display = (
                state.detected_code_type.upper()
                if state.detected_code_type
                else "UNKNOWN"
            )
            final_code = state.improved_code or state.generated_code
            code_content = f"**Generated {code_type_display} Code:**\n\n```{state.detected_code_type}\n{final_code}\n```"
            messages.append(
                ChatMessage(
                    role=Role.ASSISTANT, contents=[TextContent(text=code_content)]
                )
            )

            # Execution result message
            if state.execution_success:
                execution_content = f"**✅ Execution Successful**\n\n**Output:**\n```\n{state.execution_output or 'No output'}\n```"
                if state.execution_executor:
                    execution_content += (
                        f"\n\n**Executed using:** {state.execution_executor}"
                    )

                # Add improvement information if applicable
                if state.improvement_attempts > 0:
                    execution_content += f"\n\n**Improvements Made:** {state.improvement_attempts} iteration(s)"

            else:
                execution_content = f"**❌ Execution Failed**\n\n**Error:**\n```\n{state.execution_error or 'Unknown error'}\n```"
                execution_content += f"\n\n**Exit Code:** {state.execution_exit_code}"

                # Add improvement information
                if state.improvement_attempts > 0:
                    execution_content += (
                        f"\n\n**Improvement Attempts:** {state.improvement_attempts}"
                    )
                    if state.error_analysis:
                        execution_content += f"\n**Error Type:** {state.error_analysis.get('error_type', 'unknown')}"
                        execution_content += f"\n**Root Cause:** {state.error_analysis.get('root_cause', 'unknown')}"

            # Add timing information
            execution_content += (
                ".2f"
                ".2f"
                ".2f"
                ".2f"
                f"""
\n\n**Performance:**
- Generation: {state.generation_time:.2f}s
- Execution: {state.execution_time:.2f}s
- Improvement: {state.improvement_time:.2f}s
- Total: {state.total_time:.2f}s
"""
            )

            messages.append(
                ChatMessage(
                    role=Role.ASSISTANT, contents=[TextContent(text=execution_content)]
                )
            )

            # Add improvement history if applicable
            if state.improvement_history and len(state.improvement_history) > 0:
                history_content = "**Improvement History:**\n\n"
                for i, improvement in enumerate(state.improvement_history, 1):
                    history_content += f"**Attempt {i}:**\n"
                    history_content += f"- **Error:** {improvement['error_message'][:100]}{'...' if len(improvement['error_message']) > 100 else ''}\n"
                    history_content += f"- **Fix:** {improvement['explanation'][:150]}{'...' if len(improvement['explanation']) > 150 else ''}\n\n"

                messages.append(
                    ChatMessage(
                        role=Role.ASSISTANT,
                        contents=[TextContent(text=history_content)],
                    )
                )

            # Create final response
            state.final_response = AgentRunResponse(messages=messages)
            state.status = ExecutionStatus.SUCCESS

            return None

        except Exception as e:
            state.errors.append(f"Response formatting failed: {e!s}")
            state.status = ExecutionStatus.FAILED
            return None


class CodeExecutionWorkflow:
    """Complete code execution workflow as explicit state machine."""

    def __init__(self):
        """Initialize the code execution workflow."""
        self._start_node: _Node = InitializeCodeExecution()

    async def execute(
        self,
        user_query: str,
        code_type: str | None = None,
        use_docker: bool = True,
        use_jupyter: bool = False,
        jupyter_config: dict[str, Any] | None = None,
        max_retries: int = 3,
        timeout: float = 60.0,
        enable_improvement: bool = True,
        max_improvement_attempts: int = 3,
    ) -> CodeExecutionWorkflowState:
        """Execute the complete code generation and execution workflow.

        Args:
            user_query: Natural language description of desired operation
            code_type: Type of code to generate ("bash", "python", or None for auto-detection)
            use_docker: Whether to use Docker for execution
            use_jupyter: Whether to use Jupyter for execution
            jupyter_config: Configuration for Jupyter execution
            max_retries: Maximum number of execution retries
            timeout: Execution timeout in seconds
            enable_improvement: Whether to enable automatic code improvement on errors
            max_improvement_attempts: Maximum number of improvement attempts

        Returns:
            Final workflow state with results
        """
        # Initialize state
        initial_state = CodeExecutionWorkflowState(
            user_query=user_query,
            code_type=code_type,
            use_docker=use_docker,
            use_jupyter=use_jupyter,
            jupyter_config=jupyter_config or {},
            max_retries=max_retries,
            timeout=timeout,
            enable_improvement=enable_improvement,
            max_improvement_attempts=max_improvement_attempts,
        )

        # Execute explicit state-machine loop (nodes return the next step).
        state = initial_state
        current: Any = self._start_node
        while current is not None:
            step = getattr(current, "run", None)
            if step is None:
                break
            step_result = step(state)
            if inspect.isawaitable(step_result):
                step_result = await step_result
            if step_result is None:
                break
            current = step_result

        return state


# Convenience functions for direct usage
async def execute_code_workflow(
    user_query: str, code_type: str | None = None, **kwargs
) -> AgentRunResponse | None:
    """Execute a code generation and execution workflow.

    Args:
        user_query: Natural language description of desired operation
        code_type: Type of code to generate ("bash", "python", or None for auto-detection)
        **kwargs: Additional configuration options

    Returns:
        AgentRunResponse with execution results, or None if failed
    """
    workflow = CodeExecutionWorkflow()
    result = await workflow.execute(user_query, code_type, **kwargs)
    return result.final_response


async def generate_and_execute_code(
    description: str,
    code_type: str | None = None,
    use_docker: bool = True,
) -> dict[str, Any]:
    """Generate and execute code from a natural language description.

    Args:
        description: Natural language description of desired operation
        code_type: Type of code to generate ("bash", "python", or None for auto-detection)
        use_docker: Whether to use Docker for execution

    Returns:
        Dictionary with complete execution results
    """
    workflow = CodeExecutionWorkflow()
    state = await workflow.execute(
        user_query=description, code_type=code_type, use_docker=use_docker
    )

    return {
        "success": state.status == ExecutionStatus.SUCCESS and state.execution_success,
        "generated_code": state.generated_code,
        "code_type": state.detected_code_type,
        "execution_output": state.execution_output,
        "execution_error": state.execution_error,
        "execution_time": state.execution_time,
        "total_time": state.total_time,
        "executor": state.execution_executor,
        "response": state.final_response,
    }

"""
Tests for specialized agent implementations.

This module tests all specialized agents including ParserAgent, PlannerAgent,
ExecutorAgent, SearchAgent, RAGAgent, BioinformaticsAgent, DeepSearchAgent,
and EvaluatorAgent.
"""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from DeepResearch.agents import (
    BaseAgent,
    BioinformaticsAgent,
    DeepAgentFilesystemAgent,
    DeepAgentGeneralAgent,
    DeepAgentOrchestrationAgent,
    DeepAgentPlanningAgent,
    DeepAgentResearchAgent,
    DeepSearchAgent,
    EvaluatorAgent,
    ExecutorAgent,
    ParserAgent,
    PlannerAgent,
    RAGAgent,
    SearchAgent,
    create_agent,
)
from DeepResearch.src.agents.deep_agent_implementations import (
    AgentExecutionResult,
    PlanningAgent,
)
from DeepResearch.src.datatypes.agents import (
    AgentDependencies,
    AgentResult,
    AgentType,
    ExecutionHistory,
)
from DeepResearch.src.datatypes.bioinformatics import (
    DataFusionRequest,
    FusedDataset,
    ReasoningTask,
)
from DeepResearch.src.datatypes.llm_models import DEFAULT_PYDANTIC_AI_MODEL
from DeepResearch.src.datatypes.rag import (
    Document,
    RAGQuery,
    RAGResponse,
    SearchResult,
    SearchType,
)
from DeepResearch.src.utils.model_registry import resolve_pydantic_ai_model


def _patch_pydantic_agent_class(mock_agent_instance: MagicMock):
    """Patch Agent so ``Agent[Deps, Result](...)`` resolves (pydantic-ai generic Agent API)."""

    constructor = MagicMock(return_value=mock_agent_instance)
    agent_cls = MagicMock()
    agent_cls.__getitem__ = MagicMock(return_value=constructor)
    return patch("DeepResearch.agents.Agent", agent_cls)


# Fixtures
@pytest.fixture
def mock_pydantic_agent():
    """Mock Pydantic AI Agent."""
    agent = Mock()
    agent.run = AsyncMock()
    agent.tool = Mock()
    return agent


@pytest.fixture
def execution_history():
    """Create execution history."""
    return ExecutionHistory()


def assert_valid_plan(plan, query_value):
    """Reusable validator for plan structure and params."""
    # Check overall structure
    assert isinstance(plan, list)
    assert len(plan) == 6
    assert all(isinstance(step, dict) for step in plan)
    assert all("tool" in step and "params" in step for step in plan)

    # Check parameter details for each tool
    expected_tools = [
        "rewrite",
        "web_search",
        "summarize",
        "references",
        "finalize",
        "evaluator",
    ]
    assert [step["tool"] for step in plan] == expected_tools, "Tool sequence mismatch"

    # Check specific sequence and keys
    expected_params = [
        {"query": query_value},
        {"query": "${rewrite.queries}"},
        {"snippets": "${web_search.results}"},
        {"answer": "${summarize.summary}", "web": "${web_search.results}"},
        {"draft": "${references.answer_with_refs}"},
        {"question": query_value, "answer": "${finalize.final}"},
    ]

    for step, expected in zip(plan, expected_params, strict=False):
        assert step["params"] == expected, f"Params mismatch for {step['tool']}"


# PlannerAgent Tests
class TestPlannerAgent:
    """Test PlannerAgent functionality."""

    def test_planner_agent_initialization(self):
        """Test PlannerAgent initialization."""
        with patch("DeepResearch.agents.Agent"):
            agent = PlannerAgent()

            assert agent.agent_type == AgentType.PLANNER
            assert agent.model_name == DEFAULT_PYDANTIC_AI_MODEL

    def test_planner_custom_model(self):
        """Test PlannerAgent with custom model."""
        with patch("DeepResearch.agents.Agent"):
            agent = PlannerAgent(model_name="openai:gpt-4")

            assert agent.model_name == "openai:gpt-4"

    @pytest.mark.asyncio
    async def test_create_plan_success(self):
        """Test successful plan creation."""
        agent = PlannerAgent()

        mock_result = Mock(spec=AgentResult)
        mock_result.success = True
        mock_result.data = {
            "steps": [
                {"tool": "rewrite", "params": {"query": "test"}},
                {"tool": "search", "params": {"query": "${rewrite.queries}"}},
            ]
        }

        with patch.object(
            BaseAgent, "execute", new=AsyncMock(return_value=mock_result)
        ):
            parsed = {"intent": "research", "query": "test question"}
            plan = await agent.create_plan(parsed)

        # Basic type and length checks
        assert isinstance(plan, list), f"Expected list, got {type(plan)}"
        assert len(plan) == 2, f"Expected 2 steps, got {len(plan)}"
        assert all(isinstance(step, dict) for step in plan), "Each step must be a dict"
        assert all("tool" in step and "params" in step for step in plan), (
            "Each step must contain 'tool' and 'params' keys"
        )

        # Expected tool sequence
        expected_tools = ["rewrite", "search"]
        assert [step["tool"] for step in plan] == expected_tools, (
            f"Expected tools {expected_tools}, got {[step['tool'] for step in plan]}"
        )

        # Detailed parameter validation
        assert plan[0]["params"] == {"query": "test"}, (
            f"Unexpected params for 'rewrite': {plan[0]['params']}"
        )
        assert plan[1]["params"] == {"query": "${rewrite.queries}"}, (
            f"Unexpected params for 'search': {plan[1]['params']}"
        )

    @pytest.mark.asyncio
    async def test_create_plan_fallback(self):
        """Test plan creation falls back to default."""
        agent = PlannerAgent()

        mock_result = Mock(spec=AgentResult)
        mock_result.success = True
        mock_result.data = {}  # No steps

        with patch.object(
            BaseAgent, "execute", new=AsyncMock(return_value=mock_result)
        ):
            parsed = {"query": "test question"}
            plan = await agent.create_plan(parsed)
            # Detailed parameter validation
            assert_valid_plan(plan, "test question")

    def test_get_default_plan(self):
        """Test default plan generation."""
        agent = PlannerAgent()
        plan = agent._get_default_plan("test query")
        # Detailed parameter validation
        assert_valid_plan(plan, "test query")

    def test_plan_sync_method(self):
        """Test synchronous plan method."""
        agent = PlannerAgent()

        mock_result = Mock(spec=AgentResult)
        mock_result.success = True
        mock_result.data = {"steps": [{"tool": "test"}]}

        with patch.object(BaseAgent, "execute_sync", return_value=mock_result):
            parsed = {"query": "test"}
            plan = agent.plan(parsed)
        assert len(plan) == 1
        assert plan[0]["tool"] == "test"

    def test_plan_sync_fallback_on_exception(self):
        """Test synchronous plan method falls back when execute_sync raises an exception."""
        agent = PlannerAgent()

        with patch.object(
            BaseAgent, "execute_sync", side_effect=RuntimeError("Model timeout")
        ):
            parsed = {"query": "test fallback"}
            plan = agent.plan(parsed)
        # Detailed parameter validation
        assert_valid_plan(plan, "test fallback")


# ExecutorAgent Tests
class TestExecutorAgent:
    """Test ExecutorAgent functionality."""

    def test_executor_initialization_default(self):
        """Test ExecutorAgent initialization with defaults."""
        with patch("DeepResearch.agents.Agent"):
            agent = ExecutorAgent()

            assert agent.agent_type == AgentType.EXECUTOR
            assert agent.retries == 2

    def test_executor_initialization_custom_retries(self):
        """Test ExecutorAgent with custom retries."""
        with patch("DeepResearch.agents.Agent"):
            agent = ExecutorAgent(retries=5)

            assert agent.retries == 5

    @pytest.mark.asyncio
    async def test_execute_plan_success(self, execution_history):
        """Test successful plan execution."""
        with patch("DeepResearch.agents.Agent"):
            with patch("DeepResearch.agents.registry") as mock_registry:
                agent = ExecutorAgent()

                # Mock tool runner
                mock_runner = Mock()
                mock_runner.run.return_value = Mock(
                    success=True, data={"result": "test_output"}
                )
                mock_registry.make.return_value = mock_runner

                plan = [{"tool": "test_tool", "params": {"query": "test"}}]

                result = await agent.execute_plan(plan, execution_history)

                assert "test_tool.result" in result
                assert result["test_tool.result"] == "test_output"

    @pytest.mark.asyncio
    async def test_execute_plan_with_placeholders(self, execution_history):
        """Test parameter materialization with placeholders across multiple steps."""
        with patch("DeepResearch.agents.Agent"):
            with patch("DeepResearch.agents.registry") as mock_registry:
                agent = ExecutorAgent()

                # Mock two tool runners: search and summarize
                mock_search_runner = Mock()
                mock_search_runner.run.return_value = Mock(
                    success=True, data={"results": "retrieved_data"}
                )

                mock_summarize_runner = Mock()
                mock_summarize_runner.run.return_value = Mock(
                    success=True, data={"summary": "final_summary"}
                )

                # Configure registry to return correct runner per tool name
                def make_tool(tool_name):
                    if tool_name == "search":
                        return mock_search_runner
                    if tool_name == "summarize":
                        return mock_summarize_runner
                    raise ValueError(f"Unexpected tool: {tool_name}")

                mock_registry.make.side_effect = make_tool

                # Plan with placeholder: summarize uses output from search
                plan = [
                    {"tool": "search", "params": {"query": "test query"}},
                    {"tool": "summarize", "params": {"content": "${search.results}"}},
                ]

                result = await agent.execute_plan(plan, execution_history)

                # Verify search was called with static param
                mock_search_runner.run.assert_called_once_with({"query": "test query"})

                # Verify summarize was called with MATERIALIZED param (not placeholder)
                mock_summarize_runner.run.assert_called_once_with(
                    {"content": "retrieved_data"}
                )

                # Verify bag contains scoped and flat keys
                assert "search.results" in result
                assert result["search.results"] == "retrieved_data"
                assert "results" in result  # flat alias
                assert result["results"] == "retrieved_data"

                assert "summarize.summary" in result
                assert result["summarize.summary"] == "final_summary"
                assert "summary" in result  # flat alias
                assert result["summary"] == "final_summary"

    @pytest.mark.asyncio
    async def test_execute_plan_retries_on_failure(self, execution_history):
        """Test that execute_plan retries a tool if it fails on first attempt."""
        with patch("DeepResearch.agents.Agent"):
            with patch("DeepResearch.agents.registry") as mock_registry:
                agent = ExecutorAgent(retries=1)  # Allow 1 retry (2 total attempts max)

                # Mock a tool that fails on first call, succeeds on second
                mock_runner = Mock()
                mock_runner.run.side_effect = [
                    Mock(success=False, error="Network timeout"),  # First attempt: fail
                    Mock(
                        success=True, data={"result": "retried_success"}
                    ),  # Second: succeed
                ]
                mock_registry.make.return_value = mock_runner

                plan = [{"tool": "flaky_tool", "params": {"query": "retry_test"}}]

                result = await agent.execute_plan(plan, execution_history)

                # Verify the tool was called twice (due to retry)
                assert mock_runner.run.call_count == 2

                # Verify final bag contains the successful result
                assert "flaky_tool.result" in result
                assert result["flaky_tool.result"] == "retried_success"

                # Verify both attempts were recorded in history
                assert len(execution_history.items) == 2

                # Inspect the recorded attempts
                first_attempt = execution_history.items[0]
                second_attempt = execution_history.items[1]

                assert first_attempt["success"] is False
                assert first_attempt["error"] == "Network timeout"
                assert first_attempt["tool"] == "flaky_tool"
                assert first_attempt["params"] == {"query": "retry_test"}

                assert second_attempt["success"] is True
                assert second_attempt["tool"] == "flaky_tool"
                assert second_attempt["params"] == {"query": "retry_test"}

    @pytest.mark.asyncio
    async def test_execute_plan_fails_after_retries(self, execution_history):
        """Test that execute_plan stops execution if a tool fails after all retries."""
        with patch("DeepResearch.agents.Agent"):
            with patch("DeepResearch.agents.registry") as mock_registry:
                agent = ExecutorAgent(
                    retries=2
                )  # Allow 2 retries (3 total attempts max)

                # Mock runners
                mock_first_runner = Mock()
                mock_first_runner.run.return_value = Mock(
                    success=True, data={"preliminary": "data_from_first"}
                )

                mock_failing_runner = Mock()
                # Fail on all 3 attempts (1 initial + 2 retries)
                mock_failing_runner.run.return_value = Mock(
                    success=False, error="Always fails"
                )

                mock_final_runner = Mock()  # Should NOT be called
                mock_final_runner.run.return_value = Mock(
                    success=True, data={"final": "should_not_happen"}
                )

                def make_tool(tool_name):
                    if tool_name == "first_tool":
                        return mock_first_runner
                    if tool_name == "failing_tool":
                        return mock_failing_runner
                    if tool_name == "final_tool":
                        return mock_final_runner
                    raise ValueError(f"Unexpected tool: {tool_name}")

                mock_registry.make.side_effect = make_tool

                # Plan: first_tool → failing_tool → final_tool
                plan = [
                    {"tool": "first_tool", "params": {}},
                    {"tool": "failing_tool", "params": {}},
                    {"tool": "final_tool", "params": {}},  # Should not execute
                ]

                result = await agent.execute_plan(plan, execution_history)

                # Verify first tool was called once
                mock_first_runner.run.assert_called_once_with({})

                # Verify failing tool was called 3 times (1 initial + 2 retries)
                assert mock_failing_runner.run.call_count == 3

                # Verify final tool was NEVER called
                mock_final_runner.run.assert_not_called()

                # Verify bag contains data from first step only (partial result)
                assert "first_tool.preliminary" in result
                assert result["first_tool.preliminary"] == "data_from_first"
                # Ensure final_tool data is NOT in result
                assert "final_tool.final" not in result

    def test_run_plan_sync_wrapper(self, execution_history):
        """Test that run_plan correctly wraps execute_plan."""
        with patch("DeepResearch.agents.Agent"):
            with patch("DeepResearch.agents.registry") as mock_registry:
                agent = ExecutorAgent()

                # Mock a tool that returns known data
                mock_runner = Mock()
                mock_runner.run.return_value = Mock(
                    success=True, data={"sync_result": "from_sync_wrapper"}
                )
                mock_registry.make.return_value = mock_runner

                plan = [{"tool": "sync_test_tool", "params": {"query": "sync_test"}}]

                # Call the synchronous method
                result = agent.run_plan(plan, execution_history)

                # Verify the tool was called (via execute_plan)
                mock_runner.run.assert_called_once_with({"query": "sync_test"})

                # Verify the result is returned correctly
                assert "sync_test_tool.sync_result" in result
                assert result["sync_test_tool.sync_result"] == "from_sync_wrapper"

                # Ensure it behaves the same as execute_plan
                # (We could optionally call execute_plan directly and compare results)

    def test_adaptive_parameter_adjustment(self):
        """Test that _adjust_parameters provides fallback values for empty parameters."""
        with patch("DeepResearch.agents.Agent"):
            agent = ExecutorAgent()

            # Simulate an initial parameter set with empty/missing values
            initial_params = {
                "query": "",  # Empty query should get fallback
                "snippets": "",  # Empty snippets should get fallback from bag
                "other_param": "already_set",  # Should remain unchanged
            }

            # Simulate a bag with fallback data for snippets
            bag = {"search.snippets": "fallback_snippets_from_bag"}

            # Call the internal adjustment method
            adjusted = agent._adjust_parameters(initial_params, bag)

            # Verify adjustments were made
            assert adjusted["query"] == "general information", (
                "Empty query should default to 'general information'"
            )

            assert adjusted["snippets"] == "fallback_snippets_from_bag", (
                "Empty snippets should default to value from bag['search.snippets']"
            )

            # Verify unchanged params
            assert adjusted["other_param"] == "already_set", (
                "Non-empty params should remain unchanged"
            )

            # Ensure original dict was not mutated
            assert initial_params["query"] == "", (
                "Original params should not be modified"
            )

    # Just to make sure that _adjust_parameters is actually
    # called during execution, I am patching it:
    @pytest.mark.asyncio
    async def test_adjust_parameters_called_on_retry(self, execution_history):
        """Test that _adjust_parameters is invoked during retries."""
        with patch("DeepResearch.agents.Agent"):
            with patch("DeepResearch.agents.registry") as mock_registry:
                agent = ExecutorAgent(retries=1)

                # Mock a tool that fails once, then succeeds
                mock_runner = Mock()
                mock_runner.run.side_effect = [
                    Mock(success=False, error="Bad query"),
                    Mock(success=True, data={"result": "adjusted_success"}),
                ]
                mock_registry.make.return_value = mock_runner

                # Patch the adjustment method to verify it's called
                with patch.object(agent, "_adjust_parameters") as mock_adjust:
                    mock_adjust.return_value = {"query": "adjusted_query"}

                    plan = [{"tool": "adjust_test", "params": {"query": "bad_query"}}]
                    _ = await agent.execute_plan(plan, execution_history)

                    # Verify adjustment was called once
                    mock_adjust.assert_called_once()

                    # Check that the first call was with the expected params and an (initially empty) bag
                    # The bag might contain data from previous attempts or initial state
                    args, _ = mock_adjust.call_args
                    called_params, _ = args
                    assert called_params == {"query": "bad_query"}
                    # Bag might be non-empty due to execution context — just verify params part
                    assert "query" in called_params

    def test_materialize_params(self):
        """Test parameter materialization."""
        with patch("DeepResearch.agents.Agent"):
            agent = ExecutorAgent()

            bag = {"rewrite.queries": ["q1", "q2"], "search.results": ["r1"]}
            params = {"query": "${rewrite.queries}", "type": "web"}

            result = agent._materialize_params(params, bag)

            assert result["query"] == ["q1", "q2"]
            assert result["type"] == "web"

    def test_materialize_params_missing_key(self):
        """Test parameter materialization with missing key."""
        with patch("DeepResearch.agents.Agent"):
            agent = ExecutorAgent()

            bag = {}
            params = {"query": "${missing.key}"}

            result = agent._materialize_params(params, bag)

            assert result["query"] == ""

    def test_adjust_parameters(self):
        """Test parameter adjustment for retries."""
        with patch("DeepResearch.agents.Agent"):
            agent = ExecutorAgent()

            bag = {"search.snippets": "test data"}
            params = {"query": "", "snippets": ""}

            adjusted = agent._adjust_parameters(params, bag)

            assert adjusted["query"] == "general information"
            assert adjusted["snippets"] == "test data"

    @pytest.mark.asyncio
    async def test_bag_flat_aliasing(self, execution_history):
        """Test that execute_plan populates bag with both scoped and flat keys."""
        with patch("DeepResearch.agents.Agent"):
            with patch("DeepResearch.agents.registry") as mock_registry:
                agent = ExecutorAgent()

                # Mock a tool that returns multiple data keys
                mock_runner = Mock()
                mock_runner.run.return_value = Mock(
                    success=True,
                    data={
                        "summary": "brief_summary",
                        "confidence": 0.95,
                        "metadata": {"source": "web"},
                    },
                )
                mock_registry.make.return_value = mock_runner

                plan = [{"tool": "test_alias_tool", "params": {"query": "alias_test"}}]

                result = await agent.execute_plan(plan, execution_history)

                # Verify scoped keys exist
                assert "test_alias_tool.summary" in result
                assert result["test_alias_tool.summary"] == "brief_summary"

                assert "test_alias_tool.confidence" in result
                assert result["test_alias_tool.confidence"] == 0.95

                assert "test_alias_tool.metadata" in result
                assert result["test_alias_tool.metadata"] == {"source": "web"}

                # Verify flat aliases also exist
                assert "summary" in result
                assert result["summary"] == "brief_summary"

                assert "confidence" in result
                assert result["confidence"] == 0.95

                assert "metadata" in result
                assert result["metadata"] == {"source": "web"}

    @pytest.mark.asyncio
    async def test_bag_flat_alias_collision(self, execution_history):
        """Test that flat aliases can overwrite each other, but scoped keys remain distinct."""
        with patch("DeepResearch.agents.Agent"):
            with patch("DeepResearch.agents.registry") as mock_registry:
                agent = ExecutorAgent()

                mock_first_runner = Mock()
                mock_first_runner.run.return_value = Mock(
                    success=True, data={"common_key": "from_first"}
                )

                mock_second_runner = Mock()
                mock_second_runner.run.return_value = Mock(
                    success=True, data={"common_key": "from_second"}
                )

                def make_tool(tool_name):
                    if tool_name == "first_tool":
                        return mock_first_runner
                    if tool_name == "second_tool":
                        return mock_second_runner
                    raise ValueError(f"Unexpected tool: {tool_name}")

                mock_registry.make.side_effect = make_tool

                plan = [
                    {"tool": "first_tool", "params": {}},
                    {"tool": "second_tool", "params": {}},
                ]

                result = await agent.execute_plan(plan, execution_history)

                # Scoped keys remain distinct
                assert result["first_tool.common_key"] == "from_first"
                assert result["second_tool.common_key"] == "from_second"

                # Flat alias overwrites previous value
                assert result["common_key"] == "from_second"  # from second tool

    @pytest.mark.asyncio
    async def test_execute_plan_halts_on_tool_failure(self, execution_history):
        """Test that execute_plan stops execution when a tool fails after all retries."""
        with patch("DeepResearch.agents.Agent"):
            with patch("DeepResearch.agents.registry") as mock_registry:
                agent = ExecutorAgent(retries=0)  # No retries — fail immediately

                # Mock runners:
                # 1. first_tool: succeeds → adds data to bag
                mock_first_runner = Mock()
                mock_first_runner.run.return_value = Mock(
                    success=True, data={"first_result": "data_from_first"}
                )

                # 2. failing_tool: fails after retries → should halt execution
                mock_failing_runner = Mock()
                mock_failing_runner.run.return_value = Mock(
                    success=False, error="Critical failure"
                )

                # 3. subsequent_tool: should NOT be called due to halt
                mock_subsequent_runner = Mock()
                mock_subsequent_runner.run.return_value = Mock(
                    success=True, data={"never_reached": "should_not_happen"}
                )

                def make_tool(tool_name):
                    if tool_name == "first_tool":
                        return mock_first_runner
                    if tool_name == "failing_tool":
                        return mock_failing_runner
                    if tool_name == "subsequent_tool":
                        return mock_subsequent_runner
                    raise ValueError(f"Unexpected tool: {tool_name}")

                mock_registry.make.side_effect = make_tool

                # Plan: first_tool → failing_tool → subsequent_tool
                plan = [
                    {"tool": "first_tool", "params": {"query": "step1"}},
                    {"tool": "failing_tool", "params": {"query": "step2"}},
                    {
                        "tool": "subsequent_tool",
                        "params": {"query": "step3"},
                    },  # Should not execute
                ]

                result = await agent.execute_plan(plan, execution_history)

                # Verify first_tool was called
                mock_first_runner.run.assert_called_once_with({"query": "step1"})

                # Verify failing_tool was called (and failed)
                mock_failing_runner.run.assert_called_once_with({"query": "step2"})

                # Verify subsequent_tool was NEVER called (execution halted)
                mock_subsequent_runner.run.assert_not_called()

                # Verify bag contains data from first_tool only (partial result)
                assert "first_tool.first_result" in result
                assert result["first_tool.first_result"] == "data_from_first"
                assert "first_result" in result  # flat alias
                assert result["first_result"] == "data_from_first"

                # Verify bag does NOT contain data from subsequent_tool
                assert "subsequent_tool.never_reached" not in result
                assert "never_reached" not in result

                # Verify history contains entries for first_tool (success) and failing_tool (failure)
                assert len(execution_history.items) == 2
                assert execution_history.items[0]["tool"] == "first_tool"
                assert execution_history.items[0]["success"] is True
                assert execution_history.items[1]["tool"] == "failing_tool"
                assert execution_history.items[1]["success"] is False


# SearchAgent Tests
class TestSearchAgent:
    """Test SearchAgent functionality."""

    def test_search_agent_initialization(self):
        """Test SearchAgent initialization."""
        with patch("DeepResearch.agents.Agent"):
            agent = SearchAgent()

            assert agent.agent_type == AgentType.SEARCH
            assert agent.model_name == resolve_pydantic_ai_model(None, "search")

    def test_search_agent_custom_model(self):
        """Test SearchAgent with custom model."""
        with patch("DeepResearch.agents.Agent"):  # Patch BaseAgent.__init__
            agent = SearchAgent(model_name="openai:gpt-4")

            assert agent.agent_type == AgentType.SEARCH
            assert agent.model_name == "openai:gpt-4"

    @pytest.mark.asyncio
    async def test_search_method_success(self):
        """Test successful search execution."""
        with patch("DeepResearch.agents.Agent"):
            agent = SearchAgent()

            # Mock the execute method to return a successful result
            mock_result = Mock(spec=AgentResult)
            mock_result.success = True
            mock_result.data = {
                "results": [
                    {
                        "title": "Test Result",
                        "url": "http://example.com",
                        "snippet": "Test snippet",
                    }
                ]
            }

            # Create the async mock and patch it
            mock_execute = AsyncMock(return_value=mock_result)

            with patch.object(BaseAgent, "execute", mock_execute):
                result = await agent.search(
                    "test query", search_type="search", num_results=5
                )

            # Verify the execute method was called with correct parameters
            mock_execute.assert_called_once_with(
                {"query": "test query", "search_type": "search", "num_results": 5}
            )

            # Verify the result is returned correctly
            assert result == mock_result.data

    @pytest.mark.asyncio
    async def test_search_method_default_parameters(self):
        """Test search uses default parameters when not specified."""
        with patch("DeepResearch.agents.Agent"):
            agent = SearchAgent()

            mock_result = Mock(spec=AgentResult)
            mock_result.success = True
            mock_result.data = {"results": []}
            # Create the async mock and patch it
            mock_execute = AsyncMock(return_value=mock_result)

            with patch.object(BaseAgent, "execute", mock_execute):
                result = await agent.search("test query")

            # Verify execute was called with default values
            mock_execute.assert_called_once_with(
                {
                    "query": "test query",
                    "search_type": "search",  # Default
                    "num_results": 10,  # Default
                }
            )

            assert result == {"results": []}

    @pytest.mark.asyncio
    async def test_search_method_failure(self):
        """Test search returns error when execution fails."""
        with patch("DeepResearch.agents.Agent"):
            agent = SearchAgent()

            # Mock the execute method to return a failed result
            mock_result = Mock(spec=AgentResult)
            mock_result.success = False
            mock_result.error = "Search failed due to network timeout"

            # Create the async mock and patch it
            mock_execute = AsyncMock(return_value=mock_result)

            with patch.object(BaseAgent, "execute", mock_execute):
                result = await agent.search("test query")

            # Verify the execute method was called with correct default parameters
            mock_execute.assert_called_once_with(
                {
                    "query": "test query",
                    "search_type": "search",  # Default value
                    "num_results": 10,  # Default value
                }
            )

            # Verify the result contains the error (as per search() implementation)
            assert result == {"error": "Search failed due to network timeout"}

    def test_tool_registration_success(self):
        """SearchAgent registers web search tools on the pydantic-ai Agent."""
        # Mock the Pydantic AI Agent instance
        mock_agent_instance = Mock()

        with patch(
            "DeepResearch.src.tools.websearch_tools.WebSearchTool"
        ) as mock_web_search_tool_cls:
            with patch(
                "DeepResearch.src.tools.websearch_tools.ChunkedSearchTool"
            ) as mock_chunked_search_tool_cls:
                # Create a SearchAgent instance with a mocked _agent
                with patch("DeepResearch.agents.Agent"):
                    agent = SearchAgent()

                agent._agent = mock_agent_instance

                agent._register_tools()

                # __init__ registers tools on the patched Agent(); we then replace _agent
                # and register again on mock_agent_instance.
                assert mock_web_search_tool_cls.call_count == 2
                assert mock_chunked_search_tool_cls.call_count == 2
                assert mock_agent_instance.tool.call_count == 2

    def test_tool_registration_failure_graceful(self):
        """Test that SearchAgent handles tool import failures gracefully."""
        with patch("DeepResearch.agents.Agent"):
            # Mock the _agent's tool method to track if it gets called
            with patch(
                "DeepResearch.agents.Agent._agent", create=True
            ) as mock_agent_instance:
                # Configure the mock agent to have a tool method that we can track
                mock_tool_method = Mock()
                mock_agent_instance.tool = mock_tool_method

                # Patch the tool imports to fail
                with patch(
                    "DeepResearch.src.tools.websearch_tools.WebSearchTool",
                    side_effect=ImportError("WebSearchTool not available"),
                ):
                    with patch(
                        "DeepResearch.src.tools.websearch_tools.ChunkedSearchTool",
                        side_effect=ImportError("ChunkedSearchTool not available"),
                    ):
                        # This should not raise an exception - the import failure should be caught silently
                        agent = SearchAgent()

                        # Verify the agent was created successfully despite the import failure
                        assert agent.agent_type == AgentType.SEARCH
                        assert agent.model_name == resolve_pydantic_ai_model(
                            None, "search"
                        )

                        # Verify that the tool method was never called (since imports failed)
                        mock_tool_method.assert_not_called()


# RAGAgent Tests
class TestRAGAgent:
    """Test RAGAgent functionality."""

    def test_rag_agent_initialization(self):
        """Test RAGAgent initialization."""
        with patch("DeepResearch.agents.Agent"):  # Patch BaseAgent.__init__
            agent = RAGAgent()

            assert agent.agent_type == AgentType.RAG
            assert agent.model_name == DEFAULT_PYDANTIC_AI_MODEL
            assert agent._agent is not None

    def test_rag_agent_custom_model(self):
        """Test RAGAgent with custom model."""
        with patch("DeepResearch.agents.Agent"):  # Patch BaseAgent.__init__
            agent = RAGAgent(model_name="openai:gpt-4")

            assert agent.model_name == "openai:gpt-4"
            assert agent._agent is not None

    @pytest.mark.asyncio
    async def test_query_method_success(self):
        """Test successful RAG query execution."""
        with patch("DeepResearch.agents.Agent"):
            agent = RAGAgent()

            # Create a mock RAGQuery
            mock_rag_query = Mock(spec=RAGQuery)
            mock_rag_query.model_dump.return_value = {
                "text": "test query",
                "search_type": SearchType.SIMILARITY,
                "top_k": 5,
                "score_threshold": 0.7,
                "retrieval_query": "custom retrieval",
                "filters": {"source": "research"},
            }
            mock_rag_query.text = "test query"  # For error fallback

            # Mock the execute method to return a successful result
            mock_result = Mock(spec=AgentResult)
            mock_result.success = True
            mock_result.data = {
                "query": "test query",
                "retrieved_documents": [
                    {
                        "document": {
                            "id": "doc_001",
                            "content": "document content here",
                            "metadata": {"source": "test"},
                        },
                        "score": 0.95,
                        "rank": 1,
                    }
                ],
                "generated_answer": "This is the answer",
                "context": "retrieved context",
                "processing_time": 0.123,
                "metadata": {},
            }

            mock_execute = AsyncMock(return_value=mock_result)

            with patch.object(BaseAgent, "execute", mock_execute):
                result = await agent.query(mock_rag_query)

            # Verify the execute method was called with the dumped query data
            mock_execute.assert_called_once_with(
                {
                    "text": "test query",
                    "search_type": SearchType.SIMILARITY,
                    "top_k": 5,
                    "score_threshold": 0.7,
                    "retrieval_query": "custom retrieval",
                    "filters": {"source": "research"},
                }
            )

            # The result should now be created successfully
            # --- Basic Value Assertions ---
            assert isinstance(result, RAGResponse)
            assert result.query == "test query"
            assert result.generated_answer == "This is the answer"
            assert result.context == "retrieved context"
            assert isinstance(result.metadata, dict)
            assert result.processing_time > 0

            # --- Retrieved Documents Assertions ---
            docs = result.retrieved_documents
            assert isinstance(docs, list)
            assert len(docs) == 1

            doc_result = docs[0]
            assert isinstance(doc_result, SearchResult)
            assert isinstance(doc_result.document, Document)
            assert doc_result.rank == 1
            assert 0 <= doc_result.score <= 1

            # --- Document Content Assertions ---
            doc = doc_result.document
            assert doc.id == "doc_001"
            assert "document content" in doc.content
            assert doc.metadata.get("source") == "test"
            assert doc.updated_at is None
            assert doc.cross_references == {}
            assert doc.quality_score is None

    @pytest.mark.asyncio
    async def test_query_method_failure(self):
        """Test RAG query returns default response when execution fails."""
        with patch("DeepResearch.agents.Agent"):
            agent = RAGAgent()

            mock_rag_query = Mock(spec=RAGQuery)
            mock_rag_query.text = "test query"  # For error response fallback
            mock_rag_query.model_dump.return_value = {"text": "test query"}

            # Mock the execute method to return a failed result
            mock_result = Mock(spec=RAGResponse)
            mock_result.success = False
            mock_result.error = "RAG query failed"

            mock_execute = AsyncMock(return_value=mock_result)

            with patch.object(BaseAgent, "execute", mock_execute):
                result = await agent.query(mock_rag_query)

            # Verify the execute method was called
            mock_execute.assert_called_once_with({"text": "test query"})

            # The agent should handle the failure gracefully and return a default response
            # --- Basic Value Assertions ---
            assert isinstance(result, RAGResponse)
            assert result.query == "test query"
            assert result.generated_answer == ""
            assert result.context == ""
            assert isinstance(result.metadata, dict)
            assert result.processing_time == 0.0
            assert result.retrieved_documents == []

            # Metadata should contain errors
            assert result.metadata == {"error": mock_result.error}

    def test_initialize_agent_handles_exception(self):
        """Initialise the RAGAgent with a random LLM to initiate a failure."""
        # This test is enough in my opinion because inside the BaseAgent class
        # def, a pydantic agent is initialised only if the self.model_name
        # is a valid LLM name, like "openai:gpt-4o" etc.
        # Obviously, no tools are registered!
        agent = RAGAgent("Random_LLM_That_Doesn't_Exist")
        assert agent._agent is None

    def test_initialize_agent_registers_rag_tools_successfully(self, monkeypatch):
        """RAGAgent registers integrated search tools on the pydantic-ai Agent."""

        with patch(
            "DeepResearch.src.tools.integrated_search_tools.IntegratedSearchTool"
        ) as MockIntegrated:
            with patch(
                "DeepResearch.src.tools.integrated_search_tools.RAGSearchTool"
            ) as _:
                with patch("DeepResearch.agents.Agent"):
                    agent = RAGAgent()
                    agent._initialize_agent(system_prompt="Test", instructions="Test")
                    agent._register_tools()
                    assert MockIntegrated.called
                    assert hasattr(agent._agent, "tool")


# BioinformaticsAgent Tests
class TestBioinformaticsAgent:
    """Test BioinformaticsAgent functionality."""

    def test_bioinformatics_agent_default_model(self):
        """Test BioinformaticsAgent initialization with default model."""
        with patch("DeepResearch.agents.Agent"):
            agent = BioinformaticsAgent()

            assert agent.agent_type == AgentType.BIOINFORMATICS
            assert agent.model_name == DEFAULT_PYDANTIC_AI_MODEL
            assert agent._agent is not None

    def test_bioinformatics_agent_custom_model(self):
        """Test BioinformaticsAgent initialization with custom model."""
        with patch("DeepResearch.agents.Agent"):
            agent = BioinformaticsAgent("openai:gpt-4o")

            assert agent.agent_type == AgentType.BIOINFORMATICS
            assert agent.model_name == "openai:gpt-4o"
            assert agent._agent is not None

    @pytest.mark.asyncio
    async def test_fuse_data_success(self, mock_pydantic_agent):
        """Test successful data fusion."""
        mock_pydantic_agent.run = AsyncMock(
            return_value=SimpleNamespace(
                data={
                    "fused_dataset": {
                        "dataset_id": "test_ds",
                        "name": "Test Dataset",
                        "description": "Test description",
                        "source_databases": ["GO", "PubMed"],
                    }
                }
            )
        )
        with _patch_pydantic_agent_class(mock_pydantic_agent):
            agent = BioinformaticsAgent()

            fusion_request = DataFusionRequest(
                request_id="req_1",
                fusion_type="MultiSource",
                source_databases=["GO", "PubMed"],
                quality_threshold=0.8,
            )

            result = await agent.fuse_data(fusion_request)

            assert isinstance(result, FusedDataset)
            assert result.dataset_id == "test_ds"
            assert len(result.source_databases) == 2
            assert result.name == "Test Dataset"

    @pytest.mark.asyncio
    async def test_fuse_data_failure(self, mock_pydantic_agent):
        """Test data fusion failure handling."""
        with _patch_pydantic_agent_class(mock_pydantic_agent):
            agent = BioinformaticsAgent()

            mock_pydantic_agent.run.side_effect = Exception("Fusion failed")

            fusion_request = DataFusionRequest(
                request_id="req_1",
                fusion_type="MultiSource",
                source_databases=["GO"],
                quality_threshold=0.8,
            )

            result = await agent.fuse_data(fusion_request)

            assert result.dataset_id == "error"
            assert "Failed" in result.description

    @pytest.mark.asyncio
    async def test_perform_reasoning_success(self, mock_pydantic_agent):
        """Test successful reasoning operation."""
        mock_pydantic_agent.run = AsyncMock(
            return_value=SimpleNamespace(
                data={
                    "reasoning_result": "test result",
                    "confidence": 0.9,
                }
            )
        )
        with _patch_pydantic_agent_class(mock_pydantic_agent):
            agent = BioinformaticsAgent()

            task = ReasoningTask(
                task_id="task_1",
                task_type="general_reasoning",
                question="test question",
                difficulty_level="medium",
            )
            dataset = FusedDataset(
                dataset_id="ds_1",
                name="Test",
                description="Test dataset",
                source_databases=["GO"],
            )

            result = await agent.perform_reasoning(task, dataset)

            assert "reasoning_result" in result
            assert result["reasoning_result"] == "test result"
            assert result["confidence"] == 0.9

    @pytest.mark.asyncio
    async def test_perform_reasoning_failure(self, mock_pydantic_agent):
        """Test reasoning failure handling."""
        with _patch_pydantic_agent_class(mock_pydantic_agent):
            agent = BioinformaticsAgent()

            mock_pydantic_agent.run.side_effect = Exception("Reasoning failed")

            task = ReasoningTask(
                task_id="task_1",
                task_type="general_reasoning",
                question="test",
                difficulty_level="medium",
            )
            dataset = FusedDataset(
                dataset_id="ds_1",
                name="Test",
                description="Test",
                source_databases=[],
            )

            result = await agent.perform_reasoning(task, dataset)

            assert "error" in result


# DeepSearchAgent Tests
class TestDeepSearchAgent:
    """Test DeepSearchAgent functionality."""

    def test_deepsearch_agent_default_model(self):
        """Test DeepSearchAgent initialization with default model."""
        with patch("DeepResearch.agents.Agent"):
            agent = DeepSearchAgent()

            assert agent.agent_type == AgentType.DEEPSEARCH
            assert agent.model_name == DEFAULT_PYDANTIC_AI_MODEL
            assert agent._agent is not None

    def test_deepsearch_agent_custom_model(self):
        """Test DeepSearchAgent initialization with custom model."""
        with patch("DeepResearch.agents.Agent"):
            agent = DeepSearchAgent("openai:gpt-4o")

            assert agent.agent_type == AgentType.DEEPSEARCH
            assert agent.model_name == "openai:gpt-4o"
            assert agent._agent is not None

    @pytest.mark.asyncio
    async def test_deep_search_success(self, mock_pydantic_agent):
        """Test successful deep search."""
        mock_pydantic_agent.run = AsyncMock(
            return_value=SimpleNamespace(
                data={
                    "answer": "Deep search answer",
                    "iterations": 5,
                    "sources": ["source1", "source2"],
                }
            )
        )
        with _patch_pydantic_agent_class(mock_pydantic_agent):
            agent = DeepSearchAgent()

            result = await agent.deep_search("test question", max_steps=20)

            assert "answer" in result
            assert "iterations" in result

    @pytest.mark.asyncio
    async def test_deep_search_with_custom_steps(self, mock_pydantic_agent):
        """Test deep search with custom max steps."""
        run_mock = AsyncMock(
            return_value=SimpleNamespace(data={"answer": "test", "iterations": 10})
        )
        mock_pydantic_agent.run = run_mock
        with _patch_pydantic_agent_class(mock_pydantic_agent):
            agent = DeepSearchAgent()

            _ = await agent.deep_search("question", max_steps=10)

            # Verify execution was called with correct params
            run_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_deep_search_failure(self, mock_pydantic_agent):
        """Test deep search failure handling."""
        with _patch_pydantic_agent_class(mock_pydantic_agent):
            agent = DeepSearchAgent()

            mock_pydantic_agent.run.side_effect = Exception("Deep search failed")

            result = await agent.deep_search("question")

            assert "error" in result


# EvaluatorAgent Tests
class TestEvaluatorAgent:
    """Test EvaluatorAgent functionality."""

    def test_evaluator_agent_default_model(self):
        """Test EvaluatorAgent initialization with default model."""
        with patch("DeepResearch.agents.Agent"):
            agent = EvaluatorAgent()

            assert agent.agent_type == AgentType.EVALUATOR
            assert agent.model_name == DEFAULT_PYDANTIC_AI_MODEL
            assert agent._agent is not None

    def test_evaluator_agent_custom_model(self):
        """Test EvaluatorAgent initialization with default model."""
        with patch("DeepResearch.agents.Agent"):
            agent = EvaluatorAgent("openai:gpt-4o")

            assert agent.agent_type == AgentType.EVALUATOR
            assert agent.model_name == "openai:gpt-4o"
            assert agent._agent is not None

    @pytest.mark.asyncio
    async def test_evaluate_success(self, mock_pydantic_agent):
        """Test successful evaluation."""
        mock_pydantic_agent.run = AsyncMock(
            return_value=SimpleNamespace(
                data={
                    "quality_score": 0.85,
                    "completeness": 0.9,
                    "accuracy": 0.8,
                    "feedback": "Good answer",
                }
            )
        )
        with _patch_pydantic_agent_class(mock_pydantic_agent):
            agent = EvaluatorAgent()

            result = await agent.evaluate(
                "What is AI?", "AI is artificial intelligence"
            )

            assert "quality_score" in result
            assert result["quality_score"] == 0.85

    @pytest.mark.asyncio
    async def test_evaluate_poor_answer(self, mock_pydantic_agent):
        """Test evaluation of poor answer."""
        mock_pydantic_agent.run = AsyncMock(
            return_value=SimpleNamespace(
                data={
                    "quality_score": 0.3,
                    "completeness": 0.2,
                    "accuracy": 0.4,
                    "feedback": "Needs improvement",
                }
            )
        )
        with _patch_pydantic_agent_class(mock_pydantic_agent):
            agent = EvaluatorAgent()

            result = await agent.evaluate("Complex question", "Short answer")

            assert result["quality_score"] < 0.5

    @pytest.mark.asyncio
    async def test_evaluate_failure(self, mock_pydantic_agent):
        """Test evaluation failure handling."""
        with _patch_pydantic_agent_class(mock_pydantic_agent):
            agent = EvaluatorAgent()

            mock_pydantic_agent.run.side_effect = Exception("Evaluation failed")

            result = await agent.evaluate("question", "answer")

            assert "error" in result


# Integration Tests for DeepAgent w/ PlanningAgent
class TestDeepAgentPlanningAgent:
    """Integration tests for DeepAgent fused with other Agents."""

    def test_initialization_default_model(self):
        """Test DeepAgentPlanningAgent initialization with default model."""
        with patch("DeepResearch.agents.Agent"):
            agent = DeepAgentPlanningAgent()

            mock_deep_agent = Mock(spec=PlanningAgent)
            agent._deep_agent = mock_deep_agent

            assert agent.agent_type == AgentType.DEEP_AGENT_PLANNING
            assert agent.model_name == DEFAULT_PYDANTIC_AI_MODEL
            assert agent._deep_agent is not None

    @pytest.mark.asyncio
    async def test_create_plan_with_deep_agent(self):
        agent = DeepAgentPlanningAgent()

        mock_deep_agent = Mock(spec=PlanningAgent)
        mock_result = AgentExecutionResult(
            success=True,
            result={"plan": "Mock plan"},
            error=None,
            execution_time=0.12,
            tools_used=["mock_tool"],
        )
        mock_deep_agent.create_plan.return_value = mock_result

        # Inject mock deep agent
        agent._deep_agent = mock_deep_agent

        result = await agent.create_plan("Organize research tasks")
        assert result.result is not None
        assert result.success is True
        assert "plan" in result.result.keys()
        mock_deep_agent.create_plan.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_create_plan_without_deep_agent(self):
        agent = DeepAgentPlanningAgent()
        agent._deep_agent = None

        with patch.object(agent, "execute", AsyncMock()) as mock_execute:
            mock_execute.return_value = AsyncMock(
                success=True,
                data={"plan": "Fallback plan"},
                error=None,
                execution_time=0.2,
            )

            result = await agent.create_plan("Fallback planning task")

        assert result.result is not None
        assert result.success is True
        assert result.tools_used == ["standard_planning"]
        assert result.result["plan"] == "Fallback plan"
        mock_execute.assert_awaited_once()


class TestDeepAgentFilesystem:
    """Integration tests for DeepAgent fused with other Agents."""

    def test_initialization_default_model(self):
        """Test DeepAgentFilesystemAgent initialization with default model."""
        with patch("DeepResearch.agents.Agent"):
            agent = DeepAgentFilesystemAgent()

            mock_deep_agent = Mock(spec=PlanningAgent)
            agent._deep_agent = mock_deep_agent

            assert agent.agent_type == AgentType.DEEP_AGENT_FILESYSTEM
            assert agent.model_name == DEFAULT_PYDANTIC_AI_MODEL
            assert agent._deep_agent is not None

    @pytest.mark.asyncio
    async def test_manage_files_with_deep_agent(self):
        agent = DeepAgentFilesystemAgent()
        mock_deep_agent = AsyncMock()
        mock_result = AgentExecutionResult(
            success=True,
            result={"status": "file_listed"},
            error=None,
            execution_time=0.05,
            tools_used=["list_files"],
        )
        mock_deep_agent.manage_files.return_value = mock_result
        agent._deep_agent = mock_deep_agent

        result = await agent.manage_files("list_files")
        assert result.success
        assert result.result is not None
        assert result.result["status"] == "file_listed"
        mock_deep_agent.manage_files.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_manage_files_fallback(self):
        agent = DeepAgentFilesystemAgent()
        agent._deep_agent = None

        with patch.object(agent, "execute", AsyncMock()) as mock_execute:
            mock_execute.return_value = AsyncMock(
                success=True,
                data={"status": "fallback_operation"},
                error=None,
                execution_time=0.12,
            )

            result = await agent.manage_files("write_file")

        assert result.success
        assert result.result is not None
        assert result.result["status"] == "fallback_operation"
        assert result.tools_used == ["standard_filesystem"]


class TestDeepAgentResearchAgent:
    @pytest.mark.asyncio
    async def test_conduct_research_with_deep_agent(self):
        agent = DeepAgentResearchAgent()
        mock_deep_agent = AsyncMock()
        mock_result = AgentExecutionResult(
            success=True,
            result={"summary": "AI accelerates discovery"},
            error=None,
            execution_time=0.32,
            tools_used=["rag_query"],
        )
        mock_deep_agent.conduct_research.return_value = mock_result
        agent._deep_agent = mock_deep_agent

        result = await agent.conduct_research("impact of AI on bioinformatics")
        assert result.success
        assert result.result is not None
        assert "summary" in result.result
        mock_deep_agent.conduct_research.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_conduct_research_fallback(self):
        agent = DeepAgentResearchAgent()
        agent._deep_agent = None

        with patch.object(agent, "execute", AsyncMock()) as mock_execute:
            mock_execute.return_value = AsyncMock(
                success=True,
                data={"summary": "Fallback executed"},
                error=None,
                execution_time=0.15,
            )

            result = await agent.conduct_research("fallback test")

        assert result.success
        assert result.result is not None
        assert result.result["summary"] == "Fallback executed"
        assert result.tools_used == ["standard_research"]


def test_agent_creation():
    agent_classes = {
        AgentType.PARSER: ParserAgent,
        AgentType.PLANNER: PlannerAgent,
        AgentType.EXECUTOR: ExecutorAgent,
        AgentType.SEARCH: SearchAgent,
        AgentType.RAG: RAGAgent,
        AgentType.BIOINFORMATICS: BioinformaticsAgent,
        AgentType.DEEPSEARCH: DeepSearchAgent,
        AgentType.EVALUATOR: EvaluatorAgent,
        # DeepAgent types
        AgentType.DEEP_AGENT_PLANNING: DeepAgentPlanningAgent,
        AgentType.DEEP_AGENT_FILESYSTEM: DeepAgentFilesystemAgent,
        AgentType.DEEP_AGENT_RESEARCH: DeepAgentResearchAgent,
        AgentType.DEEP_AGENT_ORCHESTRATION: DeepAgentOrchestrationAgent,
        AgentType.DEEP_AGENT_GENERAL: DeepAgentGeneralAgent,
    }
    for agent_type, agent_class in agent_classes.items():
        with patch("DeepResearch.agents.Agent"):
            kwargs: dict = {}
            if agent_type in {
                AgentType.DEEP_AGENT_PLANNING,
                AgentType.DEEP_AGENT_FILESYSTEM,
                AgentType.DEEP_AGENT_RESEARCH,
                AgentType.DEEP_AGENT_ORCHESTRATION,
                AgentType.DEEP_AGENT_GENERAL,
            }:
                kwargs["config"] = {"deep_agent": {"enabled": True}}
            agent = create_agent(agent_type=agent_type, **kwargs)

            assert agent.agent_type == agent_type
            assert isinstance(agent.model_name, str)
            assert agent.model_name
            assert agent._agent is not None
            assert isinstance(agent, agent_class)


# # Additional Integration Tests
# class TestSpecializedAgentsIntegration:
#     """Integration tests for specialized agents."""

#     @pytest.mark.asyncio
#     async def test_parser_planner_executor_workflow(self):
#         """Test workflow from parser to planner to executor."""
#         with patch("DeepResearch.agents.Agent") as mock_agent_class:
#             with patch("DeepResearch.agents.registry") as mock_registry:
#                 # Setup mocks
#                 mock_agent = Mock()
#                 mock_agent.run = AsyncMock()
#                 mock_agent.tool = Mock()
#                 mock_agent_class.return_value = mock_agent

#                 # Parser result
#                 mock_agent.run.return_value = Mock(
#                     data={"intent": "research", "query": "test"}
#                 )
#                 parser = ParserAgent()
#                 parsed = await parser.parse_question("test question")

#                 # Planner result
#                 mock_agent.run.return_value = Mock(
#                     data={"steps": [{"tool": "test_tool", "params": {"query": "test"}}]}
#                 )
#                 planner = PlannerAgent()
#                 plan = await planner.create_plan(parsed)

#                 # Executor
#                 mock_runner = Mock()
#                 mock_runner.run.return_value = Mock(
#                     success=True, data={"result": "final"}
#                 )
#                 mock_registry.make.return_value = mock_runner

#                 executor = ExecutorAgent()
#                 history = ExecutionHistory()
#                 result = await executor.execute_plan(plan, history)

#                 assert "result" in result or "test_tool.result" in result

#     @pytest.mark.asyncio
#     async def test_search_rag_workflow(self, mock_pydantic_agent):
#         """Test workflow from search to RAG."""
#         with _patch_pydantic_agent_class(mock_pydantic_agent):
#             # Search
#             search_agent = SearchAgent()
#             mock_pydantic_agent.run.return_value = Mock(
#                 data={"results": [{"content": "doc1"}]}
#             )
#             search_results = await search_agent.search("query")

#             # RAG with search results
#             rag_agent = RAGAgent()
#             mock_pydantic_agent.run.return_value = Mock(
#                 data={
#                     "query": "query",
#                     "retrieved_documents": search_results.get("results", []),
#                     "generated_answer": "answer",
#                     "context": "context",
#                     "processing_time": 1.0,
#                     "metadata": {},
#                 }
#             )

#             rag_query = RAGQuery(text="query", top_k=5)
#             rag_result = await rag_agent.query(rag_query)

#             assert isinstance(rag_result, RAGResponse)

#     @pytest.mark.asyncio
#     async def test_agent_error_propagation(self, mock_pydantic_agent):
#         """Test error propagation through agent chain."""
#         with _patch_pydantic_agent_class(mock_pydantic_agent):
#             # Parser fails
#             parser = ParserAgent()
#             mock_pydantic_agent.run.side_effect = Exception("Parser error")
#             parsed = await parser.parse_question("test")

#             assert "error" in parsed

#             # Planner uses fallback
#             planner = PlannerAgent()
#             mock_pydantic_agent.run.side_effect = Exception("Planner error")
#             plan = planner.plan(parsed)

#             # Should get default plan
#             assert len(plan) > 0

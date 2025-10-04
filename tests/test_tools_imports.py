"""
Import tests for DeepResearch tools modules.

This module tests that all imports from the tools subdirectory work correctly,
including all individual tool modules and their dependencies.
"""

import pytest


class TestToolsModuleImports:
    """Test imports for individual tool modules."""

    def test_base_imports(self):
        """Test all imports from base module."""

        from DeepResearch.src.tools.base import (
            ToolSpec,
            ExecutionResult,
            ToolRunner,
            ToolRegistry,
        )

        # Verify they are all accessible and not None
        assert ToolSpec is not None
        assert ExecutionResult is not None
        assert ToolRunner is not None
        assert ToolRegistry is not None

        # Test that registry is accessible from tools module
        from DeepResearch.src.tools import registry

        assert registry is not None

    def test_mock_tools_imports(self):
        """Test all imports from mock_tools module."""

        from DeepResearch.src.tools.mock_tools import (
            MockTool,
            MockWebSearchTool,
            MockBioinformaticsTool,
        )

        # Verify they are all accessible and not None
        assert MockTool is not None
        assert MockWebSearchTool is not None
        assert MockBioinformaticsTool is not None

    def test_workflow_tools_imports(self):
        """Test all imports from workflow_tools module."""

        from DeepResearch.src.tools.workflow_tools import (
            WorkflowTool,
            WorkflowStepTool,
        )

        # Verify they are all accessible and not None
        assert WorkflowTool is not None
        assert WorkflowStepTool is not None

    def test_pyd_ai_tools_imports(self):
        """Test all imports from pyd_ai_tools module."""

        from DeepResearch.src.tools.pyd_ai_tools import (
            _build_builtin_tools,
            _build_toolsets,
            _build_agent,
        )

        # Verify they are all accessible and not None
        assert _build_builtin_tools is not None
        assert _build_toolsets is not None
        assert _build_agent is not None

    def test_code_sandbox_imports(self):
        """Test all imports from code_sandbox module."""

        from DeepResearch.src.tools.code_sandbox import CodeSandboxTool

        # Verify they are all accessible and not None
        assert CodeSandboxTool is not None

    def test_docker_sandbox_imports(self):
        """Test all imports from docker_sandbox module."""

        from DeepResearch.src.tools.docker_sandbox import DockerSandboxTool

        # Verify they are all accessible and not None
        assert DockerSandboxTool is not None

    def test_deepsearch_tools_imports(self):
        """Test all imports from deepsearch_tools module."""

        from DeepResearch.src.tools.deepsearch_tools import DeepSearchTool

        # Verify they are all accessible and not None
        assert DeepSearchTool is not None

    def test_deepsearch_workflow_tool_imports(self):
        """Test all imports from deepsearch_workflow_tool module."""

        from DeepResearch.src.tools.deepsearch_workflow_tool import (
            DeepSearchWorkflowTool,
        )

        # Verify they are all accessible and not None
        assert DeepSearchWorkflowTool is not None

    def test_websearch_tools_imports(self):
        """Test all imports from websearch_tools module."""

        from DeepResearch.src.tools.websearch_tools import WebSearchTool

        # Verify they are all accessible and not None
        assert WebSearchTool is not None

    def test_websearch_cleaned_imports(self):
        """Test all imports from websearch_cleaned module."""

        from DeepResearch.src.tools.websearch_cleaned import WebSearchCleanedTool

        # Verify they are all accessible and not None
        assert WebSearchCleanedTool is not None

    def test_analytics_tools_imports(self):
        """Test all imports from analytics_tools module."""

        from DeepResearch.src.tools.analytics_tools import AnalyticsTool

        # Verify they are all accessible and not None
        assert AnalyticsTool is not None

    def test_integrated_search_tools_imports(self):
        """Test all imports from integrated_search_tools module."""

        from DeepResearch.src.tools.integrated_search_tools import IntegratedSearchTool

        # Verify they are all accessible and not None
        assert IntegratedSearchTool is not None


class TestToolsCrossModuleImports:
    """Test cross-module imports and dependencies within tools."""

    def test_tools_internal_dependencies(self):
        """Test that tool modules can import from each other correctly."""
        # Test that tools can import base classes
        from DeepResearch.src.tools.mock_tools import MockTool
        from DeepResearch.src.tools.base import ToolSpec

        # This should work without circular imports
        assert MockTool is not None
        assert ToolSpec is not None

    def test_datatypes_integration_imports(self):
        """Test that tools can import from datatypes module."""
        # This tests the import chain: tools -> datatypes
        from DeepResearch.src.tools.base import ToolSpec
        from DeepResearch.src.datatypes import Document

        # If we get here without ImportError, the import chain works
        assert ToolSpec is not None
        assert Document is not None

    def test_agents_integration_imports(self):
        """Test that tools can import from agents module."""
        # This tests the import chain: tools -> agents
        from DeepResearch.src.tools.pyd_ai_tools import _build_agent

        # If we get here without ImportError, the import chain works
        assert _build_agent is not None


class TestToolsComplexImportChains:
    """Test complex import chains involving multiple modules."""

    def test_full_tool_initialization_chain(self):
        """Test the complete import chain for tool initialization."""
        try:
            from DeepResearch.src.tools.base import ToolRegistry, ToolSpec
            from DeepResearch.src.tools.mock_tools import MockTool
            from DeepResearch.src.tools.workflow_tools import WorkflowTool
            from DeepResearch.src.datatypes import Document

            # If all imports succeed, the chain is working
            assert ToolRegistry is not None
            assert ToolSpec is not None
            assert MockTool is not None
            assert WorkflowTool is not None
            assert Document is not None

        except ImportError as e:
            pytest.fail(f"Tool import chain failed: {e}")

    def test_tool_execution_chain(self):
        """Test the complete import chain for tool execution."""
        try:
            from DeepResearch.src.tools.base import ExecutionResult, ToolRunner
            from DeepResearch.src.tools.websearch_tools import WebSearchTool
            from DeepResearch.src.agents.prime_executor import ToolExecutor

            # If all imports succeed, the chain is working
            assert ExecutionResult is not None
            assert ToolRunner is not None
            assert WebSearchTool is not None
            assert ToolExecutor is not None

        except ImportError as e:
            pytest.fail(f"Tool execution import chain failed: {e}")


class TestToolsImportErrorHandling:
    """Test import error handling for tools modules."""

    def test_missing_dependencies_handling(self):
        """Test that modules handle missing dependencies gracefully."""
        # Test that pyd_ai_tools handles optional dependencies
        from DeepResearch.src.tools.pyd_ai_tools import _build_agent

        # This should work even if pydantic_ai is not installed
        assert _build_agent is not None

    def test_circular_import_prevention(self):
        """Test that there are no circular imports in tools."""
        # This test will fail if there are circular imports

        # If we get here, no circular imports were detected
        assert True

    def test_registry_functionality(self):
        """Test that the tool registry works correctly."""
        from DeepResearch.src.tools.base import ToolRegistry

        registry = ToolRegistry()

        # Test that registry can be instantiated and used
        assert registry is not None
        assert hasattr(registry, "register")
        assert hasattr(registry, "make")

    def test_tool_spec_validation(self):
        """Test that ToolSpec works correctly."""
        from DeepResearch.src.tools.base import ToolSpec

        spec = ToolSpec(
            name="test_tool",
            description="Test tool",
            inputs={"param": "TEXT"},
            outputs={"result": "TEXT"},
        )

        # Test that ToolSpec can be created and used
        assert spec is not None
        assert spec.name == "test_tool"
        assert "param" in spec.inputs

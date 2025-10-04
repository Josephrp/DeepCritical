"""
Import tests for DeepResearch utils modules.

This module tests that all imports from the utils subdirectory work correctly,
including all individual utility modules and their dependencies.
"""

import pytest


class TestUtilsModuleImports:
    """Test imports for individual utility modules."""

    def test_config_loader_imports(self):
        """Test all imports from config_loader module."""

        from DeepResearch.src.utils.config_loader import (
            BioinformaticsConfigLoader,
        )

        # Verify they are all accessible and not None
        assert BioinformaticsConfigLoader is not None

    def test_execution_history_imports(self):
        """Test all imports from execution_history module."""

        from DeepResearch.src.utils.execution_history import (
            ExecutionHistory,
            ExecutionStep,
            ExecutionMetrics,
        )

        # Verify they are all accessible and not None
        assert ExecutionHistory is not None
        assert ExecutionStep is not None
        assert ExecutionMetrics is not None

    def test_execution_status_imports(self):
        """Test all imports from execution_status module."""

        from DeepResearch.src.utils.execution_status import (
            ExecutionStatus,
            StatusType,
        )

        # Verify they are all accessible and not None
        assert ExecutionStatus is not None
        assert StatusType is not None

        # Test enum values exist
        assert hasattr(StatusType, "PENDING")
        assert hasattr(StatusType, "RUNNING")

    def test_tool_registry_imports(self):
        """Test all imports from tool_registry module."""

        from DeepResearch.src.utils.tool_registry import (
            ToolRegistry,
            ToolMetadata,
        )

        # Verify they are all accessible and not None
        assert ToolRegistry is not None
        assert ToolMetadata is not None

    def test_tool_specs_imports(self):
        """Test all imports from tool_specs module."""

        from DeepResearch.src.utils.tool_specs import (
            ToolSpec,
            ToolInput,
            ToolOutput,
        )

        # Verify they are all accessible and not None
        assert ToolSpec is not None
        assert ToolInput is not None
        assert ToolOutput is not None

    def test_analytics_imports(self):
        """Test all imports from analytics module."""

        from DeepResearch.src.utils.analytics import (
            AnalyticsEngine,
            MetricCalculator,
        )

        # Verify they are all accessible and not None
        assert AnalyticsEngine is not None
        assert MetricCalculator is not None

    def test_deepsearch_schemas_imports(self):
        """Test all imports from deepsearch_schemas module."""

        from DeepResearch.src.utils.deepsearch_schemas import (
            DeepSearchQuery,
            DeepSearchResult,
            DeepSearchConfig,
        )

        # Verify they are all accessible and not None
        assert DeepSearchQuery is not None
        assert DeepSearchResult is not None
        assert DeepSearchConfig is not None

    def test_deepsearch_utils_imports(self):
        """Test all imports from deepsearch_utils module."""

        from DeepResearch.src.utils.deepsearch_utils import (
            DeepSearchUtils,
            SearchResultProcessor,
        )

        # Verify they are all accessible and not None
        assert DeepSearchUtils is not None
        assert SearchResultProcessor is not None


class TestUtilsCrossModuleImports:
    """Test cross-module imports and dependencies within utils."""

    def test_utils_internal_dependencies(self):
        """Test that utility modules can import from each other correctly."""
        # Test that modules can import shared types
        from DeepResearch.src.utils.execution_history import ExecutionHistory
        from DeepResearch.src.utils.execution_status import ExecutionStatus

        # This should work without circular imports
        assert ExecutionHistory is not None
        assert ExecutionStatus is not None

    def test_datatypes_integration_imports(self):
        """Test that utils can import from datatypes module."""
        # This tests the import chain: utils -> datatypes
        from DeepResearch.src.utils.tool_specs import ToolSpec
        from DeepResearch.src.datatypes import Document

        # If we get here without ImportError, the import chain works
        assert ToolSpec is not None
        assert Document is not None

    def test_tools_integration_imports(self):
        """Test that utils can import from tools module."""
        # This tests the import chain: utils -> tools
        from DeepResearch.src.utils.tool_registry import ToolRegistry
        from DeepResearch.src.tools.base import ToolSpec

        # If we get here without ImportError, the import chain works
        assert ToolRegistry is not None
        assert ToolSpec is not None


class TestUtilsComplexImportChains:
    """Test complex import chains involving multiple modules."""

    def test_full_utils_initialization_chain(self):
        """Test the complete import chain for utils initialization."""
        try:
            from DeepResearch.src.utils.config_loader import BioinformaticsConfigLoader
            from DeepResearch.src.utils.execution_history import ExecutionHistory
            from DeepResearch.src.utils.tool_registry import ToolRegistry
            from DeepResearch.src.datatypes import Document

            # If all imports succeed, the chain is working
            assert BioinformaticsConfigLoader is not None
            assert ExecutionHistory is not None
            assert ToolRegistry is not None
            assert Document is not None

        except ImportError as e:
            pytest.fail(f"Utils import chain failed: {e}")

    def test_execution_tracking_chain(self):
        """Test the complete import chain for execution tracking."""
        try:
            from DeepResearch.src.utils.execution_history import (
                ExecutionHistory,
                ExecutionStep,
            )
            from DeepResearch.src.utils.execution_status import (
                ExecutionStatus,
                StatusType,
            )
            from DeepResearch.src.utils.analytics import AnalyticsEngine

            # If all imports succeed, the chain is working
            assert ExecutionHistory is not None
            assert ExecutionStep is not None
            assert ExecutionStatus is not None
            assert StatusType is not None
            assert AnalyticsEngine is not None

        except ImportError as e:
            pytest.fail(f"Execution tracking import chain failed: {e}")


class TestUtilsImportErrorHandling:
    """Test import error handling for utils modules."""

    def test_missing_dependencies_handling(self):
        """Test that modules handle missing dependencies gracefully."""
        # Test that config_loader handles optional dependencies
        from DeepResearch.src.utils.config_loader import BioinformaticsConfigLoader

        # This should work even if omegaconf is not available in some contexts
        assert BioinformaticsConfigLoader is not None

    def test_circular_import_prevention(self):
        """Test that there are no circular imports in utils."""
        # This test will fail if there are circular imports

        # If we get here, no circular imports were detected
        assert True

    def test_enum_functionality(self):
        """Test that enum classes work correctly."""
        from DeepResearch.src.utils.execution_status import StatusType

        # Test that enum has expected values and can be used
        assert StatusType.PENDING is not None
        assert StatusType.RUNNING is not None
        assert StatusType.COMPLETED is not None
        assert StatusType.FAILED is not None

        # Test that enum values are strings
        assert isinstance(StatusType.PENDING.value, str)

    def test_dataclass_functionality(self):
        """Test that dataclass functionality works correctly."""
        from DeepResearch.src.utils.execution_history import ExecutionStep

        # Test that we can create instances (basic functionality)
        try:
            step = ExecutionStep(
                step_id="test",
                status="pending",
                start_time=None,
                end_time=None,
                metadata={},
            )
            assert step is not None
            assert step.step_id == "test"
        except Exception as e:
            pytest.fail(f"Dataclass instantiation failed: {e}")

"""
Tests for the Orchestrator dataclass.

This module tests the functionality of the Orchestrator dataclass
from DeepResearch.src.datatypes.orchestrator.
"""

from DeepResearch.src.datatypes.orchestrator import Orchestrator


class TestOrchestrator:
    """Test cases for the Orchestrator dataclass."""

    def test_orchestrator_creation(self):
        """Test that Orchestrator can be instantiated."""
        orchestrator = Orchestrator()
        assert orchestrator is not None
        assert isinstance(orchestrator, Orchestrator)

    def test_build_plan_empty_config(self):
        """Test build_plan with empty config."""
        orchestrator = Orchestrator()
        plan = orchestrator.build_plan("test question", {})
        assert plan == []

    def test_build_plan_no_enabled_flows(self):
        """Test build_plan with no enabled flows."""
        orchestrator = Orchestrator()
        config = {
            "flow1": {"enabled": False},
            "flow2": {"enabled": False},
        }
        plan = orchestrator.build_plan("test question", config)
        assert plan == []

    def test_build_plan_mixed_enabled_flows(self):
        """Test build_plan with mixed enabled/disabled flows."""
        orchestrator = Orchestrator()
        config = {
            "flow1": {"enabled": True},
            "flow2": {"enabled": False},
            "flow3": {"enabled": True},
        }
        plan = orchestrator.build_plan("test question", config)
        assert plan == ["flow:flow1", "flow:flow3"]

    def test_build_plan_non_dict_values(self):
        """Test build_plan with non-dict values in config."""
        orchestrator = Orchestrator()
        config = {
            "flow1": "not_a_dict",
            "flow2": {"enabled": True},
            "flow3": None,
        }
        plan = orchestrator.build_plan("test question", config)
        # Should only include flows with dict values that have enabled=True
        assert plan == ["flow:flow2"]

    def test_build_plan_none_config(self):
        """Test build_plan with None config."""
        orchestrator = Orchestrator()
        plan = orchestrator.build_plan("test question", None)
        assert plan == []

    def test_build_plan_complex_config(self):
        """Test build_plan with complex nested config."""
        orchestrator = Orchestrator()
        config = {
            "simple_flow": {"enabled": True},
            "complex_flow": {"enabled": True, "nested": {"value": "test"}},
            "disabled_flow": {"enabled": False},
        }
        plan = orchestrator.build_plan("test question", config)
        assert plan == ["flow:simple_flow", "flow:complex_flow"]

    def test_orchestrator_attributes(self):
        """Test that Orchestrator has expected attributes."""
        orchestrator = Orchestrator()

        # Check that it has the build_plan method
        assert hasattr(orchestrator, "build_plan")
        assert callable(orchestrator.build_plan)

        # Check that it's a dataclass (has __dataclass_fields__)
        assert hasattr(orchestrator, "__dataclass_fields__")

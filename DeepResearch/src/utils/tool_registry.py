from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type
from abc import ABC, abstractmethod
import importlib
import inspect

from .tool_specs import ToolSpec, ToolCategory


@dataclass
class ToolMetadata:
    """Metadata for registered tools."""

    name: str
    category: ToolCategory
    description: str
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)


@dataclass
class ExecutionResult:
    """Result of tool execution."""

    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ToolRunner(ABC):
    """Abstract base class for tool runners."""

    def __init__(self, tool_spec: ToolSpec):
        self.tool_spec = tool_spec

    @abstractmethod
    def run(self, parameters: Dict[str, Any]) -> ExecutionResult:
        """Execute the tool with given parameters."""
        pass

    def validate_inputs(self, parameters: Dict[str, Any]) -> ExecutionResult:
        """Validate input parameters against tool specification."""
        for param_name, expected_type in self.tool_spec.input_schema.items():
            if param_name not in parameters:
                return ExecutionResult(
                    success=False, error=f"Missing required parameter: {param_name}"
                )

            if not self._validate_type(parameters[param_name], expected_type):
                return ExecutionResult(
                    success=False,
                    error=f"Invalid type for parameter '{param_name}': expected {expected_type}",
                )

        return ExecutionResult(success=True)

    def _validate_type(self, value: Any, expected_type: str) -> bool:
        """Validate that value matches expected type."""
        type_mapping = {
            "string": str,
            "int": int,
            "float": float,
            "list": list,
            "dict": dict,
            "bool": bool,
        }

        expected_python_type = type_mapping.get(expected_type, Any)
        return isinstance(value, expected_python_type)


class MockToolRunner(ToolRunner):
    """Mock implementation of tool runner for testing."""

    def run(self, parameters: Dict[str, Any]) -> ExecutionResult:
        """Mock execution that returns simulated results."""
        # Validate inputs first
        validation = self.validate_inputs(parameters)
        if not validation.success:
            return validation

        # Generate mock results based on tool type
        if self.tool_spec.category == ToolCategory.KNOWLEDGE_QUERY:
            return self._mock_knowledge_query(parameters)
        elif self.tool_spec.category == ToolCategory.SEQUENCE_ANALYSIS:
            return self._mock_sequence_analysis(parameters)
        elif self.tool_spec.category == ToolCategory.STRUCTURE_PREDICTION:
            return self._mock_structure_prediction(parameters)
        elif self.tool_spec.category == ToolCategory.MOLECULAR_DOCKING:
            return self._mock_molecular_docking(parameters)
        elif self.tool_spec.category == ToolCategory.DE_NOVO_DESIGN:
            return self._mock_de_novo_design(parameters)
        elif self.tool_spec.category == ToolCategory.FUNCTION_PREDICTION:
            return self._mock_function_prediction(parameters)
        else:
            return ExecutionResult(
                success=True,
                data={"result": "mock_execution_completed"},
                metadata={"tool": self.tool_spec.name, "mock": True},
            )

    def _mock_knowledge_query(self, parameters: Dict[str, Any]) -> ExecutionResult:
        """Mock knowledge query results."""
        query = parameters.get("query", "")
        return ExecutionResult(
            success=True,
            data={
                "sequences": [
                    "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
                ],
                "annotations": {
                    "organism": "Homo sapiens",
                    "function": "Protein function annotation",
                    "confidence": 0.95,
                },
            },
            metadata={"query": query, "mock": True},
        )

    def _mock_sequence_analysis(self, parameters: Dict[str, Any]) -> ExecutionResult:
        """Mock sequence analysis results."""
        sequence = parameters.get("sequence", "")
        return ExecutionResult(
            success=True,
            data={
                "hits": [
                    {
                        "id": "P12345",
                        "description": "Similar protein",
                        "e_value": 1e-10,
                    },
                    {
                        "id": "Q67890",
                        "description": "Another similar protein",
                        "e_value": 1e-8,
                    },
                ],
                "e_values": [1e-10, 1e-8],
                "domains": [{"name": "PF00001", "start": 10, "end": 50, "score": 25.5}],
            },
            metadata={"sequence_length": len(sequence), "mock": True},
        )

    def _mock_structure_prediction(self, parameters: Dict[str, Any]) -> ExecutionResult:
        """Mock structure prediction results."""
        sequence = parameters.get("sequence", "")
        return ExecutionResult(
            success=True,
            data={
                "structure": "ATOM      1  N   ALA A   1      20.154  16.967  23.862  1.00 11.18           N",
                "confidence": {
                    "plddt": 85.5,
                    "global_confidence": 0.89,
                    "per_residue_confidence": [0.9, 0.85, 0.88, 0.92],
                },
            },
            metadata={"sequence_length": len(sequence), "mock": True},
        )

    def _mock_molecular_docking(self, parameters: Dict[str, Any]) -> ExecutionResult:
        """Mock molecular docking results."""
        return ExecutionResult(
            success=True,
            data={
                "poses": [
                    {"id": 1, "binding_affinity": -7.2, "rmsd": 1.5},
                    {"id": 2, "binding_affinity": -6.8, "rmsd": 2.1},
                ],
                "binding_affinity": -7.2,
                "confidence": 0.75,
            },
            metadata={"num_poses": 2, "mock": True},
        )

    def _mock_de_novo_design(self, parameters: Dict[str, Any]) -> ExecutionResult:
        """Mock de novo design results."""
        num_designs = parameters.get("num_designs", 1)
        return ExecutionResult(
            success=True,
            data={
                "structures": [
                    f"DESIGNED_STRUCTURE_{i + 1}.pdb" for i in range(num_designs)
                ],
                "sequences": [
                    f"MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG_{i + 1}"
                    for i in range(num_designs)
                ],
                "confidence": 0.82,
            },
            metadata={"num_designs": num_designs, "mock": True},
        )

    def _mock_function_prediction(self, parameters: Dict[str, Any]) -> ExecutionResult:
        """Mock function prediction results."""
        return ExecutionResult(
            success=True,
            data={
                "function": "Enzyme activity",
                "confidence": 0.88,
                "predictions": {
                    "catalytic_activity": 0.92,
                    "binding_activity": 0.75,
                    "structural_stability": 0.85,
                },
            },
            metadata={"mock": True},
        )


class ToolRegistry:
    """Registry for managing and executing tools in the PRIME ecosystem."""

    def __init__(self):
        self.tools: Dict[str, ToolSpec] = {}
        self.runners: Dict[str, ToolRunner] = {}
        self.mock_mode = True  # Default to mock mode for development

    def register_tool(
        self, tool_spec: ToolSpec, runner_class: Optional[Type[ToolRunner]] = None
    ) -> None:
        """Register a tool with its specification and runner."""
        self.tools[tool_spec.name] = tool_spec

        if runner_class:
            self.runners[tool_spec.name] = runner_class(tool_spec)
        elif self.mock_mode:
            self.runners[tool_spec.name] = MockToolRunner(tool_spec)

    def get_tool_spec(self, tool_name: str) -> Optional[ToolSpec]:
        """Get tool specification by name."""
        return self.tools.get(tool_name)

    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self.tools.keys())

    def list_tools_by_category(self, category: ToolCategory) -> List[str]:
        """List tools by category."""
        return [name for name, spec in self.tools.items() if spec.category == category]

    def execute_tool(
        self, tool_name: str, parameters: Dict[str, Any]
    ) -> ExecutionResult:
        """Execute a tool with given parameters."""
        if tool_name not in self.tools:
            return ExecutionResult(success=False, error=f"Tool not found: {tool_name}")

        if tool_name not in self.runners:
            return ExecutionResult(
                success=False, error=f"No runner registered for tool: {tool_name}"
            )

        runner = self.runners[tool_name]
        return runner.run(parameters)

    def validate_tool_execution(
        self, tool_name: str, parameters: Dict[str, Any]
    ) -> ExecutionResult:
        """Validate tool execution without running it."""
        if tool_name not in self.tools:
            return ExecutionResult(success=False, error=f"Tool not found: {tool_name}")

        if tool_name not in self.runners:
            return ExecutionResult(
                success=False, error=f"No runner registered for tool: {tool_name}"
            )

        runner = self.runners[tool_name]
        return runner.validate_inputs(parameters)

    def get_tool_dependencies(self, tool_name: str) -> List[str]:
        """Get dependencies for a tool."""
        if tool_name not in self.tools:
            return []

        return self.tools[tool_name].dependencies

    def check_dependency_availability(self, tool_name: str) -> Dict[str, bool]:
        """Check if all dependencies for a tool are available."""
        dependencies = self.get_tool_dependencies(tool_name)
        availability = {}

        for dep in dependencies:
            availability[dep] = dep in self.tools

        return availability

    def enable_mock_mode(self) -> None:
        """Enable mock mode for all tools."""
        self.mock_mode = True
        # Re-register all tools with mock runners
        for tool_name, tool_spec in self.tools.items():
            if tool_name not in self.runners:
                self.runners[tool_name] = MockToolRunner(tool_spec)

    def disable_mock_mode(self) -> None:
        """Disable mock mode (requires real runners to be registered)."""
        self.mock_mode = False

    def load_tools_from_module(self, module_name: str) -> None:
        """Load tool specifications and runners from a Python module."""
        try:
            module = importlib.import_module(module_name)

            # Look for tool specifications
            for name, obj in inspect.getmembers(module):
                if isinstance(obj, ToolSpec):
                    self.register_tool(obj)

            # Look for tool runner classes
            for name, obj in inspect.getmembers(module):
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, ToolRunner)
                    and obj != ToolRunner
                ):
                    # Find corresponding tool spec
                    tool_name = getattr(obj, "tool_name", None)
                    if tool_name and tool_name in self.tools:
                        self.register_tool(self.tools[tool_name], obj)

        except ImportError as e:
            print(f"Warning: Could not load tools from module {module_name}: {e}")

    def get_registry_summary(self) -> Dict[str, Any]:
        """Get a summary of the tool registry."""
        categories = {}
        for tool_name, tool_spec in self.tools.items():
            category = tool_spec.category.value
            if category not in categories:
                categories[category] = []
            categories[category].append(tool_name)

        return {
            "total_tools": len(self.tools),
            "tools_with_runners": len(self.runners),
            "mock_mode": self.mock_mode,
            "categories": categories,
            "available_tools": list(self.tools.keys()),
        }


# Global registry instance
registry = ToolRegistry()

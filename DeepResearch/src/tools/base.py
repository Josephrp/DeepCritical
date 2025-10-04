from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Callable, Tuple


@dataclass
class ToolSpec:
    name: str
    description: str = ""
    inputs: Dict[str, str] = field(default_factory=dict)  # param: type
    outputs: Dict[str, str] = field(default_factory=dict)  # key: type


@dataclass
class ExecutionResult:
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class ToolRunner:
    spec: ToolSpec

    def __init__(self, spec: ToolSpec):
        self.spec = spec

    def validate(self, params: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        for k, t in self.spec.inputs.items():
            if k not in params:
                return False, f"Missing required param: {k}"
            # basic type gate (string types only for placeholder)
            if t.endswith("PATH") or t.endswith("ID") or t in {"TEXT", "AA SEQUENCE"}:
                if not isinstance(params[k], str):
                    return False, f"Invalid type for {k}: expected str for {t}"
        return True, None

    def run(self, params: Dict[str, Any]) -> ExecutionResult:
        raise NotImplementedError


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Callable[[], ToolRunner]] = {}

    def register(self, name: str, factory: Callable[[], ToolRunner]):
        self._tools[name] = factory

    def make(self, name: str) -> ToolRunner:
        if name not in self._tools:
            raise KeyError(f"Tool not found: {name}")
        return self._tools[name]()

    def list(self):
        return list(self._tools.keys())


registry = ToolRegistry()

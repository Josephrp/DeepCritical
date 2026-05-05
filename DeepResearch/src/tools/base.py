from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

from .registry import (
    CanonicalExecutionResult,
    CanonicalTool,
    CanonicalToolSpec,
    ToolCollisionError,
    ToolNotFoundError,
    canonical_registry,
)


@dataclass
class ToolSpec:
    name: str
    description: str = ""
    inputs: dict[str, str] = field(default_factory=dict)  # param: type
    outputs: dict[str, str] = field(default_factory=dict)  # key: type


@dataclass
class ExecutionResult:
    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class ToolRunner:
    spec: ToolSpec

    def __init__(self, spec: ToolSpec):
        self.spec = spec

    @staticmethod
    def _normalize_input_spec(spec_type: str) -> tuple[str, bool]:
        normalized = spec_type.split("(", 1)[0].strip()
        return normalized, "(optional" in spec_type.lower()

    def validate(self, params: dict[str, Any]) -> tuple[bool, str | None]:
        for k, t in self.spec.inputs.items():
            normalized_type, is_optional = self._normalize_input_spec(t)
            if k not in params:
                if is_optional:
                    continue
                return False, f"Missing required param: {k}"
            # basic type gate (string types only for placeholder)
            if normalized_type.endswith(("PATH", "ID")) or normalized_type in {
                "TEXT",
                "AA SEQUENCE",
            }:
                if not isinstance(params[k], str):
                    return (
                        False,
                        f"Invalid type for {k}: expected str for {normalized_type}",
                    )
        return True, None

    def run(self, params: dict[str, Any]) -> ExecutionResult:
        raise NotImplementedError


class _CanonicalToolAdapter(CanonicalTool):
    """Adapt legacy ToolRunner factories to the canonical tool interface."""

    def __init__(self, runner_factory: Callable[[], ToolRunner]):
        self._factory = runner_factory
        runner = runner_factory()
        self._runner = runner
        self.spec = CanonicalToolSpec(
            name=runner.spec.name,
            description=runner.spec.description,
            inputs=dict(runner.spec.inputs),
            outputs=dict(runner.spec.outputs),
        )

    def validate(self, params: dict[str, Any]) -> tuple[bool, str | None]:
        return self._runner.validate(params)

    def run(self, params: dict[str, Any]) -> CanonicalExecutionResult:
        res = self._runner.run(params)
        return CanonicalExecutionResult(
            success=res.success,
            data=dict(res.data),
            error=res.error,
            metrics=dict(res.metrics),
        )

    async def arun(self, params: dict[str, Any]) -> CanonicalExecutionResult:
        runner = self._runner
        runner_any: Any = runner
        if inspect.iscoroutinefunction(getattr(runner_any, "_arun", None)):
            res = await runner_any._arun(params)
        elif inspect.iscoroutinefunction(getattr(runner_any, "arun", None)):
            res = await runner_any.arun(params)
        else:
            res = await asyncio.to_thread(runner.run, params)
        return CanonicalExecutionResult(
            success=res.success,
            data=dict(res.data),
            error=res.error,
            metrics=dict(res.metrics),
        )


class ToolRegistry:
    """Backward-compatible registry facade over the canonical registry."""

    def register(self, name: str, factory: Callable[[], ToolRunner]):
        # Register into canonical registry; preserve legacy behavior by allowing
        # explicit override only when the caller passes override=True via canonical.
        canonical_registry.register(
            name,
            lambda: _CanonicalToolAdapter(factory),
            override=False,
        )

    def make(self, name: str) -> ToolRunner:
        try:
            canonical_tool = canonical_registry.make(name)
        except ToolNotFoundError as e:
            raise KeyError(str(e)) from e

        # Return the underlying ToolRunner when available.
        if isinstance(canonical_tool, _CanonicalToolAdapter):
            return canonical_tool._runner
        msg = f"Tool '{name}' is not a legacy ToolRunner"
        raise KeyError(msg)

    def list(self):
        return canonical_registry.list_tools()


registry = ToolRegistry()

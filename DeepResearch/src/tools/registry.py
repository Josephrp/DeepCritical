from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


class ToolRegistryError(Exception):
    pass


class ToolCollisionError(ToolRegistryError):
    pass


class ToolNotFoundError(ToolRegistryError):
    pass


@dataclass(frozen=True)
class CanonicalToolSpec:
    """Canonical tool specification used across DeepCritical.

    `inputs`/`outputs` are intentionally lightweight; validation is handled by tools.
    """

    name: str
    description: str = ""
    inputs: dict[str, str] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    # Optional semantic grouping, e.g. "web", "rag", "mcp", "sandbox"
    category: str | None = None
    # Tool names that should exist before this tool is considered available.
    dependencies: list[str] = field(default_factory=list)


@dataclass
class CanonicalExecutionResult:
    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class CanonicalTool(Protocol):
    spec: CanonicalToolSpec

    def validate(self, params: dict[str, Any]) -> tuple[bool, str | None]: ...

    def run(self, params: dict[str, Any]) -> CanonicalExecutionResult: ...

    async def arun(self, params: dict[str, Any]) -> CanonicalExecutionResult: ...


class ToolRegistry:
    """Canonical tool registry with collision detection and async support."""

    def __init__(self) -> None:
        # Canonical name -> factory
        self._tools: dict[str, Callable[[], CanonicalTool]] = {}
        # Alias name -> canonical name
        self._aliases: dict[str, str] = {}

    def register(
        self,
        name: str,
        factory: Callable[[], CanonicalTool],
        *,
        override: bool = False,
        aliases: list[str] | None = None,
    ) -> None:
        if (name in self._tools or name in self._aliases) and not override:
            raise ToolCollisionError(
                f"Tool name collision for '{name}'. Use override=True or rename."
            )
        self._tools[name] = factory
        for alias in aliases or []:
            if (alias in self._tools or alias in self._aliases) and not override:
                raise ToolCollisionError(
                    f"Alias collision for '{alias}' -> '{name}'. Use override=True."
                )
            self._aliases[alias] = name

    def alias(self, alias: str, canonical_name: str, *, override: bool = False) -> None:
        if canonical_name not in self._tools:
            raise ToolNotFoundError(f"Cannot alias to missing tool '{canonical_name}'.")
        if (alias in self._tools or alias in self._aliases) and not override:
            raise ToolCollisionError(
                f"Alias collision for '{alias}'. Use override=True."
            )
        self._aliases[alias] = canonical_name

    def resolve_name(self, name: str) -> str:
        return self._aliases.get(name, name)

    def list_tools(self) -> list[str]:
        return sorted(self._tools.keys())

    def list_aliases(self) -> dict[str, str]:
        return dict(self._aliases)

    def make(self, name: str) -> CanonicalTool:
        canonical = self.resolve_name(name)
        if canonical not in self._tools:
            raise ToolNotFoundError(f"Tool not found: {name}")
        return self._tools[canonical]()

    def _check_dependencies(self, tool: CanonicalTool) -> tuple[bool, str | None]:
        deps = getattr(tool.spec, "dependencies", []) if hasattr(tool, "spec") else []
        missing = [d for d in deps if self.resolve_name(d) not in self._tools]
        if missing:
            return False, f"Missing tool dependencies: {missing}"
        return True, None

    def execute(
        self,
        name: str,
        params: dict[str, Any],
        *,
        context: dict[str, Any] | None = None,
    ) -> CanonicalExecutionResult:
        start = time.time()
        tool = self.make(name)

        ok, dep_err = self._check_dependencies(tool)
        if not ok:
            return CanonicalExecutionResult(
                success=False,
                error=dep_err,
                metadata={
                    "tool": name,
                    "resolved": tool.spec.name,
                    "context": context or {},
                },
                metrics={"duration_s": time.time() - start},
            )

        validate = getattr(tool, "validate", None)
        if callable(validate):
            ok, err = validate(params)
            if not ok:
                return CanonicalExecutionResult(
                    success=False,
                    error=err,
                    metadata={
                        "tool": name,
                        "resolved": tool.spec.name,
                        "context": context or {},
                    },
                    metrics={"duration_s": time.time() - start},
                )

        try:
            result = tool.run(params)
        except Exception as e:
            result = CanonicalExecutionResult(success=False, error=str(e))

        result.metadata = {
            **(result.metadata or {}),
            "tool": name,
            "resolved": tool.spec.name,
            "context": context or {},
        }
        result.metrics = {**(result.metrics or {}), "duration_s": time.time() - start}
        return result

    async def aexecute(
        self,
        name: str,
        params: dict[str, Any],
        *,
        context: dict[str, Any] | None = None,
    ) -> CanonicalExecutionResult:
        start = time.time()
        tool = self.make(name)

        ok, dep_err = self._check_dependencies(tool)
        if not ok:
            return CanonicalExecutionResult(
                success=False,
                error=dep_err,
                metadata={
                    "tool": name,
                    "resolved": tool.spec.name,
                    "context": context or {},
                },
                metrics={"duration_s": time.time() - start},
            )

        validate = getattr(tool, "validate", None)
        if callable(validate):
            ok, err = validate(params)
            if not ok:
                return CanonicalExecutionResult(
                    success=False,
                    error=err,
                    metadata={
                        "tool": name,
                        "resolved": tool.spec.name,
                        "context": context or {},
                    },
                    metrics={"duration_s": time.time() - start},
                )

        try:
            arun = getattr(tool, "arun", None)
            if callable(arun):
                result = await arun(params)
            else:
                raise NotImplementedError
        except NotImplementedError:
            # Fall back to sync execution if tool does not implement async.
            result = await asyncio.to_thread(tool.run, params)
        except Exception as e:
            result = CanonicalExecutionResult(success=False, error=str(e))

        result.metadata = {
            **(result.metadata or {}),
            "tool": name,
            "resolved": tool.spec.name,
            "context": context or {},
        }
        result.metrics = {**(result.metrics or {}), "duration_s": time.time() - start}
        return result


# Canonical singleton used throughout the codebase.
canonical_registry = ToolRegistry()


def list_registered_tools() -> list[str]:
    """Registered canonical tool names."""
    from DeepResearch.src.tools import bootstrap_default_tools

    bootstrap_default_tools()
    return canonical_registry.list_tools()


def __getattr__(name: str) -> Any:
    # Expose ``registry.list()`` without assigning ``list = ...``, which would shadow
    # the builtin and break ``list[str]`` annotations in this module for type checkers.
    if name == "list":
        return list_registered_tools
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)

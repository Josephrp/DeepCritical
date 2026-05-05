"""
Pydantic AI tools data types for DeepCritical research workflows.

This module defines Pydantic AI specific tool runners and related data types
that integrate with the Pydantic AI framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Import registry locally to avoid circular imports
# from ..tools.base import registry  # Commented out to avoid circular imports
from DeepResearch.src.tools.base import ExecutionResult, ToolRunner, ToolSpec
from DeepResearch.src.utils.pydantic_ai_utils import build_agent as _build_agent
from DeepResearch.src.utils.pydantic_ai_utils import (
    build_builtin_tools as _build_builtin_tools,
)
from DeepResearch.src.utils.pydantic_ai_utils import build_toolsets as _build_toolsets

# Import utility functions from utils module
from DeepResearch.src.utils.pydantic_ai_utils import get_pydantic_ai_config as _get_cfg
from DeepResearch.src.utils.pydantic_ai_utils import run_agent_sync as _run_sync


class WebSearchBuiltinRunner:
    """Pydantic AI builtin web search wrapper."""

    def __init__(self):
        # Import base classes locally to avoid circular imports

        # Initialize spec for validation
        self.spec = ToolSpec(
            name="web_search",
            description="Pydantic AI builtin web search wrapper.",
            inputs={"query": "TEXT"},
            outputs={"results": "TEXT", "sources": "TEXT"},
        )

    def validate(self, params: dict[str, Any]) -> tuple[bool, str | None]:
        """Validate parameters."""
        for k, t in self.spec.inputs.items():
            if k not in params:
                return False, f"Missing required param: {k}"
            if t == "TEXT" and not isinstance(params[k], str):
                return False, f"Invalid type for {k}: expected str"
        return True, None

    def run(self, params: dict[str, Any]) -> ExecutionResult:
        ok, err = self.validate(params)
        if not ok:
            return ExecutionResult(success=False, error=err)

        q = str(params.get("query", "")).strip()
        if not q:
            return ExecutionResult(success=False, error="Empty query")

        cfg = _get_cfg()
        builtin_tools = _build_builtin_tools(cfg)
        if not any(
            getattr(t, "__class__", object).__name__ == "WebSearchTool"
            for t in builtin_tools
        ):
            # Force add WebSearchTool if not already on
            try:
                from pydantic_ai import WebSearchTool

                builtin_tools.append(WebSearchTool())
            except Exception:
                return ExecutionResult(success=False, error="pydantic_ai not available")

        toolsets = _build_toolsets(cfg)
        agent, _ = _build_agent(cfg, builtin_tools, toolsets)
        if agent is None:
            return ExecutionResult(
                success=False, error="pydantic_ai not available or misconfigured"
            )

        result = _run_sync(agent, q)
        if not result:
            return ExecutionResult(success=False, error="web search failed")

        text = getattr(result, "output", "")
        # Best-effort extract sources when provider supports it; keep as string
        sources = ""
        try:
            parts = getattr(result, "parts", None)
            if parts:
                sources = "\n".join(
                    [str(p) for p in parts if "web_search" in str(p).lower()]
                )
        except Exception:
            pass

        return ExecutionResult(success=True, data={"results": text, "sources": sources})


class CodeExecBuiltinRunner:
    """Pydantic AI builtin code execution wrapper."""

    def __init__(self):
        # Import base classes locally to avoid circular imports

        # Initialize spec for validation
        self.spec = ToolSpec(
            name="pyd_code_exec",
            description="Pydantic AI builtin code execution wrapper.",
            inputs={"code": "TEXT"},
            outputs={"output": "TEXT"},
        )

    def validate(self, params: dict[str, Any]) -> tuple[bool, str | None]:
        """Validate parameters."""
        for k, t in self.spec.inputs.items():
            if k not in params:
                return False, f"Missing required param: {k}"
            if t == "TEXT" and not isinstance(params[k], str):
                return False, f"Invalid type for {k}: expected str"
        return True, None

    def run(self, params: dict[str, Any]) -> ExecutionResult:
        ok, err = self.validate(params)
        if not ok:
            return ExecutionResult(success=False, error=err)

        code = str(params.get("code", "")).strip()
        if not code:
            return ExecutionResult(success=False, error="Empty code")

        cfg = _get_cfg()
        builtin_tools = _build_builtin_tools(cfg)
        # Ensure CodeExecutionTool present
        if not any(
            getattr(t, "__class__", object).__name__ == "CodeExecutionTool"
            for t in builtin_tools
        ):
            try:
                from pydantic_ai import CodeExecutionTool

                builtin_tools.append(CodeExecutionTool())
            except Exception:
                return ExecutionResult(success=False, error="pydantic_ai not available")

        toolsets = _build_toolsets(cfg)
        agent, _ = _build_agent(cfg, builtin_tools, toolsets)
        if agent is None:
            return ExecutionResult(
                success=False, error="pydantic_ai not available or misconfigured"
            )

        # Load system prompt from Hydra (if available)
        try:
            from DeepResearch.src.prompts import PromptLoader

            # In this wrapper, cfg may be empty; PromptLoader expects DictConfig-like object
            loader = PromptLoader(cfg)  # type: ignore
            system_prompt = loader.get("code_exec")
            prompt = (
                system_prompt.replace("${code}", code)
                if system_prompt
                else f"Execute the following code and return ONLY the final output as plain text.\n\n{code}"
            )
        except Exception:
            prompt = f"Execute the following code and return ONLY the final output as plain text.\n\n{code}"

        result = _run_sync(agent, prompt)
        if not result:
            return ExecutionResult(success=False, error="code execution failed")
        return ExecutionResult(
            success=True, data={"output": getattr(result, "output", "")}
        )


class UrlContextBuiltinRunner:
    """Pydantic AI builtin URL context wrapper."""

    def __init__(self):
        # Import base classes locally to avoid circular imports

        # Initialize spec for validation
        self.spec = ToolSpec(
            name="pyd_url_context",
            description="Pydantic AI builtin URL context wrapper.",
            inputs={"url": "TEXT"},
            outputs={"content": "TEXT"},
        )

    def validate(self, params: dict[str, Any]) -> tuple[bool, str | None]:
        """Validate parameters."""
        for k, t in self.spec.inputs.items():
            if k not in params:
                return False, f"Missing required param: {k}"
            if t == "TEXT" and not isinstance(params[k], str):
                return False, f"Invalid type for {k}: expected str"
        return True, None

    def run(self, params: dict[str, Any]) -> ExecutionResult:
        ok, err = self.validate(params)
        if not ok:
            return ExecutionResult(success=False, error=err)

        url = str(params.get("url", "")).strip()
        if not url:
            return ExecutionResult(success=False, error="Empty url")

        cfg = _get_cfg()
        builtin_tools = _build_builtin_tools(cfg)
        # Ensure UrlContextTool present
        if not any(
            getattr(t, "__class__", object).__name__ == "UrlContextTool"
            for t in builtin_tools
        ):
            try:
                from pydantic_ai import UrlContextTool

                builtin_tools.append(UrlContextTool())
            except Exception:
                return ExecutionResult(success=False, error="pydantic_ai not available")

        toolsets = _build_toolsets(cfg)
        agent, _ = _build_agent(cfg, builtin_tools, toolsets)
        if agent is None:
            return ExecutionResult(
                success=False, error="pydantic_ai not available or misconfigured"
            )

        prompt = (
            f"What is this? {url}\n\nExtract the main content or a concise summary."
        )
        result = _run_sync(agent, prompt)
        if not result:
            return ExecutionResult(success=False, error="url context failed")
        return ExecutionResult(
            success=True, data={"content": getattr(result, "output", "")}
        )


# Registry overrides and additions

# Registry registrations (commented out to avoid circular imports)
# registry.register(
#     "web_search", WebSearchBuiltinRunner
# )  # override previous synthetic runner
# registry.register("pyd_code_exec", CodeExecBuiltinRunner)
# registry.register("pyd_url_context", UrlContextBuiltinRunner)

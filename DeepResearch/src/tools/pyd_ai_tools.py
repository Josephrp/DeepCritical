from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .base import ToolSpec, ToolRunner, ExecutionResult, registry


def _get_cfg() -> Dict[str, Any]:
    try:
        # Lazy import Hydra/OmegaConf if available via app context; fall back to env-less defaults
        # In this lightweight wrapper, we don't have direct cfg access; return empty
        return {}
    except Exception:
        return {}


def _build_builtin_tools(cfg: Dict[str, Any]) -> List[Any]:
    try:
        # Import from Pydantic AI (exported at package root)
        from pydantic_ai import WebSearchTool, CodeExecutionTool, UrlContextTool
    except Exception:
        return []

    pyd_cfg = (cfg or {}).get("pyd_ai", {})
    builtin_cfg = pyd_cfg.get("builtin_tools", {})

    tools: List[Any] = []

    # Web Search
    ws_cfg = builtin_cfg.get("web_search", {})
    if ws_cfg.get("enabled", True):
        kwargs: Dict[str, Any] = {}
        if ws_cfg.get("search_context_size"):
            kwargs["search_context_size"] = ws_cfg.get("search_context_size")
        if ws_cfg.get("user_location"):
            kwargs["user_location"] = ws_cfg.get("user_location")
        if ws_cfg.get("blocked_domains"):
            kwargs["blocked_domains"] = ws_cfg.get("blocked_domains")
        if ws_cfg.get("allowed_domains"):
            kwargs["allowed_domains"] = ws_cfg.get("allowed_domains")
        if ws_cfg.get("max_uses") is not None:
            kwargs["max_uses"] = ws_cfg.get("max_uses")
        try:
            tools.append(WebSearchTool(**kwargs))
        except Exception:
            tools.append(WebSearchTool())

    # Code Execution
    ce_cfg = builtin_cfg.get("code_execution", {})
    if ce_cfg.get("enabled", False):
        try:
            tools.append(CodeExecutionTool())
        except Exception:
            pass

    # URL Context
    uc_cfg = builtin_cfg.get("url_context", {})
    if uc_cfg.get("enabled", False):
        try:
            tools.append(UrlContextTool())
        except Exception:
            pass

    return tools


def _build_toolsets(cfg: Dict[str, Any]) -> List[Any]:
    toolsets: List[Any] = []
    pyd_cfg = (cfg or {}).get("pyd_ai", {})
    ts_cfg = pyd_cfg.get("toolsets", {})

    # LangChain toolset (optional)
    lc_cfg = ts_cfg.get("langchain", {})
    if lc_cfg.get("enabled"):
        try:
            from pydantic_ai.ext.langchain import LangChainToolset

            # Expect user to provide instantiated tools or a toolkit provider name; here we do nothing dynamic
            tools = []  # placeholder if user later wires concrete LangChain tools
            toolsets.append(LangChainToolset(tools))
        except Exception:
            pass

    # ACI toolset (optional)
    aci_cfg = ts_cfg.get("aci", {})
    if aci_cfg.get("enabled"):
        try:
            from pydantic_ai.ext.aci import ACIToolset

            toolsets.append(
                ACIToolset(
                    aci_cfg.get("tools", []),
                    linked_account_owner_id=aci_cfg.get("linked_account_owner_id"),
                )
            )
        except Exception:
            pass

    return toolsets


def _build_agent(
    cfg: Dict[str, Any],
    builtin_tools: Optional[List[Any]] = None,
    toolsets: Optional[List[Any]] = None,
):
    try:
        from pydantic_ai import Agent
        from pydantic_ai.models.openai import OpenAIResponsesModelSettings
    except Exception:
        return None, None

    pyd_cfg = (cfg or {}).get("pyd_ai", {})
    model_name = pyd_cfg.get("model", "anthropic:claude-sonnet-4-0")

    settings = None
    # OpenAI Responses specific settings (include web search sources)
    if model_name.startswith("openai-responses:"):
        ws_include = (
            (pyd_cfg.get("builtin_tools", {}) or {}).get("web_search", {}) or {}
        ).get("openai_include_sources", False)
        try:
            settings = OpenAIResponsesModelSettings(
                openai_include_web_search_sources=bool(ws_include)
            )
        except Exception:
            settings = None

    agent = Agent(
        model_name,
        builtin_tools=builtin_tools or [],
        toolsets=toolsets or [],
        settings=settings,
    )

    return agent, pyd_cfg


def _run_sync(agent, prompt: str) -> Optional[Any]:
    try:
        return agent.run_sync(prompt)
    except Exception:
        return None


@dataclass
class WebSearchBuiltinRunner(ToolRunner):
    def __init__(self):
        super().__init__(
            ToolSpec(
                name="web_search",
                description="Pydantic AI builtin web search wrapper.",
                inputs={"query": "TEXT"},
                outputs={"results": "TEXT", "sources": "TEXT"},
            )
        )

    def run(self, params: Dict[str, Any]) -> ExecutionResult:
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


@dataclass
class CodeExecBuiltinRunner(ToolRunner):
    def __init__(self):
        super().__init__(
            ToolSpec(
                name="pyd_code_exec",
                description="Pydantic AI builtin code execution wrapper.",
                inputs={"code": "TEXT"},
                outputs={"output": "TEXT"},
            )
        )

    def run(self, params: Dict[str, Any]) -> ExecutionResult:
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
            from ..prompts import PromptLoader  # type: ignore

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


@dataclass
class UrlContextBuiltinRunner(ToolRunner):
    def __init__(self):
        super().__init__(
            ToolSpec(
                name="pyd_url_context",
                description="Pydantic AI builtin URL context wrapper.",
                inputs={"url": "TEXT"},
                outputs={"content": "TEXT"},
            )
        )

    def run(self, params: Dict[str, Any]) -> ExecutionResult:
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
registry.register(
    "web_search", WebSearchBuiltinRunner
)  # override previous synthetic runner
registry.register("pyd_code_exec", CodeExecBuiltinRunner)
registry.register("pyd_url_context", UrlContextBuiltinRunner)

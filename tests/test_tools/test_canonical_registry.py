import asyncio

import pytest

from DeepResearch.src.tools.registry import (
    CanonicalExecutionResult,
    CanonicalTool,
    CanonicalToolSpec,
    ToolCollisionError,
    ToolNotFoundError,
    ToolRegistry,
)


class _EchoTool:
    spec = CanonicalToolSpec(
        name="echo",
        description="Echo tool for testing",
        inputs={"text": "TEXT"},
        outputs={"text": "TEXT"},
    )

    def validate(self, params: dict):
        return True, None

    def run(self, params: dict):
        return CanonicalExecutionResult(
            success=True, data={"text": params.get("text", "")}
        )

    async def arun(self, params: dict):
        raise NotImplementedError


def test_register_and_execute_sync():
    reg = ToolRegistry()
    reg.register("echo", lambda: _EchoTool())

    res = reg.execute("echo", {"text": "hi"})
    assert res.success
    assert res.data["text"] == "hi"


def test_collision_detection():
    reg = ToolRegistry()
    reg.register("echo", lambda: _EchoTool())
    with pytest.raises(ToolCollisionError):
        reg.register("echo", lambda: _EchoTool(), override=False)


def test_alias_resolution():
    reg = ToolRegistry()
    reg.register("echo", lambda: _EchoTool())
    reg.alias("say", "echo")

    res = reg.execute("say", {"text": "yo"})
    assert res.success
    assert res.data["text"] == "yo"


@pytest.mark.asyncio
async def test_aexecute_sync_tool_fallback():
    reg = ToolRegistry()
    reg.register("echo", lambda: _EchoTool())

    res = await reg.aexecute("echo", {"text": "async"})
    assert res.success
    assert res.data["text"] == "async"


def test_not_found():
    reg = ToolRegistry()
    with pytest.raises(ToolNotFoundError):
        reg.execute("missing", {})

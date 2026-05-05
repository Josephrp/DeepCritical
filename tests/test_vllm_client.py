"""Tests for OpenAI-compatible vLLM client (simulate + HTTP paths)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from DeepResearch.src.datatypes.vllm_dataclass import (
    BatchRequest,
    ChatCompletionRequest,
    CompletionRequest,
    EmbeddingRequest,
)
from DeepResearch.src.utils.vllm_client import VLLMAgent, VLLMClient


@pytest.mark.asyncio
async def test_chat_completions_simulate() -> None:
    agent = VLLMAgent(VLLMClient(transport_mode="simulate"))
    resp = await agent.chat_completions(
        ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "hello"}],
        )
    )
    content = resp.choices[0].message.content or ""
    assert "[Simulated reply" in content


@pytest.mark.asyncio
async def test_chat_invokes_chat_completions_once() -> None:
    agent = VLLMAgent(VLLMClient(transport_mode="simulate"))
    called: list[int] = []
    orig = agent.chat_completions

    async def wrap(req: ChatCompletionRequest):
        called.append(1)
        return await orig(req)

    agent.chat_completions = wrap  # type: ignore[method-assign]
    text = await agent.chat([{"role": "user", "content": "hello"}], model="m")
    assert len(called) == 1
    assert "[Simulated reply" in text


@pytest.mark.asyncio
async def test_complete_invokes_completions_once() -> None:
    agent = VLLMAgent(VLLMClient(transport_mode="simulate"))
    called: list[int] = []
    orig = agent.completions

    async def wrap(req: CompletionRequest):
        called.append(1)
        return await orig(req)

    agent.completions = wrap  # type: ignore[method-assign]
    text = await agent.complete("hello there", model="m")
    assert len(called) == 1
    assert "[Simulated completion" in text


@pytest.mark.asyncio
async def test_batch_request_embedding_branch() -> None:
    client = VLLMClient(transport_mode="simulate")
    agent = VLLMAgent(client)
    batch = BatchRequest(
        requests=[
            ChatCompletionRequest(
                model="m", messages=[{"role": "user", "content": "a"}]
            ),
            CompletionRequest(model="m", prompt="b"),
            EmbeddingRequest(model="emb", input=["x"]),
        ]
    )
    out = await agent.batch_request(batch)
    assert out.processing_time >= 0.0
    assert out.total_requests == 3
    assert out.successful_requests == 3
    assert out.failed_requests == 0
    assert len(out.responses) == 3
    assert not out.errors


@pytest.mark.asyncio
async def test_vllm_client_delegates_to_agent() -> None:
    client = VLLMClient(transport_mode="simulate")
    resp = await client.chat_completions(
        ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "x"}],
        )
    )
    assert resp.choices[0].message.content
    assert "[Simulated reply" in (resp.choices[0].message.content or "")


@pytest.mark.asyncio
async def test_batch_records_errors_on_item_failure() -> None:
    client = VLLMClient(transport_mode="simulate")
    agent = VLLMAgent(client)
    batch = BatchRequest(
        requests=[
            ChatCompletionRequest(
                model="m", messages=[{"role": "user", "content": "a"}]
            ),
        ]
    )

    async def boom(_self: VLLMAgent, _req: ChatCompletionRequest):
        raise OSError("network down")

    with patch.object(VLLMAgent, "chat_completions", boom):
        out = await agent.batch_request(batch)
    assert out.failed_requests == 1
    assert out.successful_requests == 0
    assert out.errors


@pytest.mark.asyncio
async def test_http_chat_completion_parses_response() -> None:
    client = VLLMClient(
        base_url="http://127.0.0.1:9",
        transport_mode="http",
        timeout=1.0,
        max_retries=0,
    )
    agent = VLLMAgent(client)
    fake_payload = {
        "choices": [{"message": {"content": "from-server"}}],
    }

    with patch.object(VLLMAgent, "_http_request", new_callable=AsyncMock) as m:
        m.return_value = fake_payload
        req = ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
        )
        resp = await agent.chat_completions(req)
        assert resp.choices[0].message.content == "from-server"
        m.assert_awaited()

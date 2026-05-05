"""
Comprehensive VLLM client with OpenAI API compatibility for Pydantic AI agents.

This module provides a complete VLLM client that can be used as a custom agent
in Pydantic AI, supporting all VLLM features while maintaining OpenAI API compatibility.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING, Any, Literal, cast

import aiohttp
from pydantic import BaseModel, ConfigDict, Field

from DeepResearch.src.datatypes.vllm_dataclass import (
    BatchRequest,
    BatchResponse,
    CacheConfig,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    DeviceConfig,
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    ModelConfig,
    ObservabilityConfig,
    ParallelConfig,
    QuantizationMethod,
    SchedulerConfig,
    UsageStats,
    VllmConfig,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = logging.getLogger(__name__)


def _kwargs_for_model(
    model_cls: type[BaseModel], kwargs: dict[str, Any]
) -> dict[str, Any]:
    fields = model_cls.model_fields
    return {k: v for k, v in kwargs.items() if k in fields}


class VLLMClientError(Exception):
    """Base exception for VLLM client errors."""


class VLLMConnectionError(VLLMClientError):
    """Connection-related errors."""


class VLLMAPIError(VLLMClientError):
    """API-related errors."""


class VLLMClient(BaseModel):
    """Configuration holder and entrypoint for OpenAI-compatible vLLM HTTP calls.

    Use :class:`VLLMAgent` for execution. Methods on this model delegate to
    ``VLLMAgent(self)`` so a single implementation path handles simulate vs HTTP.
    """

    base_url: str = Field("http://localhost:8000", description="VLLM server base URL")
    api_key: str | None = Field(None, description="API key for authentication")
    timeout: float = Field(60.0, description="Request timeout in seconds")
    max_retries: int = Field(3, description="Maximum number of retries")
    retry_delay: float = Field(1.0, description="Delay between retries in seconds")
    transport_mode: Literal["simulate", "http"] = Field(
        "simulate",
        description=(
            "simulate: deterministic local responses (CI-safe); "
            "http: OpenAI-compatible POSTs to {base_url}/v1/..."
        ),
    )

    # VLLM-specific configuration
    vllm_config: VllmConfig | None = Field(None, description="VLLM configuration")

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_schema_extra={
            "example": {
                "base_url": "http://localhost:8000",
                "api_key": None,
                "timeout": 60.0,
                "max_retries": 3,
                "retry_delay": 1.0,
                "transport_mode": "simulate",
            }
        },
    )

    async def chat_completions(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """OpenAI-compatible chat completions (delegates to :class:`VLLMAgent`)."""
        return await VLLMAgent(self).chat_completions(request)

    async def completions(self, request: CompletionRequest) -> CompletionResponse:
        """OpenAI-compatible completions (delegates to :class:`VLLMAgent`)."""
        return await VLLMAgent(self).completions(request)

    async def embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """OpenAI-compatible embeddings (delegates to :class:`VLLMAgent`)."""
        return await VLLMAgent(self).embeddings(request)


class VLLMAgent:
    """OpenAI-compatible vLLM client (simulate or HTTP)."""

    def __init__(self, vllm_client: VLLMClient):
        self.client = vllm_client

    def _v1_base(self) -> str:
        u = self.client.base_url.rstrip("/")
        if u.endswith("/v1"):
            return u
        return f"{u}/v1"

    def _use_http(self) -> bool:
        return self.client.transport_mode == "http"

    async def _http_request(
        self, method: str, path: str, json_body: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        url = f"{self._v1_base()}{path}"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.client.api_key:
            key = self.client.api_key.strip()
            if key and key.upper() != "EMPTY":
                headers["Authorization"] = f"Bearer {key}"
        timeout = aiohttp.ClientTimeout(total=self.client.timeout)
        last_err: BaseException | None = None
        for attempt in range(self.client.max_retries + 1):
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.request(
                        method, url, json=json_body, headers=headers
                    ) as resp:
                        text = await resp.text()
                        if resp.status >= 400:
                            raise VLLMAPIError(f"HTTP {resp.status}: {text[:500]}")
                        return json.loads(text)
            except (VLLMAPIError, aiohttp.ClientError, json.JSONDecodeError) as e:
                last_err = e
                if attempt < self.client.max_retries:
                    await asyncio.sleep(self.client.retry_delay)
                else:
                    raise VLLMConnectionError(str(e)) from e
        raise VLLMConnectionError(str(last_err)) from last_err

    async def _simulate_chat_reply(self, messages: list[dict[str, str]]) -> str:
        last = messages[-1].get("content", "") if messages else ""
        return f"[Simulated reply to: {last[:120]}]"

    async def _simulate_completion(self, prompt: str) -> str:
        return f"[Simulated completion: {prompt[:120]}]"

    async def _simulate_embeddings(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 384 for _ in texts]

    async def _chat_text_from_http(self, request: ChatCompletionRequest) -> str:
        payload = request.model_dump(exclude_none=True)
        payload["stream"] = False
        data = await self._http_request("POST", "/chat/completions", payload)
        try:
            choices = data.get("choices") or []
            msg = (choices[0].get("message") or {}) if choices else {}
            content = msg.get("content")
            return content if isinstance(content, str) else ""
        except (IndexError, KeyError, TypeError) as e:
            raise VLLMAPIError(f"Unexpected chat completion response: {data!r}") from e

    async def _completion_text_from_http(self, request: CompletionRequest) -> str:
        payload = request.model_dump(exclude_none=True)
        payload["stream"] = False
        data = await self._http_request("POST", "/completions", payload)
        try:
            choices = data.get("choices") or []
            text = choices[0].get("text") if choices else None
            return text if isinstance(text, str) else ""
        except (IndexError, KeyError, TypeError) as e:
            raise VLLMAPIError(f"Unexpected completion response: {data!r}") from e

    async def _embeddings_from_http(
        self, request: EmbeddingRequest
    ) -> list[list[float]]:
        payload = request.model_dump(exclude_none=True)
        data = await self._http_request("POST", "/embeddings", payload)
        try:
            out: list[list[float]] = []
            for i, item in enumerate(data.get("data") or []):
                emb = item.get("embedding")
                if isinstance(emb, list):
                    out.append([float(x) for x in emb])
                else:
                    raise VLLMAPIError(f"Missing embedding at index {i}")
            return out
        except VLLMAPIError:
            raise
        except (KeyError, TypeError, ValueError) as e:
            raise VLLMAPIError(f"Unexpected embeddings response: {data!r}") from e

    async def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Chat with the VLLM model (builds a request; no recursion)."""
        kw = dict(kwargs)
        model = cast("str", kw.pop("model", None) or "vllm-model")
        kw.pop("messages", None)
        req = ChatCompletionRequest(
            model=model,
            messages=messages,
            **_kwargs_for_model(ChatCompletionRequest, kw),
        )
        response = await self.chat_completions(req)
        content = response.choices[0].message.content
        return content if content is not None else ""

    async def complete(self, prompt: str | list[str], **kwargs: Any) -> str:
        """Complete text with the VLLM model (builds a request; no recursion)."""
        kw = dict(kwargs)
        model = cast("str", kw.pop("model", None) or "vllm-model")
        kw.pop("prompt", None)
        req = CompletionRequest(
            model=model, prompt=prompt, **_kwargs_for_model(CompletionRequest, kw)
        )
        response = await self.completions(req)
        return response.choices[0].text

    async def embed(self, texts: str | list[str], **kwargs: Any) -> list[list[float]]:
        """Generate embeddings for texts (builds a request; no recursion)."""
        inp: list[str] = [texts] if isinstance(texts, str) else list(texts)
        kw = dict(kwargs)
        model = cast("str", kw.pop("model", None) or "vllm-embedding-model")
        kw.pop("input", None)
        req = EmbeddingRequest(
            model=model, input=inp, **_kwargs_for_model(EmbeddingRequest, kw)
        )
        response = await self.embeddings(req)
        return [item.embedding for item in response.data]

    def to_pydantic_ai_agent(self, model_name: str = "vllm-agent"):
        """Convert to Pydantic AI agent format."""
        from pydantic_ai import Agent, RunContext

        agent = Agent(
            model_name,
            deps_type=VLLMAgent,
            system_prompt="You are a helpful AI assistant powered by VLLM.",
        )

        @agent.tool
        async def chat_completion(
            ctx: RunContext[VLLMAgent],
            messages: list[dict[str, str]],
            **kwargs: Any,
        ) -> str:
            """Chat completion using VLLM."""
            return await ctx.deps.chat(messages, **kwargs)

        @agent.tool
        async def text_completion(
            ctx: RunContext[VLLMAgent], prompt: str, **kwargs: Any
        ) -> str:
            """Text completion using VLLM."""
            return await ctx.deps.complete(prompt, **kwargs)

        @agent.tool
        async def generate_embeddings(
            ctx: RunContext[VLLMAgent],
            texts: str | list[str],
            **kwargs: Any,
        ) -> list[list[float]]:
            """Generate embeddings using VLLM."""
            return await ctx.deps.embed(texts, **kwargs)

        return agent

    async def health(self) -> dict[str, Any]:
        """Check server health (OpenAI-compatible)."""
        try:
            models = await self.models()
            return {"status": "healthy", "models": len(models.get("data", []))}
        except Exception:
            return {"status": "unhealthy"}

    async def models(self) -> dict[str, Any]:
        """List available models (OpenAI-compatible)."""
        if not self._use_http():
            return {"object": "list", "data": [{"id": "vllm-model", "object": "model"}]}
        return await self._http_request("GET", "/models", None)

    async def chat_completions(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Create chat completion (OpenAI-compatible)."""
        messages = [
            {"role": msg["role"], "content": msg["content"]} for msg in request.messages
        ]
        if self._use_http():
            response_text = await self._chat_text_from_http(request)
        else:
            response_text = await self._simulate_chat_reply(messages)
        return ChatCompletionResponse(
            id=f"chatcmpl-{time.time()}",
            object="chat.completion",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_text),
                    finish_reason="stop",
                )
            ],
            usage=UsageStats(
                prompt_tokens=len(request.messages),
                completion_tokens=len(response_text.split()),
                total_tokens=len(request.messages) + len(response_text.split()),
            ),
        )

    async def chat_completions_stream(
        self, request: ChatCompletionRequest
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream chat completion (OpenAI-compatible)."""
        response = await self.chat_completions(request)
        choice = response.choices[0]
        yield {
            "id": response.id,
            "object": "chat.completion.chunk",
            "created": response.created,
            "model": response.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": choice.message.content or ""},
                    "finish_reason": choice.finish_reason,
                }
            ],
        }

    async def completions(self, request: CompletionRequest) -> CompletionResponse:
        """Create completion (OpenAI-compatible)."""
        prompt_text = (
            request.prompt
            if isinstance(request.prompt, str)
            else "\n".join(request.prompt)
        )
        if self._use_http():
            response_text = await self._completion_text_from_http(request)
        else:
            response_text = await self._simulate_completion(prompt_text)
        return CompletionResponse(
            id=f"cmpl-{time.time()}",
            object="text_completion",
            created=int(time.time()),
            model=request.model,
            choices=[
                CompletionChoice(text=response_text, index=0, finish_reason="stop")
            ],
            usage=UsageStats(
                prompt_tokens=len(prompt_text.split()),
                completion_tokens=len(response_text.split()),
                total_tokens=len(prompt_text.split()) + len(response_text.split()),
            ),
        )

    async def embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Create embeddings (OpenAI-compatible)."""
        texts = (
            [request.input] if isinstance(request.input, str) else list(request.input)
        )
        if self._use_http():
            vectors = await self._embeddings_from_http(request)
        else:
            vectors = await self._simulate_embeddings(texts)
        return EmbeddingResponse(
            object="list",
            data=[
                EmbeddingData(object="embedding", embedding=emb, index=i)
                for i, emb in enumerate(vectors)
            ],
            model=request.model,
            usage=UsageStats(
                prompt_tokens=len(str(request.input).split()),
                completion_tokens=0,
                total_tokens=len(str(request.input).split()),
            ),
        )

    async def batch_request(self, request: BatchRequest) -> BatchResponse:
        """Process batch request."""
        started = time.perf_counter()
        results: list[
            ChatCompletionResponse | CompletionResponse | EmbeddingResponse
        ] = []
        errors: list[dict[str, Any]] = []
        total = len(list(request.requests))

        for index, req in enumerate(request.requests):
            try:
                if isinstance(req, ChatCompletionRequest):
                    results.append(await self.chat_completions(req))
                elif isinstance(req, CompletionRequest):
                    results.append(await self.completions(req))
                elif isinstance(req, EmbeddingRequest):
                    results.append(await self.embeddings(req))
                else:
                    errors.append(
                        {
                            "request_index": index,
                            "error": f"Unsupported request type: {type(req)!r}",
                        }
                    )
            except (VLLMClientError, aiohttp.ClientError, OSError) as e:
                logger.warning("batch item %s failed: %s", index, e)
                errors.append({"request_index": index, "error": str(e)})

        elapsed = time.perf_counter() - started
        return BatchResponse(
            batch_id=f"batch-{time.time()}",
            responses=results,
            errors=errors,
            total_requests=total,
            successful_requests=len(results),
            failed_requests=len(errors),
            processing_time=elapsed,
        )

    async def close(self) -> None:
        """Close client connections."""
        # aiohttp sessions are per-request; nothing to close


class VLLMClientBuilder:
    """Builder for creating VLLM clients with complex configurations."""

    def __init__(self):
        self._config = {
            "base_url": "http://localhost:8000",
            "timeout": 60.0,
            "max_retries": 3,
            "retry_delay": 1.0,
            "transport_mode": "simulate",
        }
        self._vllm_config = None

    def with_base_url(self, base_url: str) -> VLLMClientBuilder:
        """Set base URL."""
        self._config["base_url"] = base_url
        return self

    def with_api_key(self, api_key: str) -> VLLMClientBuilder:
        """Set API key."""
        self._config["api_key"] = api_key
        return self

    def with_timeout(self, timeout: float) -> VLLMClientBuilder:
        """Set timeout."""
        self._config["timeout"] = timeout
        return self

    def with_retries(
        self, max_retries: int, retry_delay: float = 1.0
    ) -> VLLMClientBuilder:
        """Set retry configuration."""
        self._config["max_retries"] = max_retries
        self._config["retry_delay"] = retry_delay
        return self

    def with_transport_mode(
        self, mode: Literal["simulate", "http"]
    ) -> VLLMClientBuilder:
        """Use simulate (default) or HTTP OpenAI-compatible API calls."""
        self._config["transport_mode"] = mode
        return self

    def with_vllm_config(self, config: VllmConfig) -> VLLMClientBuilder:
        """Set VLLM configuration."""
        self._vllm_config = config
        return self

    def with_model_config(
        self,
        model: str,
        tokenizer: str | None = None,
        trust_remote_code: bool = False,
        max_model_len: int | None = None,
        quantization: QuantizationMethod | None = None,
    ) -> VLLMClientBuilder:
        """Configure model settings."""
        if self._vllm_config is None:
            self._vllm_config = VllmConfig(
                model=ModelConfig(
                    model=model,
                    tokenizer=tokenizer,
                    trust_remote_code=trust_remote_code,
                    max_model_len=max_model_len,
                    quantization=quantization,
                ),
                cache=CacheConfig(),
                parallel=ParallelConfig(),
                scheduler=SchedulerConfig(),
                device=DeviceConfig(),
                observability=ObservabilityConfig(),
            )
        else:
            self._vllm_config.model = ModelConfig(
                model=model,
                tokenizer=tokenizer,
                trust_remote_code=trust_remote_code,
                max_model_len=max_model_len,
                quantization=quantization,
            )
        return self

    def with_cache_config(
        self,
        block_size: int = 16,
        gpu_memory_utilization: float = 0.9,
        swap_space: int = 4,
    ) -> VLLMClientBuilder:
        """Configure cache settings."""
        if self._vllm_config is None:
            self._vllm_config = VllmConfig(
                model=ModelConfig(model="default"),
                cache=CacheConfig(
                    block_size=block_size,
                    gpu_memory_utilization=gpu_memory_utilization,
                    swap_space=swap_space,
                ),
                parallel=ParallelConfig(),
                scheduler=SchedulerConfig(),
                device=DeviceConfig(),
                observability=ObservabilityConfig(),
            )
        else:
            self._vllm_config.cache = CacheConfig(
                block_size=block_size,
                gpu_memory_utilization=gpu_memory_utilization,
                swap_space=swap_space,
            )
        return self

    def with_parallel_config(
        self,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
    ) -> VLLMClientBuilder:
        """Configure parallel settings."""
        if self._vllm_config is None:
            self._vllm_config = VllmConfig(
                model=ModelConfig(model="default"),
                cache=CacheConfig(),
                parallel=ParallelConfig(
                    tensor_parallel_size=tensor_parallel_size,
                    pipeline_parallel_size=pipeline_parallel_size,
                ),
                scheduler=SchedulerConfig(),
                device=DeviceConfig(),
                observability=ObservabilityConfig(),
            )
        else:
            self._vllm_config.parallel = ParallelConfig(
                tensor_parallel_size=tensor_parallel_size,
                pipeline_parallel_size=pipeline_parallel_size,
            )
        return self

    def build(self) -> VLLMClient:
        """Build the VLLM client."""
        tm = self._config.get("transport_mode", "simulate")
        mode = cast("Literal['simulate', 'http']", tm)
        return VLLMClient(
            base_url=str(self._config.get("base_url", "http://localhost:8000")),
            api_key=cast("str | None", self._config.get("api_key")),
            timeout=float(self._config.get("timeout", 60.0)),
            max_retries=int(self._config.get("max_retries", 3)),
            retry_delay=float(self._config.get("retry_delay", 1.0)),
            transport_mode=mode,
            vllm_config=self._vllm_config,
        )


# ============================================================================
# Utility Functions
# ============================================================================


def create_vllm_client(
    model_name: str,
    base_url: str = "http://localhost:8000",
    api_key: str | None = None,
    **kwargs,
) -> VLLMClient:
    """Create a VLLM client with sensible defaults."""
    builder = (
        VLLMClientBuilder().with_base_url(base_url).with_model_config(model=model_name)
    )
    if api_key is not None:
        builder = builder.with_api_key(api_key)
    return builder.build()


async def test_vllm_connection(client: VLLMClient) -> bool:
    """Test if VLLM server is accessible."""
    agent = VLLMAgent(client)
    try:
        await agent.health()
        return True
    except Exception:
        return False


async def list_vllm_models(client: VLLMClient) -> list[str]:
    """List available models on the VLLM server."""
    try:
        agent = VLLMAgent(client)
        response = await agent.models()
        data = response.get("data", [])
        return [
            str(m["id"])
            for m in data
            if isinstance(m, dict) and m.get("id") is not None
        ]
    except Exception:
        return []


# ============================================================================
# Example Usage and Factory Functions
# ============================================================================


async def example_basic_usage():
    """Example of basic VLLM client usage."""
    client = create_vllm_client("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    agent = VLLMAgent(client)

    # Test connection
    if await test_vllm_connection(client):
        # List models
        await list_vllm_models(client)

        # Chat completion
        chat_request = ChatCompletionRequest(
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            messages=[{"role": "user", "content": "Hello, how are you?"}],
            max_tokens=50,
            temperature=0.7,
        )

        await agent.chat_completions(chat_request)

    await agent.close()


async def example_streaming():
    """Example of streaming usage."""
    client = create_vllm_client("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    agent = VLLMAgent(client)

    chat_request = ChatCompletionRequest(
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        messages=[{"role": "user", "content": "Tell me a story"}],
        max_tokens=100,
        temperature=0.8,
        stream=True,
    )

    async for _chunk in agent.chat_completions_stream(chat_request):
        pass

    await agent.close()


async def example_embeddings():
    """Example of embedding usage."""
    client = create_vllm_client("sentence-transformers/all-MiniLM-L6-v2")
    agent = VLLMAgent(client)

    embedding_request = EmbeddingRequest(
        model="sentence-transformers/all-MiniLM-L6-v2",
        input=["Hello world", "How are you?"],
    )

    await agent.embeddings(embedding_request)

    await agent.close()


async def example_batch_processing():
    """Example of batch processing."""
    client = create_vllm_client("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    agent = VLLMAgent(client)

    requests = [
        ChatCompletionRequest(
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            messages=[{"role": "user", "content": f"Question {i}"}],
            max_tokens=20,
        )
        for i in range(3)
    ]

    batch_request = BatchRequest(requests=requests, max_retries=2)
    await agent.batch_request(batch_request)

    await agent.close()


if __name__ == "__main__":
    # Run examples

    # Basic usage
    asyncio.run(example_basic_usage())

    # Streaming
    asyncio.run(example_streaming())

    # Embeddings
    asyncio.run(example_embeddings())

    # Batch processing
    asyncio.run(example_batch_processing())

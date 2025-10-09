"""
Custom Pydantic AI model implementations for DeepCritical.

This module provides Pydantic AI model wrappers for:
- vLLM (production-grade local LLM inference)
- llama.cpp (lightweight local inference)
- OpenAI-compatible servers (generic wrapper)

Usage:
    ```python
    from pydantic_ai import Agent
    from DeepResearch.src.models import VLLMModel, LlamaCppModel

    # vLLM
    vllm_model = VLLMModel.from_vllm(
        base_url="http://localhost:8000/v1",
        model_name="meta-llama/Llama-3-8B"
    )
    agent = Agent(vllm_model)

    # llama.cpp
    llamacpp_model = LlamaCppModel.from_llamacpp(
        base_url="http://localhost:8080/v1",
        model_name="llama-3-8b.gguf"
    )
    agent = Agent(llamacpp_model)
    ```
"""

from .openai_compatible_model import (
    LlamaCppModel,
    OpenAICompatibleModel,
    VLLMModel,
)

__all__ = [
    "LlamaCppModel",
    "OpenAICompatibleModel",
    "VLLMModel",
]

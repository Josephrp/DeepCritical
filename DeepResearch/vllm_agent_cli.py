#!/usr/bin/env python3
"""
VLLM Agent CLI for Pydantic AI.

This script demonstrates how to use the VLLM client with Pydantic AI's CLI system.
It can be used as a custom agent with `clai --agent vllm_agent_cli:vllm_agent`.

Usage:
    # Install as a custom agent
    clai --agent vllm_agent_cli:vllm_agent "Hello, how are you?"

    # Or run directly
    python vllm_agent_cli.py
"""

from __future__ import annotations

import asyncio
import argparse
from typing import Optional

from src.agents.vllm_agent import VLLMAgent, VLLMAgentConfig


class VLLMAgentCLI:
    """CLI wrapper for VLLM agent."""

    def __init__(
        self,
        model_name: str = "microsoft/DialoGPT-medium",
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        embedding_model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        **kwargs,
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.embedding_model = embedding_model

        # Create VLLM agent configuration
        self.agent_config = VLLMAgentConfig(
            client_config={
                "base_url": base_url,
                "api_key": api_key,
                "timeout": 60.0,
                **kwargs,
            },
            default_model=model_name,
            embedding_model=embedding_model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt="You are a helpful AI assistant powered by VLLM. You can perform various tasks including text generation, conversation, and analysis.",
        )

        self.agent: Optional[VLLMAgent] = None
        self.pydantic_agent = None

    async def initialize(self):
        """Initialize the VLLM agent."""
        print(f"Initializing VLLM agent with model: {self.model_name}")
        print(f"Server: {self.base_url}")

        # Create and initialize agent
        self.agent = VLLMAgent(self.agent_config)
        await self.agent.initialize()

        # Convert to Pydantic AI agent
        self.pydantic_agent = self.agent.to_pydantic_ai_agent()

        print("✓ VLLM agent initialized successfully")

    async def run_interactive(self):
        """Run interactive chat session."""
        if not self.agent:
            await self.initialize()

        print("\n🤖 VLLM Agent Interactive Session")
        print("Type 'quit' or 'exit' to end the session")
        print("Type 'stream' to toggle streaming mode")
        print("-" * 50)

        streaming = False

        while True:
            try:
                user_input = input("\nYou: ").strip()

                if user_input.lower() in ["quit", "exit", "q"]:
                    print("Goodbye! 👋")
                    break

                if user_input.lower() == "stream":
                    streaming = not streaming
                    mode = "enabled" if streaming else "disabled"
                    print(f"Streaming mode {mode}")
                    continue

                if not user_input:
                    continue

                # Prepare messages
                messages = [{"role": "user", "content": user_input}]

                if streaming:
                    print("Assistant: ", end="", flush=True)
                    response = await self.agent.chat_stream(messages)
                    print()  # New line after streaming
                else:
                    response = await self.agent.chat(messages)
                    print(f"Assistant: {response}")

            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"Error: {e}")

    async def run_single_query(self, query: str, stream: bool = False):
        """Run a single query."""
        if not self.agent:
            await self.initialize()

        messages = [{"role": "user", "content": query}]

        if stream:
            print("Assistant: ", end="", flush=True)
            response = await self.agent.chat_stream(messages)
            print()
        else:
            response = await self.agent.chat(messages)
            print(f"Assistant: {response}")

        return response

    async def run_completion(self, prompt: str):
        """Run text completion."""
        if not self.agent:
            await self.initialize()

        response = await self.agent.complete(prompt)
        print(f"Completion: {response}")
        return response

    async def run_embeddings(self, texts: list):
        """Generate embeddings."""
        if not self.agent:
            await self.initialize()

        if self.agent.config.embedding_model:
            embeddings = await self.agent.embed(texts)
            print(f"Generated {len(embeddings)} embeddings")
            for i, emb in enumerate(embeddings):
                print(f"Text {i + 1}: {len(emb)}-dimensional embedding")
        else:
            print("No embedding model configured")

    async def list_models(self):
        """List available models."""
        if not self.agent:
            await self.initialize()

        models = await self.agent.client.models()
        print("Available models:")
        for model in models.data:
            print(f"  - {model.id}")
        return models.data

    async def health_check(self):
        """Check server health."""
        if not self.agent:
            await self.initialize()

        health = await self.agent.client.health()
        print(f"Server status: {health.status}")
        print(f"Uptime: {health.uptime:.1f}s")
        print(f"Version: {health.version}")
        return health


# Global agent instance for CLI usage
_vllm_agent: Optional[VLLMAgentCLI] = None


def get_vllm_agent() -> VLLMAgentCLI:
    """Get or create the global VLLM agent instance."""
    global _vllm_agent
    if _vllm_agent is None:
        _vllm_agent = VLLMAgentCLI()
    return _vllm_agent


# Pydantic AI agent instance for CLI integration
async def create_pydantic_ai_agent():
    """Create the Pydantic AI agent instance."""
    agent_cli = get_vllm_agent()
    await agent_cli.initialize()
    return agent_cli.pydantic_agent


# ============================================================================
# CLI Interface Functions
# ============================================================================


async def chat_with_vllm(
    messages: list,
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 512,
    **kwargs,
) -> str:
    """Chat completion function for Pydantic AI."""
    agent = get_vllm_agent()

    # Override config if provided
    if model and model != agent.model_name:
        agent.model_name = model
        await agent.initialize()  # Reinitialize with new model

    return await agent.agent.chat(messages, **kwargs)


async def complete_with_vllm(
    prompt: str,
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 512,
    **kwargs,
) -> str:
    """Text completion function for Pydantic AI."""
    agent = get_vllm_agent()

    if model and model != agent.model_name:
        agent.model_name = model
        await agent.initialize()

    return await agent.agent.complete(prompt, **kwargs)


async def embed_with_vllm(texts, model: Optional[str] = None, **kwargs) -> list:
    """Embedding generation function for Pydantic AI."""
    agent = get_vllm_agent()

    if model and model != agent.model_name:
        agent.model_name = model
        await agent.initialize()

    return await agent.agent.embed(texts, **kwargs)


# ============================================================================
# Main CLI Entry Point
# ============================================================================


async def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="VLLM Agent CLI")
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/DialoGPT-medium",
        help="Model name to use",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000",
        help="VLLM server base URL",
    )
    parser.add_argument("--api-key", type=str, help="API key for authentication")
    parser.add_argument("--embedding-model", type=str, help="Embedding model name")
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=512, help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--query", type=str, help="Single query to run (non-interactive mode)"
    )
    parser.add_argument("--completion", type=str, help="Text completion prompt")
    parser.add_argument(
        "--embeddings", nargs="+", help="Generate embeddings for these texts"
    )
    parser.add_argument(
        "--list-models", action="store_true", help="List available models"
    )
    parser.add_argument(
        "--health-check", action="store_true", help="Check server health"
    )
    parser.add_argument("--stream", action="store_true", help="Enable streaming output")

    args = parser.parse_args()

    # Create agent
    agent = VLLMAgentCLI(
        model_name=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        embedding_model=args.embedding_model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    try:
        if args.list_models:
            await agent.list_models()
        elif args.health_check:
            await agent.health_check()
        elif args.embeddings:
            await agent.run_embeddings(args.embeddings)
        elif args.completion:
            await agent.run_completion(args.completion)
        elif args.query:
            await agent.run_single_query(args.query, stream=args.stream)
        else:
            # Interactive mode
            await agent.run_interactive()

    except KeyboardInterrupt:
        print("\nGoodbye! 👋")
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    import sys

    result = asyncio.run(main())
    sys.exit(result)

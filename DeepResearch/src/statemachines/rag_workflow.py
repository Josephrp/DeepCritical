"""
RAG (Retrieval-Augmented Generation) workflow state machine for DeepCritical.

This module implements a Pydantic Graph-based workflow for RAG operations,
including document ingestion, vector storage, retrieval, and generation.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any, cast
from urllib.parse import urlparse

from pydantic_graph import BaseNode, Edge, End, Graph, GraphRunContext

from DeepResearch.src.datatypes.rag import (
    Document,
    RAGConfig,
    RAGQuery,
    RAGResponse,
    SearchType,
)
from DeepResearch.src.datatypes.vllm_integration import (
    VLLMDeployment,
    VLLMEmbeddings,
    VLLMLLMProvider,
)
from DeepResearch.src.utils.execution_status import ExecutionStatus
from DeepResearch.src.vector_stores import InMemoryVectorStore, create_vector_store

if TYPE_CHECKING:
    from omegaconf import DictConfig


@dataclass
class RAGState:
    """State for RAG workflow execution."""

    question: str
    rag_config: RAGConfig | None = None
    documents: list[Document] = field(default_factory=list)
    rag_response: RAGResponse | None = None
    rag_result: dict[str, Any] | None = None  # For agent results
    processing_steps: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    config: DictConfig | None = None
    execution_status: ExecutionStatus = ExecutionStatus.PENDING


# --- RAG Workflow Nodes ---


@dataclass
class InitializeRAG(BaseNode[RAGState, None, str]):
    """Initialize RAG system with configuration."""

    async def run(self, ctx: GraphRunContext[RAGState]) -> LoadDocuments | RAGError:
        """Initialize RAG system components."""
        try:
            cfg = ctx.state.config
            rag_cfg = getattr(cfg, "rag", {})

            # Create RAG configuration from Hydra config
            rag_config = self._create_rag_config(rag_cfg, cfg)
            ctx.state.rag_config = rag_config

            ctx.state.processing_steps.append("rag_initialized")
            ctx.state.execution_status = ExecutionStatus.RUNNING

            return LoadDocuments()

        except Exception as e:
            error_msg = f"Failed to initialize RAG system: {e!s}"
            ctx.state.errors.append(error_msg)
            ctx.state.execution_status = ExecutionStatus.FAILED
            return RAGError()

    def _create_rag_config(
        self, rag_cfg: dict[str, Any], root_cfg: Any | None = None
    ) -> RAGConfig:
        """Create RAG configuration from Hydra config."""
        from DeepResearch.src.datatypes.rag import (
            EmbeddingModelType,
            EmbeddingsConfig,
            LLMModelType,
            VectorStoreConfig,
            VectorStoreType,
            VLLMConfig,
        )
        from DeepResearch.src.utils.model_registry import (
            resolve_embeddings_config,
            resolve_vllm_config,
        )

        # Create embeddings config
        embeddings_role = rag_cfg.get("embedding_model_role")
        if embeddings_role:
            embeddings_config = resolve_embeddings_config(
                root_cfg, str(embeddings_role)
            )
        else:
            embeddings_cfg = rag_cfg.get("embeddings", {})
            embeddings_config = EmbeddingsConfig(
                model_type=EmbeddingModelType(
                    embeddings_cfg.get("model_type", "sentence_transformers")
                ),
                model_name=embeddings_cfg.get("model_name", "all-MiniLM-L6-v2"),
                api_key=embeddings_cfg.get("api_key"),
                base_url=embeddings_cfg.get("base_url"),
                num_dimensions=embeddings_cfg.get("num_dimensions", 384),
                batch_size=embeddings_cfg.get("batch_size", 32),
                query_instruction=embeddings_cfg.get("query_instruction"),
                device=embeddings_cfg.get("device"),
            )

        # Create LLM config
        llm_role = rag_cfg.get("llm_model_role")
        if llm_role:
            llm_config = resolve_vllm_config(root_cfg, str(llm_role))
        else:
            llm_cfg = rag_cfg.get("llm", {})
            llm_config = VLLMConfig(
                model_type=LLMModelType(llm_cfg.get("model_type", "huggingface")),
                model_name=llm_cfg.get(
                    "model_name", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
                ),
                host=llm_cfg.get("host", "localhost"),
                port=llm_cfg.get("port", 8000),
                api_key=llm_cfg.get("api_key"),
                max_tokens=llm_cfg.get("max_tokens", 2048),
                temperature=llm_cfg.get("temperature", 0.7),
            )

        # Create vector store config
        vs_cfg = rag_cfg.get("vector_store", {})
        vector_store_config = VectorStoreConfig(
            store_type=VectorStoreType(vs_cfg.get("store_type", "chroma")),
            connection_string=vs_cfg.get("connection_string"),
            host=vs_cfg.get("host", "localhost"),
            port=vs_cfg.get("port", 8000),
            database=vs_cfg.get("database"),
            collection_name=vs_cfg.get("collection_name", "research_docs"),
            embedding_dimension=embeddings_config.num_dimensions,
        )

        return RAGConfig(
            embeddings=embeddings_config,
            llm=llm_config,
            vector_store=vector_store_config,
            chunk_size=rag_cfg.get("chunk_size", 1000),
            chunk_overlap=rag_cfg.get("chunk_overlap", 200),
        )


@dataclass
class LoadDocuments(BaseNode[RAGState, None, str]):
    """Load documents for RAG processing."""

    async def run(self, ctx: GraphRunContext[RAGState]) -> ProcessDocuments | RAGError:
        """Load documents from various sources."""
        try:
            cfg = ctx.state.config
            rag_cfg = getattr(cfg, "rag", {})

            # Load documents based on configuration
            documents = await self._load_documents(rag_cfg)
            ctx.state.documents = documents

            ctx.state.processing_steps.append(f"loaded_{len(documents)}_documents")

            return ProcessDocuments()

        except Exception as e:
            error_msg = f"Failed to load documents: {e!s}"
            ctx.state.errors.append(error_msg)
            ctx.state.execution_status = ExecutionStatus.FAILED
            return RAGError()

    async def _load_documents(self, rag_cfg: dict[str, Any]) -> list[Document]:
        """Load documents from configured sources."""
        documents = []

        # Load from file sources
        file_sources = rag_cfg.get("file_sources", [])
        for source in file_sources:
            source_docs = await self._load_from_file(source)
            documents.extend(source_docs)

        # Load from database sources
        db_sources = rag_cfg.get("database_sources", [])
        for source in db_sources:
            source_docs = await self._load_from_database(source)
            documents.extend(source_docs)

        # Load from web sources
        web_sources = rag_cfg.get("web_sources", [])
        for source in web_sources:
            source_docs = await self._load_from_web(source)
            documents.extend(source_docs)

        return documents

    async def _load_from_file(self, source: dict[str, Any]) -> list[Document]:
        """Load documents from file sources."""
        # Implementation would depend on file type (PDF, TXT, etc.)
        # For now, return empty list
        return []

    async def _load_from_database(self, source: dict[str, Any]) -> list[Document]:
        """Load documents from database sources."""
        # Implementation would connect to database and extract documents
        # For now, return empty list
        return []

    async def _load_from_web(self, source: dict[str, Any]) -> list[Document]:
        """Load documents from web sources."""
        # Implementation would scrape or fetch from web APIs
        # For now, return empty list
        return []


@dataclass
class ProcessDocuments(BaseNode[RAGState, None, str]):
    """Process and chunk documents for vector storage."""

    async def run(self, ctx: GraphRunContext[RAGState]) -> StoreDocuments | RAGError:
        """Process documents into chunks."""
        try:
            if not ctx.state.documents:
                # Create sample documents if none loaded
                ctx.state.documents = self._create_sample_documents()

            # Chunk documents based on configuration
            rag_config = ctx.state.rag_config
            if rag_config is None:
                raise RuntimeError("RAG config not initialized")
            chunked_documents = await self._chunk_documents(
                ctx.state.documents, rag_config.chunk_size, rag_config.chunk_overlap
            )
            ctx.state.documents = chunked_documents

            ctx.state.processing_steps.append(
                f"processed_{len(chunked_documents)}_chunks"
            )

            return StoreDocuments()

        except Exception as e:
            error_msg = f"Failed to process documents: {e!s}"
            ctx.state.errors.append(error_msg)
            ctx.state.execution_status = ExecutionStatus.FAILED
            return RAGError()

    def _create_sample_documents(self) -> list[Document]:
        """Create sample documents for testing."""
        return [
            Document(
                id="doc_001",
                content="Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
                metadata={"source": "research_paper", "topic": "machine_learning"},
            ),
            Document(
                id="doc_002",
                content="Deep learning uses neural networks with multiple layers to model and understand complex patterns in data.",
                metadata={"source": "research_paper", "topic": "deep_learning"},
            ),
            Document(
                id="doc_003",
                content="Natural language processing combines computational linguistics with machine learning to help computers understand human language.",
                metadata={"source": "research_paper", "topic": "nlp"},
            ),
        ]

    async def _chunk_documents(
        self, documents: list[Document], chunk_size: int, chunk_overlap: int
    ) -> list[Document]:
        """Chunk documents into smaller pieces."""
        chunked_docs = []

        for doc in documents:
            content = doc.content
            if len(content) <= chunk_size:
                chunked_docs.append(doc)
                continue

            # Simple chunking by character count
            start = 0
            chunk_id = 0
            while start < len(content):
                end = min(start + chunk_size, len(content))
                chunk_content = content[start:end]

                chunk_doc = Document(
                    id=f"{doc.id}_chunk_{chunk_id}",
                    content=chunk_content,
                    metadata={
                        **doc.metadata,
                        "chunk_id": chunk_id,
                        "original_doc_id": doc.id,
                        "chunk_start": start,
                        "chunk_end": end,
                    },
                )
                chunked_docs.append(chunk_doc)

                start = end - chunk_overlap
                chunk_id += 1

        return chunked_docs


@dataclass
class StoreDocuments(BaseNode[RAGState, None, str]):
    """Store documents in vector database."""

    async def run(self, ctx: GraphRunContext[RAGState]) -> QueryRAG | RAGError:
        """Store documents in vector store."""
        try:
            rag_config = ctx.state.rag_config

            # Initialize Embeddings via Factory
            from DeepResearch.src.datatypes.embeddings_factory import create_embeddings

            embeddings = create_embeddings(rag_config.embeddings)

            # Initialize Vector Store via Factory
            from DeepResearch.src.vector_stores import create_vector_store

            vector_store = create_vector_store(rag_config.vector_store, embeddings)

            # Initialize embeddings + LLM providers
            embeddings = self._create_embeddings_provider(rag_config)
            llm = self._create_llm_provider(rag_config)

            # Store documents
            if ctx.state.documents and vector_store:
                document_ids = await vector_store.add_documents(ctx.state.documents)
                ctx.state.processing_steps.append(
                    f"stored_{len(document_ids)}_documents"
                )

            ctx.state.processing_steps.append("embeddings_initialized")

            # Store documents (best effort). If embeddings/LLM aren't available,
            # we still keep the workflow runnable and report degraded state.
            document_ids: list[str] = []
            try:
                document_ids = await vector_store.add_documents(ctx.state.documents)
                ctx.state.processing_steps.append(
                    f"stored_{len(document_ids)}_documents"
                )
            except Exception as e:
                ctx.state.processing_steps.append("vector_store_store_failed")
                ctx.state.errors.append(f"Vector store ingest failed: {e!s}")

            # Store components in graph context for querying
            # Persist in state (GraphRunContext has no `.set()`).
            ctx.state.rag_result = ctx.state.rag_result or {}
            ctx.state.rag_result["vector_store"] = vector_store
            ctx.state.rag_result["embeddings"] = embeddings
            ctx.state.rag_result["llm"] = llm

            return QueryRAG()

        except Exception as e:
            error_msg = f"Failed to store documents: {e!s}"
            ctx.state.errors.append(error_msg)
            ctx.state.execution_status = ExecutionStatus.FAILED
            return RAGError()

    def _create_vllm_deployment(self, rag_config: RAGConfig) -> VLLMDeployment:
        """Create VLLM deployment configuration."""
        from DeepResearch.src.datatypes.vllm_integration import (
            VLLMEmbeddingServerConfig,
            VLLMServerConfig,
        )

        # Create LLM server config
        llm_server_config = VLLMServerConfig(
            model_name=rag_config.llm.model_name,
            host=rag_config.llm.host,
            port=rag_config.llm.port,
        )

        # Create embedding server config (parse URL so health checks are valid)
        if rag_config.embeddings.base_url:
            raw = str(rag_config.embeddings.base_url).strip()
            parsed = urlparse(raw if "://" in raw else f"http://{raw}")
            emb_host = parsed.hostname or "localhost"
            emb_port = parsed.port or 8001
        else:
            emb_host = "localhost"
            emb_port = 8001
        embedding_server_config = VLLMEmbeddingServerConfig(
            model_name=rag_config.embeddings.model_name,
            host=emb_host,
            port=emb_port,
        )

        return VLLMDeployment(
            llm_config=llm_server_config, embedding_config=embedding_server_config
        )

    def _create_embeddings_provider(self, rag_config: RAGConfig):
        # Prefer VLLM embeddings if base_url is configured; otherwise fall back
        # to a local sentence-transformers embedding model (best-effort).
        if rag_config.embeddings.base_url:
            return VLLMEmbeddings(rag_config.embeddings)

        # Local fallback uses sentence-transformers (dependency already declared).
        from DeepResearch.src.utils.sentence_transformers_embeddings import (
            SentenceTransformersEmbeddings,
        )

        return SentenceTransformersEmbeddings(rag_config.embeddings)

    def _create_llm_provider(self, rag_config: RAGConfig) -> VLLMLLMProvider:
        # Current production path assumes a VLLM-compatible chat/completions server.
        return VLLMLLMProvider(rag_config.llm)


@dataclass
class QueryRAG(BaseNode[RAGState, None, str]):
    """Query the RAG system with the user's question."""

    async def run(self, ctx: GraphRunContext[RAGState]) -> GenerateResponse | RAGError:
        """Execute RAG query using RAGAgent."""
        try:
            # Import here to avoid circular import
            from omegaconf import DictConfig, OmegaConf

            from DeepResearch.src.agents import RAGAgent

            # Create RAGAgent with config from state or empty config
            cfg = (
                ctx.state.config if ctx.state.config is not None else OmegaConf.create()
            )
            rag_agent = RAGAgent(cfg)
            # await rag_agent.initialize()  # Method doesn't exist

            # Create RAG query
            rag_query = RAGQuery(
                text=ctx.state.question, search_type=SearchType.SIMILARITY, top_k=5
            )

            vector_store = (ctx.state.rag_result or {}).get("vector_store")
            llm = (ctx.state.rag_result or {}).get("llm")
            if not vector_store or not llm:
                msg = "RAG components not initialized"
                raise RuntimeError(msg)

            # Execute query
            start_time = time.time()
            rag_response = await rag_agent.execute_rag_query(rag_query)
            processing_time = time.time() - start_time

            if rag_response:
                ctx.state.rag_result = (
                    rag_response.model_dump()
                    if hasattr(rag_response, "model_dump")
                    else rag_response.__dict__
                )
                ctx.state.rag_response = rag_response
                ctx.state.processing_steps.append(
                    f"query_completed_in_{processing_time:.2f}s"
                )
            return GenerateResponse()

        except Exception as e:
            error_msg = f"Failed to query RAG system: {e!s}"
            ctx.state.errors.append(error_msg)
            ctx.state.execution_status = ExecutionStatus.FAILED
            return RAGError()


@dataclass
class GenerateResponse(BaseNode[RAGState, None, str]):
    """Generate final response from RAG results."""

    async def run(
        self, ctx: GraphRunContext[RAGState]
    ) -> Annotated[End[str], Edge(label="done")]:
        """Generate and return final response."""
        try:
            rag_response = ctx.state.rag_response
            if not rag_response:
                msg = "No RAG response available"
                raise RuntimeError(msg)

            # Format final response
            final_response = self._format_response(rag_response, ctx.state)

            ctx.state.processing_steps.append("response_generated")
            ctx.state.execution_status = ExecutionStatus.SUCCESS

            return End(final_response)

        except Exception as e:
            error_msg = f"Failed to generate response: {e!s}"
            ctx.state.errors.append(error_msg)
            ctx.state.execution_status = ExecutionStatus.FAILED
            return End(self._format_error(ctx.state))

    def _format_error(self, state: RAGState) -> str:
        error_response = [
            "RAG Workflow Failed",
            "",
            f"Question: {state.question}",
            "",
            "Errors:",
        ]
        for error in state.errors:
            error_response.append(f"- {error}")
        error_response.extend(
            [
                "",
                f"Steps Completed: {', '.join(state.processing_steps)}",
                f"Status: {state.execution_status.value}",
            ]
        )
        return "\n".join(error_response)

    def _format_response(
        self, rag_response: RAGResponse | None, state: RAGState
    ) -> str:
        """Format the final response."""
        response_parts = [
            "RAG Analysis Complete",
            "",
            f"Question: {state.question}",
            "",
        ]

        # Handle agent results
        if state.rag_result:
            answer = state.rag_result.get("answer", "No answer generated")
            confidence = state.rag_result.get("confidence", 0.0)
            retrieved_docs = state.rag_result.get("retrieved_documents", [])

            response_parts.extend(
                [
                    f"Answer: {answer}",
                    f"Confidence: {confidence:.3f}",
                    "",
                    f"Retrieved Documents ({len(retrieved_docs)}):",
                ]
            )

            for i, doc in enumerate(retrieved_docs, 1):
                if isinstance(doc, dict):
                    score = doc.get("score", 0.0)
                    content = doc.get("content", "")[:200]
                    response_parts.append(f"{i}. Score: {score:.3f}")
                    response_parts.append(f"   Content: {content}...")
                else:
                    response_parts.append(f"{i}. {str(doc)[:200]}...")
                response_parts.append("")

        # Handle traditional RAG response
        elif rag_response:
            response_parts.extend(
                [
                    f"Answer: {rag_response.generated_answer}",
                    "",
                    f"Retrieved Documents ({len(rag_response.retrieved_documents)}):",
                ]
            )

            for i, result in enumerate(rag_response.retrieved_documents, 1):
                response_parts.append(f"{i}. Score: {result.score:.3f}")
                response_parts.append(f"   Content: {result.document.content[:200]}...")
                response_parts.append("")

        else:
            response_parts.append("Answer: No response generated")
            response_parts.append("")

        response_parts.extend([f"Steps Completed: {', '.join(state.processing_steps)}"])

        if state.errors:
            response_parts.extend(["", f"Errors: {', '.join(state.errors)}"])

        return "\n".join(response_parts)


@dataclass
class RAGError(BaseNode[RAGState, None, str]):
    """Handle RAG workflow errors."""

    async def run(
        self, ctx: GraphRunContext[RAGState]
    ) -> Annotated[End[str], Edge(label="error")]:
        """Handle errors and return error response."""
        error_response = [
            "RAG Workflow Failed",
            "",
            f"Question: {ctx.state.question}",
            "",
            "Errors:",
        ]

        for error in ctx.state.errors:
            error_response.append(f"- {error}")

        error_response.extend(
            [
                "",
                f"Steps Completed: {', '.join(ctx.state.processing_steps)}",
                f"Status: {ctx.state.execution_status.value}",
            ]
        )

        return End("\n".join(error_response))


# --- RAG Workflow Graph ---

rag_workflow_graph = Graph(
    nodes=(
        InitializeRAG,
        LoadDocuments,
        ProcessDocuments,
        StoreDocuments,
        QueryRAG,
        GenerateResponse,
        RAGError,
    ),
    state_type=RAGState,
    run_end_type=str,
)


def run_rag_workflow(question: str, config: DictConfig) -> str:
    """Run the complete RAG workflow."""
    # Safe synchronous entrypoint: only use this when you *own* the event loop
    # (e.g. CLI). Use `run_rag_workflow_async` from async contexts.
    state = RAGState(question=question, config=config)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(rag_workflow_graph.run(InitializeRAG(), state=state, deps=None))
        return (
            state.rag_response.generated_answer
            if state.rag_response and state.rag_response.generated_answer
            else (state.rag_response.context if state.rag_response else "")
        )
    else:
        # We're already in an event loop; callers must use the async variant.
        msg = "run_rag_workflow() cannot be called from a running event loop; use run_rag_workflow_async()."
        raise RuntimeError(msg)


async def run_rag_workflow_async(question: str, config: DictConfig) -> str:
    """Async entrypoint for running the complete RAG workflow."""
    state = RAGState(question=question, config=config)
    await rag_workflow_graph.run(InitializeRAG(), state=state, deps=None)
    return (
        state.rag_response.generated_answer
        if state.rag_response and state.rag_response.generated_answer
        else (state.rag_response.context if state.rag_response else "")
    )

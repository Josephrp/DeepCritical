from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Protocol

from DeepResearch.src.datatypes.rag import (
    Document,
    EmbeddingsConfig,
    RAGQuery,
    SearchType,
    VectorStoreConfig,
    VLLMConfig,
)
from DeepResearch.src.datatypes.vllm_integration import VLLMEmbeddings, VLLMLLMProvider
from DeepResearch.src.vector_stores import create_vector_store

from .base import ExecutionResult, ToolRunner, ToolSpec, registry


class _SupportsRAGArun(Protocol):
    async def _arun(self, params: dict[str, Any]) -> ExecutionResult: ...


def _rag_sync_run_via_arun(
    runner: _SupportsRAGArun, params: dict[str, Any]
) -> ExecutionResult:
    """Avoid ``asyncio.run()`` nested inside a running event loop (canonical path uses ``arun``)."""
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(runner._arun(params))
    return ExecutionResult(
        success=False,
        error=(
            "RAG tool synchronous run() cannot run inside an active event loop. "
            'Use canonical_registry.aexecute("rag.ingest" | "rag.retrieve" | '
            '"rag.generate", params).'
        ),
        data={"success": False, "error": "sync_run_inside_running_loop"},
        metrics={"duration_s": 0.0},
    )


def _json_loads_maybe(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        v = value.strip()
        if not v:
            return default
        return json.loads(v)
    return default


def _parse_rag_configs(
    params: dict[str, Any],
) -> tuple[EmbeddingsConfig, VectorStoreConfig, VLLMConfig]:
    embeddings_raw = _json_loads_maybe(params.get("embeddings_config"), {})
    vector_store_raw = _json_loads_maybe(params.get("vector_store_config"), {})
    llm_raw = _json_loads_maybe(params.get("llm_config"), {})

    embeddings_cfg = EmbeddingsConfig(**embeddings_raw)
    vector_store_cfg = VectorStoreConfig(**vector_store_raw)
    llm_cfg = VLLMConfig(**llm_raw)
    return embeddings_cfg, vector_store_cfg, llm_cfg


async def _build_rag_stack(
    embeddings_cfg: EmbeddingsConfig, vector_store_cfg: VectorStoreConfig
):
    """Create embeddings + vector store via the shared ``create_vector_store`` factory."""
    embeddings = VLLMEmbeddings(embeddings_cfg)
    vector_store = create_vector_store(vector_store_cfg, embeddings)
    return embeddings, vector_store


@dataclass
class RAGIngestTool(ToolRunner):
    def __init__(self):
        super().__init__(
            ToolSpec(
                name="rag.ingest",
                description="Ingest documents into a vector store for RAG retrieval.",
                inputs={
                    "documents": "JSON",
                    "embeddings_config": "JSON",
                    "vector_store_config": "JSON",
                },
                outputs={
                    "document_ids": "JSON",
                    "count": "INTEGER",
                    "success": "BOOLEAN",
                    "error": "TEXT",
                },
            )
        )

    def run(self, params: dict[str, Any]) -> ExecutionResult:
        # Sync wrapper around async ingestion (kept for legacy tool runner compat).
        return _rag_sync_run_via_arun(self, params)

    async def _arun(self, params: dict[str, Any]) -> ExecutionResult:
        start = time.time()
        try:
            docs_raw = _json_loads_maybe(params.get("documents"), [])
            docs = [
                Document(**d) if isinstance(d, dict) else Document(content=str(d))
                for d in docs_raw
            ]

            embeddings_cfg, vector_store_cfg, _llm_cfg = _parse_rag_configs(params)
            embeddings, vector_store = await _build_rag_stack(
                embeddings_cfg, vector_store_cfg
            )

            # Ensure embeddings session exists (VLLMEmbeddings manages aiohttp session lazily).
            async with embeddings:
                ids = await vector_store.add_documents(docs)

            return ExecutionResult(
                success=True,
                data={
                    "document_ids": ids,
                    "count": len(ids),
                    "success": True,
                    "error": None,
                },
                metrics={"duration_s": time.time() - start},
            )
        except (NotImplementedError, ValueError) as e:
            return ExecutionResult(
                success=False,
                error=str(e),
                data={
                    "document_ids": [],
                    "count": 0,
                    "success": False,
                    "error": str(e),
                },
                metrics={"duration_s": time.time() - start},
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                error=f"RAG ingest failed: {e!s}",
                data={
                    "document_ids": [],
                    "count": 0,
                    "success": False,
                    "error": str(e),
                },
                metrics={"duration_s": time.time() - start},
            )


@dataclass
class RAGRetrieveTool(ToolRunner):
    def __init__(self):
        super().__init__(
            ToolSpec(
                name="rag.retrieve",
                description="Retrieve relevant documents from a vector store for a query.",
                inputs={
                    "query": "TEXT",
                    "top_k": "INTEGER",
                    "search_type": "TEXT",
                    "score_threshold": "FLOAT",
                    "filters": "JSON",
                    "embeddings_config": "JSON",
                    "vector_store_config": "JSON",
                },
                outputs={
                    "results": "JSON",
                    "count": "INTEGER",
                    "success": "BOOLEAN",
                    "error": "TEXT",
                },
            )
        )

    def run(self, params: dict[str, Any]) -> ExecutionResult:
        return _rag_sync_run_via_arun(self, params)

    async def _arun(self, params: dict[str, Any]) -> ExecutionResult:
        start = time.time()
        try:
            query = str(params.get("query", "")).strip()
            if not query:
                return ExecutionResult(success=False, error="Missing query")

            top_k = int(params.get("top_k", 5))
            search_type = SearchType(str(params.get("search_type", "similarity")))
            score_threshold = params.get("score_threshold")
            filters = _json_loads_maybe(params.get("filters"), None)

            embeddings_cfg, vector_store_cfg, _llm_cfg = _parse_rag_configs(params)
            embeddings, vector_store = await _build_rag_stack(
                embeddings_cfg, vector_store_cfg
            )

            async with embeddings:
                results = await vector_store.search(
                    query=query,
                    search_type=search_type,
                    top_k=top_k,
                    score_threshold=score_threshold,
                    filters=filters or {},
                )

            results_json = [
                {
                    "score": r.score,
                    "rank": r.rank,
                    "document": r.document.model_dump(),
                }
                for r in results
            ]

            return ExecutionResult(
                success=True,
                data={
                    "results": results_json,
                    "count": len(results_json),
                    "success": True,
                    "error": None,
                },
                metrics={"duration_s": time.time() - start},
            )
        except (NotImplementedError, ValueError) as e:
            return ExecutionResult(
                success=False,
                error=str(e),
                data={"results": [], "count": 0, "success": False, "error": str(e)},
                metrics={"duration_s": time.time() - start},
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                error=f"RAG retrieve failed: {e!s}",
                data={"results": [], "count": 0, "success": False, "error": str(e)},
                metrics={"duration_s": time.time() - start},
            )


@dataclass
class RAGGenerateTool(ToolRunner):
    def __init__(self):
        super().__init__(
            ToolSpec(
                name="rag.generate",
                description="Generate an answer from retrieved documents using a configured LLM.",
                inputs={
                    "query": "TEXT",
                    "retrieved_documents": "JSON",
                    "llm_config": "JSON",
                    "max_tokens": "INTEGER",
                    "temperature": "FLOAT",
                },
                outputs={"answer": "TEXT", "success": "BOOLEAN", "error": "TEXT"},
            )
        )

    def run(self, params: dict[str, Any]) -> ExecutionResult:
        return _rag_sync_run_via_arun(self, params)

    async def _arun(self, params: dict[str, Any]) -> ExecutionResult:
        start = time.time()
        try:
            query = str(params.get("query", "")).strip()
            if not query:
                return ExecutionResult(success=False, error="Missing query")

            docs_raw = _json_loads_maybe(params.get("retrieved_documents"), [])
            # Accept either list[SearchResult-like dict] or list[Document dict]
            doc_texts: list[str] = []
            for item in docs_raw:
                if isinstance(item, dict) and "document" in item:
                    doc = item["document"]
                    if isinstance(doc, dict):
                        doc_texts.append(str(doc.get("content", "")))
                elif isinstance(item, dict):
                    doc_texts.append(str(item.get("content", "")))
                else:
                    doc_texts.append(str(item))

            context = "\n\n".join([t for t in doc_texts if t])[:8000]

            llm_raw = _json_loads_maybe(params.get("llm_config"), {})
            llm_cfg = VLLMConfig(**llm_raw)
            llm = VLLMLLMProvider(llm_cfg)

            prompt = (
                "Answer the question using only the provided context. "
                "If the context is insufficient, say what is missing.\n\n"
                f"Question: {query}\n\nContext:\n{context}"
            )

            async with llm:
                answer = await llm.generate(
                    prompt,
                    context=None,
                    max_tokens=int(params.get("max_tokens", llm_cfg.max_tokens)),
                    temperature=float(params.get("temperature", llm_cfg.temperature)),
                )

            return ExecutionResult(
                success=True,
                data={"answer": answer, "success": True, "error": None},
                metrics={"duration_s": time.time() - start},
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                error=f"RAG generate failed: {e!s}",
                data={"answer": "", "success": False, "error": str(e)},
                metrics={"duration_s": time.time() - start},
            )


registry.register("rag.ingest", RAGIngestTool)
registry.register("rag.retrieve", RAGRetrieveTool)
registry.register("rag.generate", RAGGenerateTool)

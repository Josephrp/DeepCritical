"""RAG tools use shared vector store factory."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from DeepResearch.src.datatypes.rag import (
    EmbeddingModelType,
    EmbeddingsConfig,
    VectorStoreConfig,
    VectorStoreType,
)
from DeepResearch.src.tools.rag_tools import _build_rag_stack


@pytest.mark.asyncio
async def test_build_rag_stack_calls_create_vector_store() -> None:
    emb_cfg = EmbeddingsConfig(
        model_type=EmbeddingModelType.VLLM,
        model_name="m",
        base_url="http://localhost:9999",
    )
    vs_cfg = VectorStoreConfig(
        store_type=VectorStoreType.FAISS,
        embedding_dimension=384,
    )
    with patch("DeepResearch.src.tools.rag_tools.create_vector_store") as cv:
        cv.return_value = MagicMock()
        emb, vs = await _build_rag_stack(emb_cfg, vs_cfg)
        cv.assert_called_once()
        assert emb is not None
        assert vs is cv.return_value

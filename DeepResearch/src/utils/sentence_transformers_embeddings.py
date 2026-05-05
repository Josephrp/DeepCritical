from __future__ import annotations

import asyncio
from typing import Any

from sentence_transformers import SentenceTransformer

from DeepResearch.src.datatypes.rag import Embeddings, EmbeddingsConfig


class SentenceTransformersEmbeddings(Embeddings):
    """Local embeddings provider using sentence-transformers.

    This is the production-friendly fallback when no external embedding API is configured.
    """

    def __init__(self, config: EmbeddingsConfig):
        super().__init__(config)
        # If user configured an OpenAI embedding model but didn't provide a base_url/api,
        # we still need something that works locally.
        model_name = config.model_name or "sentence-transformers/all-MiniLM-L6-v2"
        self._model = SentenceTransformer(model_name)

    async def vectorize_documents(
        self, document_chunks: list[str]
    ) -> list[list[float]]:
        if not document_chunks:
            return []
        return await asyncio.to_thread(self.vectorize_documents_sync, document_chunks)

    async def vectorize_query(self, text: str) -> list[float]:
        vectors = await self.vectorize_documents([text])
        return vectors[0] if vectors else []

    def vectorize_documents_sync(self, document_chunks: list[str]) -> list[list[float]]:
        if not document_chunks:
            return []
        embeddings = self._model.encode(document_chunks, normalize_embeddings=True)
        # sentence-transformers may return numpy arrays; convert to Python lists
        return [list(map(float, row)) for row in embeddings]

    def vectorize_query_sync(self, text: str) -> list[float]:
        vectors = self.vectorize_documents_sync([text])
        return vectors[0] if vectors else []

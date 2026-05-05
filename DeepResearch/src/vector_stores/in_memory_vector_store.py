from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

from ..datatypes.rag import (
    Chunk,
    Document,
    Embeddings,
    SearchResult,
    SearchType,
    VectorStore,
)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


@dataclass
class _StoredDoc:
    doc: Document
    embedding: list[float]


class InMemoryVectorStore(VectorStore):
    """In-memory vector store (degraded mode / dev fallback).

    This exists to keep the RAG workflow runnable when a production vector store
    isn't configured or available. It is not intended for large corpora.
    """

    def __init__(self, config, embeddings: Embeddings):
        super().__init__(config, embeddings)
        self._docs: dict[str, _StoredDoc] = {}

    async def add_documents(
        self, documents: list[Document], **kwargs: Any
    ) -> list[str]:
        if not documents:
            return []
        texts = [d.content for d in documents]
        embs = await self.embeddings.vectorize_documents(texts)
        ids: list[str] = []
        for doc, emb in zip(documents, embs, strict=False):
            self._docs[doc.id] = _StoredDoc(doc=doc, embedding=emb)
            ids.append(doc.id)
        return ids

    async def add_document_chunks(
        self, chunks: list[Chunk], **kwargs: Any
    ) -> list[str]:
        # Store chunks as Documents to satisfy VectorStore contract.
        documents = []
        for chunk in chunks:
            md: dict[str, Any] = {}
            if chunk.context is not None:
                md["context"] = chunk.context
            documents.append(
                Document(
                    id=chunk.id,
                    content=chunk.text,
                    metadata=md,
                    embedding=chunk.embedding,
                )
            )
        return await self.add_documents(documents, **kwargs)

    async def add_document_text_chunks(
        self, document_texts: list[str], **kwargs: Any
    ) -> list[str]:
        documents = [
            Document(
                id=f"inmem_chunk_{i}",
                content=text,
                metadata={"chunk_index": i, "type": "text_chunk"},
            )
            for i, text in enumerate(document_texts)
        ]
        return await self.add_documents(documents, **kwargs)

    async def delete_documents(self, document_ids: list[str]) -> bool:
        removed = False
        for doc_id in document_ids:
            removed = removed or (self._docs.pop(doc_id, None) is not None)
        return removed

    async def search(
        self,
        query: str,
        search_type: SearchType,
        retrieval_query: str | None = None,
        **kwargs: Any,
    ) -> list[SearchResult]:
        query_embedding = await self.embeddings.vectorize_query(query)
        return await self.search_with_embeddings(
            query_embedding, search_type, retrieval_query, **kwargs
        )

    async def search_with_embeddings(
        self,
        query_embedding: list[float],
        search_type: SearchType,
        retrieval_query: str | None = None,
        **kwargs: Any,
    ) -> list[SearchResult]:
        top_k = int(kwargs.get("top_k", 5))
        score_threshold = kwargs.get("score_threshold")

        scored: list[tuple[float, Document]] = []
        for stored in self._docs.values():
            score = _cosine_similarity(query_embedding, stored.embedding)
            if score_threshold is not None and score < float(score_threshold):
                continue
            scored.append((score, stored.doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        results: list[SearchResult] = []
        for idx, (score, doc) in enumerate(scored[:top_k], start=1):
            results.append(SearchResult(document=doc, score=float(score), rank=idx))
        return results

    async def get_document(self, document_id: str) -> Document | None:
        stored = self._docs.get(document_id)
        return stored.doc if stored else None

    async def update_document(self, document: Document) -> bool:
        if document.id not in self._docs:
            return False
        if document.embedding is None:
            embedding = await self.embeddings.vectorize_query(document.content)
        else:
            embedding = list(document.embedding)
        self._docs[document.id] = _StoredDoc(doc=document, embedding=embedding)
        return True

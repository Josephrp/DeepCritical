# path: neo4j_vector_service/service/embeddings.py
from __future__ import annotations
from typing import List, Literal, Optional
import os

class EmbeddingsProvider:
    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError
    def dimension(self) -> int:
        raise NotImplementedError


class SentenceTransformerEmbeddings(EmbeddingsProvider):
    """
    Proveedor de embeddings usando Sentence-Transformers.
    Por defecto: all-MiniLM-L6-v2 (384 dims, rápido).
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self._dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def dimension(self) -> int:
        return self._dim


class OpenAIEmbeddings(EmbeddingsProvider):
    """
    Proveedor usando OpenAI (opcional).
    Requiere OPENAI_API_KEY. Modelos típicos:
      - text-embedding-3-large (3072 dims)
      - text-embedding-3-small (1536 dims)
    """
    def __init__(self, model_name: str = "text-embedding-3-small", api_key: Optional[str] = None):
        try:
            from openai import OpenAI
        except Exception as e:
            raise RuntimeError("openai package not installed. pip install openai") from e

        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        if not self.client.api_key:
            raise RuntimeError("OPENAI_API_KEY no configurada.")

        self.model_name = model_name
        # Dimensiones conocidas (pueden cambiar si OpenAI actualiza modelos)
        self._dims_hint = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }

    def embed(self, texts: List[str]) -> List[List[float]]:
        resp = self.client.embeddings.create(model=self.model_name, input=texts)
        return [d.embedding for d in resp.data]

    def dimension(self) -> int:
        return self._dims_hint.get(self.model_name, 1536)

# path: neo4j_vector_service/agentes/tools.py
from typing import Any, Dict, List

class VectorSearchTool:
    """Tool wrapper for Neo4j vector search."""

    def __init__(self, vector_store: Any, k: int = 5):
        self.vector_store = vector_store
        self.k = k

    def run(self, query: str) -> List[Dict]:
        """Run a vector similarity search."""
        return self.vector_store.similarity_search(query, k=self.k)


class HybridSearchTool:
    """Tool wrapper for Neo4j hybrid search (vector + BM25)."""

    def __init__(self, vector_store: Any, k: int = 5):
        self.vector_store = vector_store
        self.k = k

    def run(self, query: str) -> List[Dict]:
        """Run a hybrid search."""
        return self.vector_store.hybrid_search(query, k=self.k)

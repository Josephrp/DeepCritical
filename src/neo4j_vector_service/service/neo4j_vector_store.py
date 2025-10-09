# path: neo4j_vector_service/service/neo4j_vector_store.py
from __future__ import annotations
from typing import List, Dict, Any, Optional, Iterable
from neo4j import GraphDatabase
import logging

try:
    # Usaremos un proveedor de embeddings inyectable (ver embeddings.py)
    from .embeddings import EmbeddingsProvider, SentenceTransformerEmbeddings
except Exception:
    # fallback mínimo si aún no existe embeddings.py
    from sentence_transformers import SentenceTransformer
    class EmbeddingsProvider:
        def embed(self, texts: List[str]) -> List[List[float]]:
            raise NotImplementedError
        def dimension(self) -> int:
            raise NotImplementedError
    class SentenceTransformerEmbeddings(EmbeddingsProvider):
        def __init__(self, model_name: str):
            self.model = SentenceTransformer(model_name)
            self._dim = self.model.get_sentence_embedding_dimension()
        def embed(self, texts: List[str]) -> List[List[float]]:
            return self.model.encode(texts, convert_to_numpy=True).tolist()
        def dimension(self) -> int:
            return self._dim

logger = logging.getLogger(__name__)


class Neo4jVectorStore:
    """
    Standalone Neo4j Vector Store Service.

    - Gestiona embeddings y búsqueda vectorial/híbrida.
    - Proporciona helpers para crear/asegurar índices en Neo4j.
    """

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        label: str = "Publication",
        text_field: str = "abstract",
        id_field: str = "doi",
        embedding_field: str = "abstract_embedding",
        vector_index_name: str = "publication_abstract_embeddings",
        fulltext_index_name: str = "publication_fulltext"
    ):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

        # Embeddings
        self.embedder: EmbeddingsProvider = SentenceTransformerEmbeddings(embedding_model)
        self.embedding_dim = self.embedder.dimension()

        # Config de grafo/índices
        self.label = label
        self.text_field = text_field
        self.id_field = id_field
        self.embedding_field = embedding_field
        self.vector_index_name = vector_index_name
        self.fulltext_index_name = fulltext_index_name

        logger.info(f"Initialized Neo4j Vector Store (dim={self.embedding_dim}, label=:{self.label})")

    # ---------- Infra ----------
    def verify_connection(self) -> bool:
        try:
            with self.driver.session(database=self.database) as s:
                s.run("RETURN 1").consume()
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    def ensure_indexes(self) -> None:
        """
        Crea (si no existen) el índice vectorial y el full-text.
        IMPORTANTE: La dimensión debe coincidir con el modelo de embeddings.
        """
        cypher_vec = f"""
        CREATE VECTOR INDEX {self.vector_index_name} IF NOT EXISTS
        FOR (n:{self.label}) ON (n.{self.embedding_field})
        OPTIONS {{
          indexConfig: {{
            `vector.dimensions`: $dim,
            `vector.similarity_function`: 'cosine'
          }}
        }}
        """
        cypher_ft = f"""
        CREATE FULLTEXT INDEX {self.fulltext_index_name} IF NOT EXISTS
        FOR (n:{self.label}) ON EACH [n.title, n.{self.text_field}]
        """
        with self.driver.session(database=self.database) as s:
            s.run(cypher_vec, dim=self.embedding_dim).consume()
            s.run(cypher_ft).consume()
        logger.info(f"Indexes ensured: {self.vector_index_name} (dim={self.embedding_dim}), {self.fulltext_index_name}")

    # ---------- Ingesta ----------
    def upsert_publications(self, rows: Iterable[Dict[str, Any]], embed_if_missing: bool = True) -> int:
        """
        Upsert de publicaciones.

        rows: dicts con keys esperadas:
          - id_field (p.ej. 'doi')  (obligatoria)
          - title (opcional)
          - text_field (p.ej. 'abstract') (opcional pero recomendado)
          - embedding (opcional; si falta y embed_if_missing, se calcula)
          - year, citedBy, meta... (opcionales)
        """
        rows = list(rows)
        if not rows:
            return 0

        # Calcula embeddings si faltan y se permite
        texts_to_embed = []
        idx_map = []
        for i, r in enumerate(rows):
            if "embedding" not in r or r["embedding"] is None:
                if not embed_if_missing:
                    continue
                # texto base: title + abstract
                title = r.get("title", "") or ""
                text = r.get(self.text_field, "") or ""
                texts_to_embed.append((i, f"{title}\n\n{text}".strip()))
                idx_map.append(i)

        if texts_to_embed:
            _, texts = zip(*texts_to_embed)
            embs = self.embedder.embed(list(texts))
            for j, emb in zip(idx_map, embs):
                rows[j]["embedding"] = emb

        # Upsert por lotes
        cypher = f"""
        UNWIND $rows AS row
        WITH row WHERE row.{self.id_field} IS NOT NULL
        MERGE (n:{self.label} {{ {self.id_field}: row.{self.id_field} }})
        SET n.title = coalesce(row.title, n.title),
            n.{self.text_field} = coalesce(row.{self.text_field}, n.{self.text_field}),
            n.{self.embedding_field} = coalesce(row.embedding, n.{self.embedding_field}),
            n.year = coalesce(row.year, n.year),
            n.citedBy = coalesce(row.citedBy, n.citedBy),
            n.meta = coalesce(row.meta, n.meta)
        RETURN count(n) AS n
        """
        with self.driver.session(database=self.database) as s:
            n = s.run(cypher, rows=rows).single()["n"]
        return int(n)

    # ---------- Búsqueda ----------
    def embed_text(self, text: str) -> List[float]:
        return self.embedder.embed([text])[0]

    def similarity_search(
        self,
        query: str,
        k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Vector similarity search (HNSW index).

        filter_dict soporta:
          - min_year, max_year, min_citations (int)
          - equals: dict simple en filter_dict["equals"] con {prop: value}
        """
        qvec = self.embed_text(query)
        params = {"index": self.vector_index_name, "k": max(k * 2, k), "qvec": qvec, "final_k": k}

        where_clauses = [f"n.{self.embedding_field} IS NOT NULL"]
        if filter_dict:
            if "min_year" in filter_dict:
                where_clauses.append("toInteger(n.year) >= $min_year")
                params["min_year"] = int(filter_dict["min_year"])
            if "max_year" in filter_dict:
                where_clauses.append("toInteger(n.year) <= $max_year")
                params["max_year"] = int(filter_dict["max_year"])
            if "min_citations" in filter_dict:
                where_clauses.append("toInteger(n.citedBy) >= $min_citations")
                params["min_citations"] = int(filter_dict["min_citations"])
            if "equals" in filter_dict and isinstance(filter_dict["equals"], dict):
                for i, (k_, v_) in enumerate(filter_dict["equals"].items()):
                    key = f"eq{i}"
                    where_clauses.append(f"n.{k_} = ${key}")
                    params[key] = v_

        where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
        cypher = f"""
        CALL db.index.vector.queryNodes($index, $k, $qvec) YIELD node AS n, score
        {where_sql}
        RETURN n.{self.id_field} AS id,
               n.title AS title,
               n.{self.text_field} AS text,
               n.year AS year,
               n.citedBy AS citations,
               score
        ORDER BY score DESC
        LIMIT $final_k
        """
        with self.driver.session(database=self.database) as s:
            rs = s.run(cypher, **params)
            return [dict(r) for r in rs]

    def hybrid_search(
        self,
        query: str,
        k: int = 10,
        vector_weight: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Híbrido vector + señal de citas (normalizadas por máx. global).
        Nota: calcular el máximo global en cada llamada puede ser costoso en grafos grandes.
        """
        qvec = self.embed_text(query)
        cypher = f"""
        // Vecinos por índice vectorial (amplía el recall con k_mult)
        CALL db.index.vector.queryNodes($index, $k_mult, $qvec) YIELD node AS n, score AS vscore

        // Normaliza citas contra el máximo global (puede ser costoso)
        MATCH (m:{self.label})
        WITH n, vscore, max(toFloat(m.citedBy)) AS max_cit
        WITH n, vscore, (CASE WHEN max_cit > 0 THEN toFloat(n.citedBy)/max_cit ELSE 0.0 END) AS cscore

        WITH n, vscore, cscore,
             ($vw * vscore + (1.0 - $vw) * cscore) AS hscore
        RETURN n.{self.id_field} AS id,
               n.title AS title,
               n.{self.text_field} AS text,
               n.year AS year,
               n.citedBy AS citations,
               vscore AS vector_score,
               cscore AS citation_score,
               hscore AS hybrid_score
        ORDER BY hybrid_score DESC
        LIMIT $k
        """
        with self.driver.session(database=self.database) as s:
            rs = s.run(
                cypher,
                index=self.vector_index_name,
                k=k,
                k_mult=max(k * 3, k),
                qvec=qvec,
                vw=float(vector_weight),
            )
            return [dict(r) for r in rs]

    def close(self) -> None:
        if self.driver:
            self.driver.close()

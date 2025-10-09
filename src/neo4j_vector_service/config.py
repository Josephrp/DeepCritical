import os
from dotenv import load_dotenv, find_dotenv

# Carga el .env más cercano (raíz del repo normalmente)
load_dotenv(find_dotenv(), override=False)

# ---- Utilidades simples de lectura ----
def _get_str(name: str, default: str | None = None) -> str | None:
    return os.getenv(name, default)

def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default

# ---- Neo4j ----
NEO4J_URI: str        = _get_str("NEO4J_URI", "bolt://localhost:7690")
NEO4J_USER: str       = _get_str("NEO4J_USER", "neo4j")
NEO4J_PASSWORD: str   = _get_str("NEO4J_PASSWORD", "neo4j")
NEO4J_DATABASE: str   = _get_str("NEO4J_DATABASE", "neo4j")

# ---- Ajustes de vectores/índice ----
# Nombre del índice vectorial (debe coincidir con el que crearás en Cypher)
VECTOR_INDEX_NAME: str = _get_str("VECTOR_INDEX_NAME", "publication_abstract_embeddings")
# Etiqueta y propiedad donde guardas el embedding
VECTOR_LABEL: str      = _get_str("VECTOR_LABEL", "Publication")
VECTOR_PROPERTY: str   = _get_str("VECTOR_PROPERTY", "embedding")

# Dimensión del embedding y función de similitud
EMBED_DIM: int         = _get_int("EMBED_DIM", 1536)
SIM_FUNCTION: str      = _get_str("VECTOR_SIMILARITY", "cosine")  # 'cosine' | 'euclidean' | 'inner' (según Neo4j)

# ---- OpenAI (opcional, si generas embeddings desde el código) ----
OPENAI_API_KEY: str | None     = _get_str("OPENAI_API_KEY")
OPENAI_EMBED_MODEL: str        = _get_str("OPENAI_EMBED_MODEL", "text-embedding-3-small")
OPENAI_TIMEOUT_SECS: int       = _get_int("OPENAI_TIMEOUT_SECS", 60)

# ---- Helper opcional para crear driver (útil en scripts) ----
def make_driver():
    """Devuelve un neo4j.Driver ya autenticado."""
    from neo4j import GraphDatabase  # import local para no forzar dependencia al importar config
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

__all__ = [
    "NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD", "NEO4J_DATABASE",
    "VECTOR_INDEX_NAME", "VECTOR_LABEL", "VECTOR_PROPERTY",
    "EMBED_DIM", "SIM_FUNCTION",
    "OPENAI_API_KEY", "OPENAI_EMBED_MODEL", "OPENAI_TIMEOUT_SECS",
    "make_driver",
]

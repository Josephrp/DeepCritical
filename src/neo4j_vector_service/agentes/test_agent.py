# path: neo4j_vector_service/agentes/test_agent.py
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from neo4j_vector_service.service.neo4j_vector_store import Neo4jVectorStore
from neo4j_vector_service.agentes.retrieval_agent import Neo4jRetrievalAgent


def main():
    # Conexión Neo4j
    store = Neo4jVectorStore(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7690"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "password"),
    )
    store.ensure_indexes()

    # Crear agente
    agent = Neo4jRetrievalAgent(
        store,
        llm_model="google/flan-t5-small",
        tool_defaults={"k": 3}
    )

    # Probar query
    query = "recycled PET packaging LCA"
    print(f"Ejecutando consulta: {query}")
    result = agent.run(query, use_hybrid=False)
    print("\n=== Respuesta ===")
    print(result)


if __name__ == "__main__":
    main()

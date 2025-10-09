# path: main.py
import os
import uvicorn
from fastapi import FastAPI
from pathlib import Path
from dotenv import load_dotenv

from neo4j_vector_service.service.neo4j_vector_store import Neo4jVectorStore
from neo4j_vector_service.agentes.retrieval_agent import Neo4jRetrievalAgent

# Cargar variables de entorno desde .env
load_dotenv(Path(__file__).resolve().parent / ".env")

app = FastAPI(title="Bibliometric APIs", version="0.1.0")

# Inicializar Neo4j store usando variables de entorno
store = Neo4jVectorStore(
    uri=os.getenv("NEO4J_URI", "bolt://localhost:7690"),
    user=os.getenv("NEO4J_USER", "neo4j"),
    password=os.getenv("NEO4J_PASSWORD", "einstein1983"),
    database=os.getenv("NEO4J_DATABASE", "neo4j"),
)
store.ensure_indexes()

# Inicializar agente con modelo HF
agent = Neo4jRetrievalAgent(store, llm_model="google/flan-t5-base")

@app.get("/")
def root():
    return {"message": "Bibliometric API is running 🚀"}

@app.get("/search")
def search(query: str):
    result = agent.run(query)
    return {"query": query, "result": result}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

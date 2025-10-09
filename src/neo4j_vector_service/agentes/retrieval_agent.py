# path: neo4j_vector_service/agentes/retrieval_agent.py
from __future__ import annotations
from typing import List, Optional, Dict

from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline

from .tools import VectorSearchTool, HybridSearchTool


class Neo4jRetrievalAgent:
    """Simplified retrieval agent for Neo4j + HuggingFace LLM (manual orchestration)."""

    def __init__(
        self,
        vector_store,
        llm_model: str = "google/flan-t5-small",
        temperature: float = 0.0,
        tool_defaults: Optional[Dict] = None,
    ):
        self.vector_store = vector_store
        self.tool_defaults = tool_defaults or {}

        # Tools
        self.vector_tool = VectorSearchTool(self.vector_store, **self.tool_defaults)
        self.hybrid_tool = HybridSearchTool(self.vector_store, **self.tool_defaults)

        # HuggingFace pipeline (text2text)
        hf_pipeline = pipeline(
            "text2text-generation",
            model=llm_model,
            device=-1  # CPU
        )
        self.llm = HuggingFacePipeline(pipeline=hf_pipeline)

    def run(self, query: str, use_hybrid: bool = False) -> str:
        """Run retrieval + summarization."""
        # 1) Run search
        if use_hybrid:
            results = self.hybrid_tool.run(query)
        else:
            results = self.vector_tool.run(query)

        if not results:
            return "No results found in the database."

        # 2) Format results
        formatted = "\n".join(
            f"- {r.get('title','[No title]')} ({r.get('year','?')}): {r.get('summary','')}"
            for r in results
        )

        # 3) Ask LLM to summarize
        prompt = f"Summarize the following papers related to '{query}':\n{formatted}"
        answer = self.llm(prompt)

        return str(answer)

"""
Pydantic AI Embedding Tools for DeepCritical

This module provides embedding tools that integrate with Pydantic AI agents,
supporting both VLLM local models and external embedding services.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from datetime import datetime
import numpy as np

try:
    import faiss
except ImportError:
    raise ImportError("faiss-cpu is required. Install with: pip install faiss-cpu")

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, defer

from ..src.datatypes.rag import (
    Document, Chunk, SearchResult, SearchType, EmbeddingsConfig, 
    EmbeddingModelType, VectorStoreConfig, VectorStoreType, RAGQuery, RAGResponse
)
from ..src.datatypes.vllm_integration import VLLMEmbeddings, VLLMLLMProvider
from ..src.datatypes.vllm_dataclass import VllmConfig, create_vllm_config
from .base import ToolRunner, ToolSpec, ExecutionResult
from .tool_registry import ToolRegistry


class EmbeddingRequest(BaseModel):
    """Request for embedding generation."""
    texts: List[str] = Field(..., description="Texts to embed")
    model: str = Field("text-embedding-3-small", description="Embedding model to use")
    batch_size: int = Field(32, description="Batch size for processing")
    normalize: bool = Field(True, description="Whether to normalize embeddings")


class EmbeddingResponse(BaseModel):
    """Response from embedding generation."""
    embeddings: List[List[float]] = Field(..., description="Generated embeddings")
    model: str = Field(..., description="Model used")
    dimensions: int = Field(..., description="Embedding dimensions")
    processing_time: float = Field(..., description="Processing time in seconds")


class VectorStoreRequest(BaseModel):
    """Request for vector store operations."""
    operation: str = Field(..., description="Operation: add, search, delete, list")
    documents: Optional[List[Document]] = Field(None, description="Documents to add")
    query: Optional[str] = Field(None, description="Search query")
    top_k: int = Field(5, description="Number of results to return")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
    database_name: str = Field("default", description="Database name")


class VectorStoreResponse(BaseModel):
    """Response from vector store operations."""
    success: bool = Field(..., description="Operation success")
    results: Optional[List[SearchResult]] = Field(None, description="Search results")
    document_ids: Optional[List[str]] = Field(None, description="Added document IDs")
    message: str = Field(..., description="Response message")
    processing_time: float = Field(..., description="Processing time in seconds")


class EmbeddingTool(ToolRunner):
    """Pydantic AI embedding tool with VLLM integration."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(ToolSpec(
            name="embedding_tool",
            description="Generate embeddings for text using VLLM or external services",
            inputs={
                "texts": "List of texts to embed",
                "model": "Embedding model name",
                "batch_size": "Batch size for processing",
                "normalize": "Whether to normalize embeddings"
            },
            outputs={
                "embeddings": "Generated embeddings",
                "model": "Model used",
                "dimensions": "Embedding dimensions",
                "processing_time": "Processing time"
            }
        ))
        
        self.config = config or {}
        self.embeddings_provider: Optional[VLLMEmbeddings] = None
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize embeddings provider based on configuration."""
        embedding_config = self.config.get("embeddings", {})
        
        if embedding_config.get("use_vllm", False):
            # Use VLLM for local embeddings
            vllm_config = embedding_config.get("vllm_config", {})
            embeddings_config = EmbeddingsConfig(
                model_type=EmbeddingModelType.CUSTOM,
                model_name=vllm_config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
                base_url=f"{vllm_config.get('host', 'localhost')}:{vllm_config.get('port', 8001)}",
                num_dimensions=vllm_config.get("dimensions", 384),
                batch_size=vllm_config.get("batch_size", 32)
            )
            self.embeddings_provider = VLLMEmbeddings(embeddings_config)
        else:
            # Use external service (OpenAI, etc.)
            self.embeddings_provider = None
    
    def run(self, params: Dict[str, Any]) -> ExecutionResult:
        """Run embedding generation synchronously."""
        try:
            request = EmbeddingRequest(**params)
            return asyncio.run(self._generate_embeddings_async(request))
        except Exception as e:
            return ExecutionResult(success=False, error=str(e))
    
    async def _generate_embeddings_async(self, request: EmbeddingRequest) -> ExecutionResult:
        """Generate embeddings asynchronously."""
        import time
        start_time = time.time()
        
        try:
            if self.embeddings_provider:
                # Use VLLM embeddings
                embeddings = await self.embeddings_provider.vectorize_documents(request.texts)
            else:
                # Use external service (OpenAI)
                embeddings = await self._generate_external_embeddings(request)
            
            processing_time = time.time() - start_time
            
            response = EmbeddingResponse(
                embeddings=embeddings,
                model=request.model,
                dimensions=len(embeddings[0]) if embeddings else 0,
                processing_time=processing_time
            )
            
            return ExecutionResult(success=True, data=response.dict())
            
        except Exception as e:
            return ExecutionResult(success=False, error=str(e))
    
    async def _generate_external_embeddings(self, request: EmbeddingRequest) -> List[List[float]]:
        """Generate embeddings using external service."""
        try:
            from openai import AsyncOpenAI
            
            client = AsyncOpenAI(
                api_key=self.config.get("openai", {}).get("api_key"),
                base_url=self.config.get("openai", {}).get("base_url")
            )
            
            embeddings = []
            batch_size = request.batch_size
            
            for i in range(0, len(request.texts), batch_size):
                batch = request.texts[i:i + batch_size]
                
                response = await client.embeddings.create(
                    model=request.model,
                    input=batch
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
            
            return embeddings
            
        except ImportError:
            raise ImportError("openai package required for external embeddings")
        except Exception as e:
            raise RuntimeError(f"Failed to generate external embeddings: {e}")


class VectorStoreTool(ToolRunner):
    """Pydantic AI vector store tool with FAISS and SQLite backend."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(ToolSpec(
            name="vector_store_tool",
            description="Manage vector store with FAISS and SQLite backend",
            inputs={
                "operation": "Operation: add, search, delete, list",
                "documents": "Documents to add",
                "query": "Search query",
                "top_k": "Number of results",
                "filters": "Metadata filters",
                "database_name": "Database name"
            },
            outputs={
                "success": "Operation success",
                "results": "Search results",
                "document_ids": "Added document IDs",
                "message": "Response message",
                "processing_time": "Processing time"
            }
        ))
        
        self.config = config or {}
        self.data_dir = Path(self.config.get("data_dir", "./data/embeddings"))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "embeddings.db"
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for metadata."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS databases (
                    name TEXT PRIMARY KEY,
                    description TEXT,
                    embedding_model TEXT,
                    embedding_dimensions INTEGER,
                    document_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    database_name TEXT,
                    content TEXT,
                    metadata TEXT,
                    embedding_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (database_name) REFERENCES databases (name)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_database 
                ON documents (database_name)
            """)
    
    def run(self, params: Dict[str, Any]) -> ExecutionResult:
        """Run vector store operation synchronously."""
        try:
            request = VectorStoreRequest(**params)
            return asyncio.run(self._process_request_async(request))
        except Exception as e:
            return ExecutionResult(success=False, error=str(e))
    
    async def _process_request_async(self, request: VectorStoreRequest) -> ExecutionResult:
        """Process vector store request asynchronously."""
        import time
        start_time = time.time()
        
        try:
            if request.operation == "add":
                result = await self._add_documents(request)
            elif request.operation == "search":
                result = await self._search_documents(request)
            elif request.operation == "delete":
                result = await self._delete_documents(request)
            elif request.operation == "list":
                result = await self._list_databases(request)
            else:
                raise ValueError(f"Unknown operation: {request.operation}")
            
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            
            return ExecutionResult(success=True, data=result.dict())
            
        except Exception as e:
            return ExecutionResult(success=False, error=str(e))
    
    async def _add_documents(self, request: VectorStoreRequest) -> VectorStoreResponse:
        """Add documents to vector store."""
        if not request.documents:
            raise ValueError("Documents required for add operation")
        
        # Generate embeddings for documents
        embedding_tool = EmbeddingTool(self.config)
        texts = [doc.content for doc in request.documents]
        
        embedding_request = EmbeddingRequest(
            texts=texts,
            model=self.config.get("embeddings", {}).get("model", "text-embedding-3-small")
        )
        
        embedding_result = await embedding_tool._generate_embeddings_async(embedding_request)
        if not embedding_result.success:
            raise RuntimeError(f"Failed to generate embeddings: {embedding_result.error}")
        
        embeddings = embedding_result.data["embeddings"]
        dimensions = embedding_result.data["dimensions"]
        
        # Load or create FAISS index
        index_path = self.data_dir / f"{request.database_name}.faiss"
        if index_path.exists():
            index = faiss.read_index(str(index_path))
        else:
            index = faiss.IndexFlatIP(dimensions)
        
        # Add embeddings to index
        embedding_ids = []
        for i, embedding in enumerate(embeddings):
            embedding_id = index.ntotal
            index.add(np.array([embedding], dtype=np.float32))
            embedding_ids.append(embedding_id)
        
        # Save FAISS index
        faiss.write_index(index, str(index_path))
        
        # Save to SQLite
        with sqlite3.connect(self.db_path) as conn:
            # Update or create database record
            conn.execute("""
                INSERT OR REPLACE INTO databases 
                (name, description, embedding_model, embedding_dimensions, document_count, updated_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                request.database_name,
                f"Database with {len(request.documents)} documents",
                embedding_request.model,
                dimensions,
                index.ntotal
            ))
            
            # Insert documents
            for doc, embedding_id in zip(request.documents, embedding_ids):
                conn.execute("""
                    INSERT OR REPLACE INTO documents 
                    (id, database_name, content, metadata, embedding_id)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    doc.id,
                    request.database_name,
                    doc.content,
                    json.dumps(doc.metadata),
                    embedding_id
                ))
        
        return VectorStoreResponse(
            success=True,
            document_ids=[doc.id for doc in request.documents],
            message=f"Added {len(request.documents)} documents to {request.database_name}"
        )
    
    async def _search_documents(self, request: VectorStoreRequest) -> VectorStoreResponse:
        """Search documents in vector store."""
        if not request.query:
            raise ValueError("Query required for search operation")
        
        # Generate query embedding
        embedding_tool = EmbeddingTool(self.config)
        embedding_request = EmbeddingRequest(
            texts=[request.query],
            model=self.config.get("embeddings", {}).get("model", "text-embedding-3-small")
        )
        
        embedding_result = await embedding_tool._generate_embeddings_async(embedding_request)
        if not embedding_result.success:
            raise RuntimeError(f"Failed to generate query embedding: {embedding_result.error}")
        
        query_embedding = embedding_result.data["embeddings"][0]
        
        # Load FAISS index
        index_path = self.data_dir / f"{request.database_name}.faiss"
        if not index_path.exists():
            return VectorStoreResponse(
                success=True,
                results=[],
                message=f"Database {request.database_name} not found"
            )
        
        index = faiss.read_index(str(index_path))
        
        # Search
        scores, indices = index.search(np.array([query_embedding], dtype=np.float32), request.top_k)
        
        # Get documents from SQLite
        results = []
        with sqlite3.connect(self.db_path) as conn:
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # No more results
                    break
                
                cursor = conn.execute("""
                    SELECT id, content, metadata FROM documents 
                    WHERE database_name = ? AND embedding_id = ?
                """, (request.database_name, int(idx)))
                
                row = cursor.fetchone()
                if row:
                    doc_id, content, metadata = row
                    metadata_dict = json.loads(metadata) if metadata else {}
                    
                    # Apply filters
                    if request.filters and not self._matches_filters(metadata_dict, request.filters):
                        continue
                    
                    document = Document(
                        id=doc_id,
                        content=content,
                        metadata=metadata_dict
                    )
                    
                    result = SearchResult(
                        document=document,
                        score=float(score),
                        rank=len(results) + 1
                    )
                    results.append(result)
        
        return VectorStoreResponse(
            success=True,
            results=results,
            message=f"Found {len(results)} results"
        )
    
    async def _delete_documents(self, request: VectorStoreRequest) -> VectorStoreResponse:
        """Delete documents from vector store."""
        # Implementation would depend on specific requirements
        # For now, return not implemented
        return VectorStoreResponse(
            success=False,
            message="Delete operation not implemented"
        )
    
    async def _list_databases(self, request: VectorStoreRequest) -> VectorStoreResponse:
        """List available databases."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT name, description, document_count, created_at 
                FROM databases ORDER BY created_at DESC
            """)
            
            databases = []
            for row in cursor.fetchall():
                databases.append({
                    "name": row[0],
                    "description": row[1],
                    "document_count": row[2],
                    "created_at": row[3]
                })
        
        return VectorStoreResponse(
            success=True,
            message=f"Found {len(databases)} databases",
            results=[SearchResult(
                document=Document(
                    id=db["name"],
                    content=db["description"],
                    metadata=db
                ),
                score=1.0,
                rank=i + 1
            ) for i, db in enumerate(databases)]
        )
    
    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if metadata matches filters."""
        for key, value in filters.items():
            if key not in metadata:
                return False
            if metadata[key] != value:
                return False
        return True


# Pydantic AI tool functions for agent integration
@defer
def generate_embeddings(
    texts: List[str], 
    model: str = "text-embedding-3-small",
    batch_size: int = 32,
    ctx: RunContext[Any]
) -> EmbeddingResponse:
    """Generate embeddings for texts using configured provider."""
    tool = EmbeddingTool(ctx.deps.get("embedding_config", {}))
    result = tool.run({
        "texts": texts,
        "model": model,
        "batch_size": batch_size,
        "normalize": True
    })
    
    if not result.success:
        raise RuntimeError(f"Failed to generate embeddings: {result.error}")
    
    return EmbeddingResponse(**result.data)


@defer
def add_documents_to_vector_store(
    documents: List[Document],
    database_name: str = "default",
    ctx: RunContext[Any]
) -> VectorStoreResponse:
    """Add documents to vector store."""
    tool = VectorStoreTool(ctx.deps.get("vector_store_config", {}))
    result = tool.run({
        "operation": "add",
        "documents": [doc.dict() for doc in documents],
        "database_name": database_name
    })
    
    if not result.success:
        raise RuntimeError(f"Failed to add documents: {result.error}")
    
    return VectorStoreResponse(**result.data)


@defer
def search_vector_store(
    query: str,
    database_name: str = "default",
    top_k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
    ctx: RunContext[Any]
) -> VectorStoreResponse:
    """Search vector store for similar documents."""
    tool = VectorStoreTool(ctx.deps.get("vector_store_config", {}))
    result = tool.run({
        "operation": "search",
        "query": query,
        "database_name": database_name,
        "top_k": top_k,
        "filters": filters or {}
    })
    
    if not result.success:
        raise RuntimeError(f"Failed to search vector store: {result.error}")
    
    return VectorStoreResponse(**result.data)


@defer
def list_vector_databases(
    ctx: RunContext[Any]
) -> VectorStoreResponse:
    """List available vector databases."""
    tool = VectorStoreTool(ctx.deps.get("vector_store_config", {}))
    result = tool.run({
        "operation": "list"
    })
    
    if not result.success:
        raise RuntimeError(f"Failed to list databases: {result.error}")
    
    return VectorStoreResponse(**result.data)


# Register tools
def register_embedding_tools():
    """Register embedding tools with the global registry."""
    registry = ToolRegistry()
    
    # Register embedding tool
    embedding_tool = EmbeddingTool()
    registry.register_tool(embedding_tool)
    
    # Register vector store tool
    vector_store_tool = VectorStoreTool()
    registry.register_tool(vector_store_tool)
    
    return registry


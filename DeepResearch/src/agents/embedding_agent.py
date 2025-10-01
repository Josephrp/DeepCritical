"""
Embedding Agent for DeepCritical

This agent orchestrates embedding operations including document processing,
vector store management, and embedding generation using Pydantic AI patterns.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from ..datatypes.rag import (
    Document, Chunk, RAGQuery, RAGResponse, RAGConfig,
    SearchResult, SearchType, EmbeddingsConfig, VectorStoreConfig
)
from ..datatypes.agent_types import AgentDependencies, AgentResult, AgentType
from .base_agent import BaseAgent
from ...tools.embedding_tools import (
    generate_embeddings, add_documents_to_vector_store, 
    search_vector_store, list_vector_databases
)


class EmbeddingTask(BaseModel):
    """Task for embedding operations."""
    task_type: str = Field(..., description="Type of task: create_db, add_docs, search, sync")
    database_name: str = Field("default", description="Target database name")
    documents: Optional[List[Document]] = Field(None, description="Documents to process")
    query: Optional[str] = Field(None, description="Search query")
    config: Optional[Dict[str, Any]] = Field(None, description="Task-specific configuration")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class EmbeddingResult(BaseModel):
    """Result from embedding operations."""
    success: bool = Field(..., description="Operation success")
    task_type: str = Field(..., description="Type of task performed")
    database_name: str = Field(..., description="Database name")
    documents_processed: int = Field(0, description="Number of documents processed")
    search_results: Optional[List[SearchResult]] = Field(None, description="Search results")
    document_ids: Optional[List[str]] = Field(None, description="Processed document IDs")
    message: str = Field(..., description="Result message")
    processing_time: float = Field(..., description="Processing time in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class EmbeddingAgent(BaseAgent):
    """Agent for orchestrating embedding operations."""
    
    def __init__(self, model_name: str = "anthropic:claude-sonnet-4-0"):
        super().__init__(AgentType.EMBEDDING, model_name)
        self._register_embedding_tools()
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for embedding agent."""
        return """You are an Embedding Agent specialized in managing vector embeddings and document processing.

Your capabilities include:
- Creating and managing embedding databases
- Processing documents and generating embeddings
- Performing semantic search across vector stores
- Synchronizing embedding databases with external services
- Optimizing embedding workflows for performance

You work with various embedding models including VLLM local models and external services like OpenAI.
You maintain FAISS vector indices and SQLite metadata stores for efficient retrieval.

Always provide clear feedback on operations and suggest optimizations when appropriate."""

    def _get_default_instructions(self) -> List[str]:
        """Get default instructions for embedding agent."""
        return [
            "Analyze the embedding task and determine the best approach",
            "Use appropriate embedding models based on the task requirements",
            "Process documents in batches for optimal performance",
            "Validate input data before processing",
            "Provide detailed feedback on operation results",
            "Suggest optimizations for embedding workflows",
            "Handle errors gracefully and provide recovery suggestions"
        ]
    
    def _register_embedding_tools(self):
        """Register embedding-specific tools."""
        # Register Pydantic AI tools
        self._agent.tool(generate_embeddings)
        self._agent.tool(add_documents_to_vector_store)
        self._agent.tool(search_vector_store)
        self._agent.tool(list_vector_databases)
    
    async def create_database(
        self, 
        database_name: str, 
        documents: List[Document],
        config: Optional[Dict[str, Any]] = None,
        deps: Optional[AgentDependencies] = None
    ) -> AgentResult:
        """Create a new embedding database from documents."""
        task = EmbeddingTask(
            task_type="create_db",
            database_name=database_name,
            documents=documents,
            config=config or {}
        )
        
        return await self.execute(task, deps)
    
    async def add_documents(
        self,
        database_name: str,
        documents: List[Document],
        config: Optional[Dict[str, Any]] = None,
        deps: Optional[AgentDependencies] = None
    ) -> AgentResult:
        """Add documents to an existing embedding database."""
        task = EmbeddingTask(
            task_type="add_docs",
            database_name=database_name,
            documents=documents,
            config=config or {}
        )
        
        return await self.execute(task, deps)
    
    async def search_database(
        self,
        database_name: str,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        deps: Optional[AgentDependencies] = None
    ) -> AgentResult:
        """Search an embedding database for similar documents."""
        task = EmbeddingTask(
            task_type="search",
            database_name=database_name,
            query=query,
            config={
                "top_k": top_k,
                "filters": filters or {}
            }
        )
        
        return await self.execute(task, deps)
    
    async def sync_database(
        self,
        database_name: str,
        sync_config: Dict[str, Any],
        deps: Optional[AgentDependencies] = None
    ) -> AgentResult:
        """Synchronize embedding database with external services."""
        task = EmbeddingTask(
            task_type="sync",
            database_name=database_name,
            config=sync_config
        )
        
        return await self.execute(task, deps)
    
    async def list_databases(
        self,
        deps: Optional[AgentDependencies] = None
    ) -> AgentResult:
        """List available embedding databases."""
        task = EmbeddingTask(
            task_type="list",
            database_name=""
        )
        
        return await self.execute(task, deps)
    
    async def process_documents(
        self,
        documents: List[Document],
        processing_config: Dict[str, Any],
        deps: Optional[AgentDependencies] = None
    ) -> AgentResult:
        """Process documents for embedding generation."""
        # This would involve chunking, preprocessing, etc.
        task = EmbeddingTask(
            task_type="process",
            documents=documents,
            config=processing_config
        )
        
        return await self.execute(task, deps)
    
    async def optimize_database(
        self,
        database_name: str,
        optimization_config: Dict[str, Any],
        deps: Optional[AgentDependencies] = None
    ) -> AgentResult:
        """Optimize an embedding database for better performance."""
        task = EmbeddingTask(
            task_type="optimize",
            database_name=database_name,
            config=optimization_config
        )
        
        return await self.execute(task, deps)
    
    def _process_result(self, result: Any) -> EmbeddingResult:
        """Process agent result into EmbeddingResult."""
        if isinstance(result, EmbeddingResult):
            return result
        
        # Handle different result types
        if isinstance(result, dict):
            return EmbeddingResult(**result)
        
        # Default processing
        return EmbeddingResult(
            success=True,
            task_type="unknown",
            database_name="default",
            message=str(result)
        )


class EmbeddingWorkflowAgent(BaseAgent):
    """Agent for managing complex embedding workflows."""
    
    def __init__(self, model_name: str = "anthropic:claude-sonnet-4-0"):
        super().__init__(AgentType.EMBEDDING, model_name)
        self.embedding_agent = EmbeddingAgent(model_name)
        self._register_workflow_tools()
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for embedding workflow agent."""
        return """You are an Embedding Workflow Agent that orchestrates complex embedding operations.

You manage multi-step workflows including:
- Document ingestion and preprocessing
- Batch embedding generation
- Vector store management
- Database synchronization
- Performance optimization
- Error recovery and retry logic

You coordinate with the Embedding Agent to execute individual operations and manage workflow state."""

    def _get_default_instructions(self) -> List[str]:
        """Get default instructions for embedding workflow agent."""
        return [
            "Break down complex embedding tasks into manageable steps",
            "Coordinate with the Embedding Agent for individual operations",
            "Manage workflow state and handle errors gracefully",
            "Optimize batch processing for large document sets",
            "Provide progress updates for long-running workflows",
            "Implement retry logic for failed operations",
            "Validate workflow results and suggest improvements"
        ]
    
    def _register_workflow_tools(self):
        """Register workflow-specific tools."""
        # Register embedding agent methods as tools
        self._agent.tool(self.embedding_agent.create_database)
        self._agent.tool(self.embedding_agent.add_documents)
        self._agent.tool(self.embedding_agent.search_database)
        self._agent.tool(self.embedding_agent.sync_database)
        self._agent.tool(self.embedding_agent.list_databases)
    
    async def execute_workflow(
        self,
        workflow_config: Dict[str, Any],
        deps: Optional[AgentDependencies] = None
    ) -> AgentResult:
        """Execute a complex embedding workflow."""
        workflow_type = workflow_config.get("type", "default")
        
        if workflow_type == "document_ingestion":
            return await self._document_ingestion_workflow(workflow_config, deps)
        elif workflow_type == "database_migration":
            return await self._database_migration_workflow(workflow_config, deps)
        elif workflow_type == "batch_processing":
            return await self._batch_processing_workflow(workflow_config, deps)
        else:
            return await self._default_workflow(workflow_config, deps)
    
    async def _document_ingestion_workflow(
        self,
        config: Dict[str, Any],
        deps: Optional[AgentDependencies] = None
    ) -> AgentResult:
        """Execute document ingestion workflow."""
        documents = config.get("documents", [])
        database_name = config.get("database_name", "default")
        chunk_size = config.get("chunk_size", 1000)
        chunk_overlap = config.get("chunk_overlap", 200)
        
        # Step 1: Process documents (chunking, preprocessing)
        processed_docs = await self._process_documents_for_ingestion(
            documents, chunk_size, chunk_overlap
        )
        
        # Step 2: Create or add to database
        if config.get("create_new", False):
            result = await self.embedding_agent.create_database(
                database_name, processed_docs, config.get("embedding_config"), deps
            )
        else:
            result = await self.embedding_agent.add_documents(
                database_name, processed_docs, config.get("embedding_config"), deps
            )
        
        return result
    
    async def _database_migration_workflow(
        self,
        config: Dict[str, Any],
        deps: Optional[AgentDependencies] = None
    ) -> AgentResult:
        """Execute database migration workflow."""
        source_db = config.get("source_database")
        target_db = config.get("target_database")
        
        # Step 1: List source databases
        source_result = await self.embedding_agent.list_databases(deps)
        
        # Step 2: Migrate data (implementation would depend on specific requirements)
        # This is a placeholder for the actual migration logic
        
        return AgentResult(
            success=True,
            data={"message": f"Migration from {source_db} to {target_db} completed"}
        )
    
    async def _batch_processing_workflow(
        self,
        config: Dict[str, Any],
        deps: Optional[AgentDependencies] = None
    ) -> AgentResult:
        """Execute batch processing workflow."""
        documents = config.get("documents", [])
        batch_size = config.get("batch_size", 100)
        database_name = config.get("database_name", "default")
        
        results = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            result = await self.embedding_agent.add_documents(
                database_name, batch, config.get("embedding_config"), deps
            )
            results.append(result)
        
        return AgentResult(
            success=True,
            data={
                "message": f"Processed {len(documents)} documents in {len(results)} batches",
                "batch_results": results
            }
        )
    
    async def _default_workflow(
        self,
        config: Dict[str, Any],
        deps: Optional[AgentDependencies] = None
    ) -> AgentResult:
        """Execute default workflow."""
        return await self.execute(config, deps)
    
    async def _process_documents_for_ingestion(
        self,
        documents: List[Document],
        chunk_size: int,
        chunk_overlap: int
    ) -> List[Document]:
        """Process documents for ingestion (chunking, preprocessing)."""
        processed_docs = []
        
        for doc in documents:
            # Simple chunking implementation
            if len(doc.content) > chunk_size:
                chunks = self._chunk_text(doc.content, chunk_size, chunk_overlap)
                for i, chunk_text in enumerate(chunks):
                    chunk_doc = Document(
                        id=f"{doc.id}_chunk_{i}",
                        content=chunk_text,
                        metadata={
                            **doc.metadata,
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "parent_document": doc.id
                        }
                    )
                    processed_docs.append(chunk_doc)
            else:
                processed_docs.append(doc)
        
        return processed_docs
    
    def _chunk_text(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Chunk text into smaller pieces."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - chunk_overlap
            
            if start >= len(text):
                break
        
        return chunks


class EmbeddingSyncAgent(BaseAgent):
    """Agent for synchronizing embedding databases with external services."""
    
    def __init__(self, model_name: str = "anthropic:claude-sonnet-4-0"):
        super().__init__(AgentType.EMBEDDING, model_name)
        self.embedding_agent = EmbeddingAgent(model_name)
        self._register_sync_tools()
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for embedding sync agent."""
        return """You are an Embedding Sync Agent that manages synchronization of embedding databases with external services.

You handle:
- Uploading databases to HuggingFace Hub
- Downloading databases from external sources
- Managing database versions and metadata
- Handling authentication and permissions
- Resolving conflicts during synchronization
- Maintaining data integrity during sync operations"""

    def _get_default_instructions(self) -> List[str]:
        """Get default instructions for embedding sync agent."""
        return [
            "Validate database integrity before synchronization",
            "Handle authentication for external services",
            "Manage version conflicts during sync operations",
            "Provide clear feedback on sync progress",
            "Implement rollback mechanisms for failed syncs",
            "Optimize sync operations for large databases",
            "Maintain audit logs of sync operations"
        ]
    
    def _register_sync_tools(self):
        """Register sync-specific tools."""
        # Register embedding agent methods
        self._agent.tool(self.embedding_agent.list_databases)
        self._agent.tool(self.embedding_agent.sync_database)
    
    async def upload_to_huggingface(
        self,
        database_name: str,
        repository: str,
        description: str = "",
        private: bool = False,
        deps: Optional[AgentDependencies] = None
    ) -> AgentResult:
        """Upload database to HuggingFace Hub."""
        sync_config = {
            "action": "upload",
            "repository": repository,
            "description": description,
            "private": private
        }
        
        return await self.embedding_agent.sync_database(database_name, sync_config, deps)
    
    async def download_from_huggingface(
        self,
        repository: str,
        local_name: Optional[str] = None,
        overwrite: bool = False,
        deps: Optional[AgentDependencies] = None
    ) -> AgentResult:
        """Download database from HuggingFace Hub."""
        sync_config = {
            "action": "download",
            "repository": repository,
            "local_name": local_name,
            "overwrite": overwrite
        }
        
        return await self.embedding_agent.sync_database("", sync_config, deps)
    
    async def sync_with_external_service(
        self,
        service_config: Dict[str, Any],
        deps: Optional[AgentDependencies] = None
    ) -> AgentResult:
        """Sync with external service using custom configuration."""
        return await self.execute(service_config, deps)


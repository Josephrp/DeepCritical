"""
Embedding Workflow for DeepCritical

This module defines Pydantic Graph workflows for embedding operations,
including document processing, vector store management, and synchronization.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from pydantic import BaseModel, Field
from pydantic_graph import BaseNode, GraphRunContext, NextNode, End

from ..datatypes.rag import Document, Chunk, RAGQuery, RAGResponse, RAGConfig
from ..datatypes.agent_types import AgentDependencies
from ..agents.embedding_agent import EmbeddingAgent, EmbeddingWorkflowAgent, EmbeddingSyncAgent
from ..datatypes.vector_store_impl import VectorStoreFactory


class EmbeddingWorkflowState(BaseModel):
    """State for embedding workflow execution."""
    # Input data
    documents: List[Document] = Field(default_factory=list, description="Documents to process")
    database_name: str = Field("default", description="Target database name")
    workflow_type: str = Field("document_ingestion", description="Type of workflow")
    
    # Configuration
    embedding_config: Dict[str, Any] = Field(default_factory=dict, description="Embedding configuration")
    vector_store_config: Dict[str, Any] = Field(default_factory=dict, description="Vector store configuration")
    sync_config: Dict[str, Any] = Field(default_factory=dict, description="Sync configuration")
    
    # Processing state
    processed_documents: List[Document] = Field(default_factory=list, description="Processed documents")
    chunks: List[Chunk] = Field(default_factory=list, description="Document chunks")
    document_ids: List[str] = Field(default_factory=list, description="Added document IDs")
    
    # Results
    search_results: List[Any] = Field(default_factory=list, description="Search results")
    sync_results: Dict[str, Any] = Field(default_factory=dict, description="Sync results")
    
    # Workflow state
    current_step: str = Field("", description="Current workflow step")
    completed_steps: List[str] = Field(default_factory=list, description="Completed steps")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")
    
    # Metadata
    processing_time: float = Field(0.0, description="Total processing time")
    start_time: Optional[datetime] = Field(None, description="Workflow start time")
    end_time: Optional[datetime] = Field(None, description="Workflow end time")


class InitializeEmbeddingWorkflow(BaseNode[EmbeddingWorkflowState]):
    """Initialize embedding workflow."""
    
    async def run(self, ctx: GraphRunContext[EmbeddingWorkflowState]) -> NextNode:
        """Initialize the embedding workflow."""
        ctx.state.start_time = datetime.now()
        ctx.state.current_step = "initialize"
        
        # Validate input
        if not ctx.state.documents and ctx.state.workflow_type != "sync":
            ctx.state.errors.append("No documents provided for processing")
            return End("No documents to process")
        
        # Initialize agents
        ctx.set("embedding_agent", EmbeddingAgent())
        ctx.set("workflow_agent", EmbeddingWorkflowAgent())
        ctx.set("sync_agent", EmbeddingSyncAgent())
        
        # Initialize vector store
        vector_store = VectorStoreFactory.create_faiss_vector_store(
            data_dir=ctx.state.vector_store_config.get("data_dir", "./data/vector_store"),
            collection_name=ctx.state.database_name
        )
        ctx.set("vector_store", vector_store)
        
        ctx.state.completed_steps.append("initialize")
        ctx.state.current_step = "process_documents"
        
        return NextNode()


class ProcessDocumentsNode(BaseNode[EmbeddingWorkflowState]):
    """Process documents for embedding."""
    
    async def run(self, ctx: GraphRunContext[EmbeddingWorkflowState]) -> NextNode:
        """Process documents for embedding."""
        ctx.state.current_step = "process_documents"
        
        try:
            workflow_agent = ctx.get("workflow_agent")
            deps = AgentDependencies.from_config(ctx.state.embedding_config)
            
            # Process documents based on workflow type
            if ctx.state.workflow_type == "document_ingestion":
                result = await workflow_agent._document_ingestion_workflow({
                    "documents": [doc.dict() for doc in ctx.state.documents],
                    "database_name": ctx.state.database_name,
                    "chunk_size": ctx.state.embedding_config.get("chunk_size", 1000),
                    "chunk_overlap": ctx.state.embedding_config.get("chunk_overlap", 200),
                    "create_new": ctx.state.embedding_config.get("create_new", True),
                    "embedding_config": ctx.state.embedding_config
                }, deps)
                
                if result.success:
                    ctx.state.processed_documents = ctx.state.documents
                    ctx.state.document_ids = result.data.get("document_ids", [])
                else:
                    ctx.state.errors.append(f"Document processing failed: {result.error}")
                    return End("Document processing failed")
            
            elif ctx.state.workflow_type == "batch_processing":
                result = await workflow_agent._batch_processing_workflow({
                    "documents": [doc.dict() for doc in ctx.state.documents],
                    "database_name": ctx.state.database_name,
                    "batch_size": ctx.state.embedding_config.get("batch_size", 100),
                    "embedding_config": ctx.state.embedding_config
                }, deps)
                
                if result.success:
                    ctx.state.processed_documents = ctx.state.documents
                    ctx.state.document_ids = result.data.get("document_ids", [])
                else:
                    ctx.state.errors.append(f"Batch processing failed: {result.error}")
                    return End("Batch processing failed")
            
            else:
                # Default processing
                ctx.state.processed_documents = ctx.state.documents
            
            ctx.state.completed_steps.append("process_documents")
            ctx.state.current_step = "generate_embeddings"
            
            return NextNode()
            
        except Exception as e:
            ctx.state.errors.append(f"Error processing documents: {str(e)}")
            return End("Document processing error")


class GenerateEmbeddingsNode(BaseNode[EmbeddingWorkflowState]):
    """Generate embeddings for documents."""
    
    async def run(self, ctx: GraphRunContext[EmbeddingWorkflowState]) -> NextNode:
        """Generate embeddings for processed documents."""
        ctx.state.current_step = "generate_embeddings"
        
        try:
            embedding_agent = ctx.get("embedding_agent")
            deps = AgentDependencies.from_config(ctx.state.embedding_config)
            
            # Generate embeddings
            if ctx.state.embedding_config.get("create_new", True):
                result = await embedding_agent.create_database(
                    ctx.state.database_name,
                    ctx.state.processed_documents,
                    ctx.state.embedding_config,
                    deps
                )
            else:
                result = await embedding_agent.add_documents(
                    ctx.state.database_name,
                    ctx.state.processed_documents,
                    ctx.state.embedding_config,
                    deps
                )
            
            if result.success:
                ctx.state.document_ids = result.data.get("document_ids", [])
                ctx.state.completed_steps.append("generate_embeddings")
                ctx.state.current_step = "validate_results"
            else:
                ctx.state.errors.append(f"Embedding generation failed: {result.error}")
                return End("Embedding generation failed")
            
            return NextNode()
            
        except Exception as e:
            ctx.state.errors.append(f"Error generating embeddings: {str(e)}")
            return End("Embedding generation error")


class ValidateResultsNode(BaseNode[EmbeddingWorkflowState]):
    """Validate embedding results."""
    
    async def run(self, ctx: GraphRunContext[EmbeddingWorkflowState]) -> NextNode:
        """Validate the embedding results."""
        ctx.state.current_step = "validate_results"
        
        try:
            embedding_agent = ctx.get("embedding_agent")
            deps = AgentDependencies.from_config(ctx.state.embedding_config)
            
            # Test search functionality
            test_query = "test query for validation"
            search_result = await embedding_agent.search_database(
                ctx.state.database_name,
                test_query,
                top_k=1,
                deps=deps
            )
            
            if search_result.success:
                ctx.state.completed_steps.append("validate_results")
                
                # Check if sync is needed
                if ctx.state.sync_config.get("enabled", False):
                    ctx.state.current_step = "sync_database"
                    return NextNode()
                else:
                    ctx.state.current_step = "finalize"
                    return NextNode()
            else:
                ctx.state.errors.append("Validation failed: search test unsuccessful")
                return End("Validation failed")
            
        except Exception as e:
            ctx.state.errors.append(f"Error validating results: {str(e)}")
            return End("Validation error")


class SyncDatabaseNode(BaseNode[EmbeddingWorkflowState]):
    """Synchronize database with external services."""
    
    async def run(self, ctx: GraphRunContext[EmbeddingWorkflowState]) -> NextNode:
        """Synchronize database with external services."""
        ctx.state.current_step = "sync_database"
        
        try:
            sync_agent = ctx.get("sync_agent")
            deps = AgentDependencies.from_config(ctx.state.sync_config)
            
            sync_action = ctx.state.sync_config.get("action", "upload")
            
            if sync_action == "upload":
                result = await sync_agent.upload_to_huggingface(
                    ctx.state.database_name,
                    ctx.state.sync_config.get("repository"),
                    ctx.state.sync_config.get("description", ""),
                    ctx.state.sync_config.get("private", False),
                    deps
                )
            elif sync_action == "download":
                result = await sync_agent.download_from_huggingface(
                    ctx.state.sync_config.get("repository"),
                    ctx.state.sync_config.get("local_name"),
                    ctx.state.sync_config.get("overwrite", False),
                    deps
                )
            else:
                result = await sync_agent.sync_with_external_service(
                    ctx.state.sync_config,
                    deps
                )
            
            if result.success:
                ctx.state.sync_results = result.data
                ctx.state.completed_steps.append("sync_database")
                ctx.state.current_step = "finalize"
            else:
                ctx.state.errors.append(f"Sync failed: {result.error}")
                # Continue to finalize even if sync fails
                ctx.state.current_step = "finalize"
            
            return NextNode()
            
        except Exception as e:
            ctx.state.errors.append(f"Error syncing database: {str(e)}")
            ctx.state.current_step = "finalize"
            return NextNode()


class FinalizeWorkflowNode(BaseNode[EmbeddingWorkflowState]):
    """Finalize the embedding workflow."""
    
    async def run(self, ctx: GraphRunContext[EmbeddingWorkflowState]) -> NextNode:
        """Finalize the workflow and prepare results."""
        ctx.state.current_step = "finalize"
        ctx.state.end_time = datetime.now()
        
        if ctx.state.start_time:
            ctx.state.processing_time = (ctx.state.end_time - ctx.state.start_time).total_seconds()
        
        # Prepare final results
        if ctx.state.errors:
            message = f"Workflow completed with {len(ctx.state.errors)} errors"
        else:
            message = f"Workflow completed successfully. Processed {len(ctx.state.documents)} documents"
        
        ctx.state.completed_steps.append("finalize")
        
        return End(message)


class EmbeddingWorkflowGraph:
    """Graph for embedding workflow execution."""
    
    def __init__(self):
        self.nodes = {
            "initialize": InitializeEmbeddingWorkflow(),
            "process_documents": ProcessDocumentsNode(),
            "generate_embeddings": GenerateEmbeddingsNode(),
            "validate_results": ValidateResultsNode(),
            "sync_database": SyncDatabaseNode(),
            "finalize": FinalizeWorkflowNode()
        }
    
    async def run_workflow(
        self,
        documents: List[Document],
        database_name: str = "default",
        workflow_type: str = "document_ingestion",
        embedding_config: Optional[Dict[str, Any]] = None,
        vector_store_config: Optional[Dict[str, Any]] = None,
        sync_config: Optional[Dict[str, Any]] = None
    ) -> EmbeddingWorkflowState:
        """Run the embedding workflow."""
        from pydantic_graph import Graph
        
        # Create initial state
        state = EmbeddingWorkflowState(
            documents=documents,
            database_name=database_name,
            workflow_type=workflow_type,
            embedding_config=embedding_config or {},
            vector_store_config=vector_store_config or {},
            sync_config=sync_config or {}
        )
        
        # Create graph
        graph = Graph(
            nodes=list(self.nodes.values()),
            start_node=self.nodes["initialize"]
        )
        
        # Run workflow
        result = await graph.run(state)
        
        return result


class EmbeddingWorkflowOrchestrator:
    """Orchestrator for embedding workflows."""
    
    def __init__(self):
        self.workflow_graph = EmbeddingWorkflowGraph()
    
    async def create_database_workflow(
        self,
        documents: List[Document],
        database_name: str,
        embedding_config: Optional[Dict[str, Any]] = None,
        vector_store_config: Optional[Dict[str, Any]] = None
    ) -> EmbeddingWorkflowState:
        """Run workflow to create a new embedding database."""
        return await self.workflow_graph.run_workflow(
            documents=documents,
            database_name=database_name,
            workflow_type="document_ingestion",
            embedding_config=embedding_config,
            vector_store_config=vector_store_config
        )
    
    async def batch_processing_workflow(
        self,
        documents: List[Document],
        database_name: str,
        batch_size: int = 100,
        embedding_config: Optional[Dict[str, Any]] = None
    ) -> EmbeddingWorkflowState:
        """Run workflow for batch processing documents."""
        config = embedding_config or {}
        config["batch_size"] = batch_size
        
        return await self.workflow_graph.run_workflow(
            documents=documents,
            database_name=database_name,
            workflow_type="batch_processing",
            embedding_config=config
        )
    
    async def sync_workflow(
        self,
        database_name: str,
        sync_config: Dict[str, Any]
    ) -> EmbeddingWorkflowState:
        """Run workflow for database synchronization."""
        return await self.workflow_graph.run_workflow(
            documents=[],
            database_name=database_name,
            workflow_type="sync",
            sync_config=sync_config
        )
    
    async def migration_workflow(
        self,
        source_database: str,
        target_database: str,
        migration_config: Optional[Dict[str, Any]] = None
    ) -> EmbeddingWorkflowState:
        """Run workflow for database migration."""
        config = migration_config or {}
        config.update({
            "source_database": source_database,
            "target_database": target_database
        })
        
        return await self.workflow_graph.run_workflow(
            documents=[],
            database_name=target_database,
            workflow_type="migration",
            embedding_config=config
        )


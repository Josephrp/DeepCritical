"""
Data types for DeepCritical research workflows.

This module provides Pydantic models and data structures for various
research workflows including bioinformatics and RAG operations.
"""

from .bioinformatics import (
    EvidenceCode,
    GOTerm,
    GOAnnotation,
    PubMedPaper,
    GEOPlatform,
    GEOSeries,
    GeneExpressionProfile,
    DrugTarget,
    PerturbationProfile,
    ProteinStructure,
    ProteinInteraction,
    FusedDataset,
    ReasoningTask,
    DataFusionRequest,
)

from .rag import (
    SearchType,
    EmbeddingModelType,
    LLMModelType,
    VectorStoreType,
    Document,
    SearchResult,
    EmbeddingsConfig,
    VLLMConfig,
    VectorStoreConfig,
    RAGQuery,
    RAGResponse,
    RAGConfig,
    Embeddings,
    VectorStore,
    LLMProvider,
    RAGSystem,
    RAGWorkflowState,
)

from .vllm_integration import (
    VLLMEmbeddings,
    VLLMLLMProvider,
    VLLMServerConfig,
    VLLMEmbeddingServerConfig,
    VLLMDeployment,
    VLLMRAGSystem,
)

__all__ = [
    # Bioinformatics types
    "EvidenceCode",
    "GOTerm",
    "GOAnnotation",
    "PubMedPaper",
    "GEOPlatform",
    "GEOSeries",
    "GeneExpressionProfile",
    "DrugTarget",
    "PerturbationProfile",
    "ProteinStructure",
    "ProteinInteraction",
    "FusedDataset",
    "ReasoningTask",
    "DataFusionRequest",
    # RAG types
    "SearchType",
    "EmbeddingModelType",
    "LLMModelType",
    "VectorStoreType",
    "Document",
    "SearchResult",
    "EmbeddingsConfig",
    "VLLMConfig",
    "VectorStoreConfig",
    "RAGQuery",
    "RAGResponse",
    "RAGConfig",
    "Embeddings",
    "VectorStore",
    "LLMProvider",
    "RAGSystem",
    "RAGWorkflowState",
    # VLLM integration types
    "VLLMEmbeddings",
    "VLLMLLMProvider",
    "VLLMServerConfig",
    "VLLMEmbeddingServerConfig",
    "VLLMDeployment",
    "VLLMRAGSystem",
]

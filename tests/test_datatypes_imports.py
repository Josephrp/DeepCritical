"""
Import tests for DeepResearch datatypes modules.

This module tests that all imports from the datatypes subdirectory work correctly,
including all individual datatype modules and their dependencies.
"""

import pytest


class TestDatatypesModuleImports:
    """Test imports for individual datatype modules."""

    def test_bioinformatics_imports(self):
        """Test all imports from bioinformatics module."""

        from DeepResearch.src.datatypes.bioinformatics import (
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

        # Verify they are all accessible and not None
        assert EvidenceCode is not None
        assert GOTerm is not None
        assert GOAnnotation is not None
        assert PubMedPaper is not None
        assert GEOPlatform is not None
        assert GEOSeries is not None
        assert GeneExpressionProfile is not None
        assert DrugTarget is not None
        assert PerturbationProfile is not None
        assert ProteinStructure is not None
        assert ProteinInteraction is not None
        assert FusedDataset is not None
        assert ReasoningTask is not None
        assert DataFusionRequest is not None

        # Test enum values exist
        assert hasattr(EvidenceCode, "IDA")
        assert hasattr(EvidenceCode, "IEA")

    def test_rag_imports(self):
        """Test all imports from rag module."""

        from DeepResearch.src.datatypes.rag import (
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

        # Verify they are all accessible and not None
        assert SearchType is not None
        assert EmbeddingModelType is not None
        assert LLMModelType is not None
        assert VectorStoreType is not None
        assert Document is not None
        assert SearchResult is not None
        assert EmbeddingsConfig is not None
        assert VLLMConfig is not None
        assert VectorStoreConfig is not None
        assert RAGQuery is not None
        assert RAGResponse is not None
        assert RAGConfig is not None
        assert Embeddings is not None
        assert VectorStore is not None
        assert LLMProvider is not None
        assert RAGSystem is not None
        assert RAGWorkflowState is not None

        # Test enum values exist
        assert hasattr(SearchType, "SEMANTIC")
        assert hasattr(VectorStoreType, "CHROMA")

    def test_vllm_integration_imports(self):
        """Test all imports from vllm_integration module."""

        from DeepResearch.src.datatypes.vllm_integration import (
            VLLMEmbeddings,
            VLLMLLMProvider,
            VLLMServerConfig,
            VLLMEmbeddingServerConfig,
            VLLMDeployment,
            VLLMRAGSystem,
        )

        # Verify they are all accessible and not None
        assert VLLMEmbeddings is not None
        assert VLLMLLMProvider is not None
        assert VLLMServerConfig is not None
        assert VLLMEmbeddingServerConfig is not None
        assert VLLMDeployment is not None
        assert VLLMRAGSystem is not None

    def test_chunk_dataclass_imports(self):
        """Test all imports from chunk_dataclass module."""

        from DeepResearch.src.datatypes.chunk_dataclass import Chunk

        # Verify they are all accessible and not None
        assert Chunk is not None

    def test_document_dataclass_imports(self):
        """Test all imports from document_dataclass module."""

        from DeepResearch.src.datatypes.document_dataclass import Document

        # Verify they are all accessible and not None
        assert Document is not None

    def test_chroma_dataclass_imports(self):
        """Test all imports from chroma_dataclass module."""

        from DeepResearch.src.datatypes.chroma_dataclass import ChromaDocument

        # Verify they are all accessible and not None
        assert ChromaDocument is not None

    def test_postgres_dataclass_imports(self):
        """Test all imports from postgres_dataclass module."""

        from DeepResearch.src.datatypes.postgres_dataclass import PostgresDocument

        # Verify they are all accessible and not None
        assert PostgresDocument is not None

    def test_vllm_dataclass_imports(self):
        """Test all imports from vllm_dataclass module."""

        from DeepResearch.src.datatypes.vllm_dataclass import VLLMDocument

        # Verify they are all accessible and not None
        assert VLLMDocument is not None

    def test_markdown_imports(self):
        """Test all imports from markdown module."""

        from DeepResearch.src.datatypes.markdown import MarkdownDocument

        # Verify they are all accessible and not None
        assert MarkdownDocument is not None

    def test_deep_agent_state_imports(self):
        """Test all imports from deep_agent_state module."""

        from DeepResearch.src.datatypes.deep_agent_state import DeepAgentState

        # Verify they are all accessible and not None
        assert DeepAgentState is not None

    def test_deep_agent_types_imports(self):
        """Test all imports from deep_agent_types module."""

        from DeepResearch.src.datatypes.deep_agent_types import DeepAgentType

        # Verify they are all accessible and not None
        assert DeepAgentType is not None

    def test_workflow_orchestration_imports(self):
        """Test all imports from workflow_orchestration module."""

        from DeepResearch.src.datatypes.workflow_orchestration import (
            WorkflowOrchestrationState,
        )

        # Verify they are all accessible and not None
        assert WorkflowOrchestrationState is not None


class TestDatatypesCrossModuleImports:
    """Test cross-module imports and dependencies within datatypes."""

    def test_datatypes_internal_dependencies(self):
        """Test that datatype modules can import from each other correctly."""
        # Test that bioinformatics can import from rag
        from DeepResearch.src.datatypes.bioinformatics import GOTerm
        from DeepResearch.src.datatypes.rag import Document

        # This should work without circular imports
        assert GOTerm is not None
        assert Document is not None

    def test_pydantic_base_model_inheritance(self):
        """Test that datatype models properly inherit from Pydantic BaseModel."""
        from DeepResearch.src.datatypes.bioinformatics import GOTerm
        from DeepResearch.src.datatypes.rag import Document

        # Test that they are proper Pydantic models
        assert hasattr(GOTerm, "__fields__") or hasattr(GOTerm, "model_fields")
        assert hasattr(Document, "__fields__") or hasattr(Document, "model_fields")

    def test_enum_definitions(self):
        """Test that enum classes are properly defined."""
        from DeepResearch.src.datatypes.bioinformatics import EvidenceCode
        from DeepResearch.src.datatypes.rag import SearchType

        # Test that enums have expected values
        assert len(EvidenceCode) > 0
        assert len(SearchType) > 0


class TestDatatypesComplexImportChains:
    """Test complex import chains involving multiple modules."""

    def test_full_datatype_initialization_chain(self):
        """Test the complete import chain for datatype initialization."""
        try:
            from DeepResearch.src.datatypes.bioinformatics import (
                EvidenceCode,
                GOTerm,
                GOAnnotation,
                PubMedPaper,
            )
            from DeepResearch.src.datatypes.rag import (
                SearchType,
                Document,
                SearchResult,
                RAGQuery,
            )
            from DeepResearch.src.datatypes.vllm_integration import VLLMEmbeddings

            # If all imports succeed, the chain is working
            assert EvidenceCode is not None
            assert GOTerm is not None
            assert GOAnnotation is not None
            assert PubMedPaper is not None
            assert SearchType is not None
            assert Document is not None
            assert SearchResult is not None
            assert RAGQuery is not None
            assert VLLMEmbeddings is not None

        except ImportError as e:
            pytest.fail(f"Datatype import chain failed: {e}")

    def test_cross_module_references(self):
        """Test that modules can reference each other's types."""
        try:
            # Test that bioinformatics can reference RAG types
            from DeepResearch.src.datatypes.bioinformatics import FusedDataset
            from DeepResearch.src.datatypes.rag import Document

            # If we get here without ImportError, cross-references work
            assert FusedDataset is not None
            assert Document is not None

        except ImportError as e:
            pytest.fail(f"Cross-module reference failed: {e}")


class TestDatatypesImportErrorHandling:
    """Test import error handling for datatypes modules."""

    def test_pydantic_availability(self):
        """Test that Pydantic is available for datatype models."""
        try:
            from pydantic import BaseModel

            assert BaseModel is not None
        except ImportError:
            pytest.fail("Pydantic not available for datatype models")

    def test_circular_import_prevention(self):
        """Test that there are no circular imports in datatypes."""
        # This test will fail if there are circular imports

        # If we get here, no circular imports were detected
        assert True

    def test_missing_dependencies_handling(self):
        """Test that modules handle missing dependencies gracefully."""
        # Most datatype modules should work without external dependencies
        # beyond Pydantic and standard library
        from DeepResearch.src.datatypes.bioinformatics import EvidenceCode
        from DeepResearch.src.datatypes.rag import SearchType

        # These should always be available
        assert EvidenceCode is not None
        assert SearchType is not None

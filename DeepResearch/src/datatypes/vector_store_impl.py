"""
Vector Store Implementation for DeepCritical

This module provides concrete implementations of vector stores using FAISS and SQLite,
integrating with the RAG system and supporting various embedding models.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime
import numpy as np

try:
    import faiss
except ImportError:
    raise ImportError("faiss-cpu is required. Install with: pip install faiss-cpu")

from pydantic import BaseModel, Field

from .rag import (
    VectorStore, Embeddings, Document, Chunk, SearchResult, SearchType,
    VectorStoreConfig, VectorStoreType
)
from .vllm_integration import VLLMEmbeddings


class FAISSVectorStore(VectorStore):
    """FAISS-based vector store implementation with SQLite metadata."""
    
    def __init__(self, config: VectorStoreConfig, embeddings: Embeddings):
        super().__init__(config, embeddings)
        self.data_dir = Path(config.connection_string or "./data/vector_store")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.data_dir / "metadata.db"
        self.index_cache: Dict[str, faiss.Index] = {}
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for metadata storage."""
        with sqlite3.connect(self.db_path) as conn:
            # Collections table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS collections (
                    name TEXT PRIMARY KEY,
                    description TEXT,
                    embedding_model TEXT,
                    embedding_dimensions INTEGER,
                    document_count INTEGER DEFAULT 0,
                    index_type TEXT DEFAULT 'IndexFlatIP',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Documents table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    collection_name TEXT,
                    content TEXT,
                    metadata TEXT,
                    embedding_id INTEGER,
                    chunk_ids TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (collection_name) REFERENCES collections (name)
                )
            """)
            
            # Chunks table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT,
                    collection_name TEXT,
                    content TEXT,
                    metadata TEXT,
                    embedding_id INTEGER,
                    chunk_index INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (document_id) REFERENCES documents (id),
                    FOREIGN KEY (collection_name) REFERENCES collections (name)
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_collection ON documents (collection_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks (document_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_collection ON chunks (collection_name)")
    
    def _get_collection_name(self, **kwargs) -> str:
        """Get collection name from kwargs or use default."""
        return kwargs.get("collection_name", self.config.collection_name or "default")
    
    def _get_index_path(self, collection_name: str) -> Path:
        """Get FAISS index path for collection."""
        return self.data_dir / f"{collection_name}.faiss"
    
    def _load_index(self, collection_name: str) -> Optional[faiss.Index]:
        """Load FAISS index for collection."""
        if collection_name in self.index_cache:
            return self.index_cache[collection_name]
        
        index_path = self._get_index_path(collection_name)
        if index_path.exists():
            index = faiss.read_index(str(index_path))
            self.index_cache[collection_name] = index
            return index
        
        return None
    
    def _save_index(self, collection_name: str, index: faiss.Index):
        """Save FAISS index for collection."""
        index_path = self._get_index_path(collection_name)
        faiss.write_index(index, str(index_path))
        self.index_cache[collection_name] = index
    
    def _create_index(self, dimensions: int, collection_name: str) -> faiss.Index:
        """Create new FAISS index."""
        index_type = self.config.index_type or "IndexFlatIP"
        
        if index_type == "IndexFlatIP":
            index = faiss.IndexFlatIP(dimensions)
        elif index_type == "IndexFlatL2":
            index = faiss.IndexFlatL2(dimensions)
        elif index_type == "IndexHNSWFlat":
            index = faiss.IndexHNSWFlat(dimensions, 32)
        else:
            # Default to IndexFlatIP
            index = faiss.IndexFlatIP(dimensions)
        
        return index
    
    async def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """Add documents to the vector store."""
        collection_name = self._get_collection_name(**kwargs)
        
        if not documents:
            return []
        
        # Generate embeddings for documents
        texts = [doc.content for doc in documents]
        embeddings = await self.embeddings.vectorize_documents(texts)
        
        if not embeddings:
            raise RuntimeError("Failed to generate embeddings")
        
        dimensions = len(embeddings[0])
        
        # Load or create index
        index = self._load_index(collection_name)
        if index is None:
            index = self._create_index(dimensions, collection_name)
        
        # Add embeddings to index
        embedding_ids = []
        for embedding in embeddings:
            embedding_id = index.ntotal
            index.add(np.array([embedding], dtype=np.float32))
            embedding_ids.append(embedding_id)
        
        # Save index
        self._save_index(collection_name, index)
        
        # Save to SQLite
        with sqlite3.connect(self.db_path) as conn:
            # Update or create collection
            conn.execute("""
                INSERT OR REPLACE INTO collections 
                (name, description, embedding_model, embedding_dimensions, document_count, index_type, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                collection_name,
                f"Collection with {len(documents)} documents",
                self.embeddings.config.model_name,
                dimensions,
                index.ntotal,
                self.config.index_type or "IndexFlatIP"
            ))
            
            # Insert documents
            document_ids = []
            for doc, embedding_id in zip(documents, embedding_ids):
                # Process chunks if any
                chunk_ids = []
                if doc.chunks:
                    for chunk in doc.chunks:
                        chunk_ids.append(chunk.id)
                
                conn.execute("""
                    INSERT OR REPLACE INTO documents 
                    (id, collection_name, content, metadata, embedding_id, chunk_ids)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    doc.id,
                    collection_name,
                    doc.content,
                    json.dumps(doc.metadata),
                    embedding_id,
                    json.dumps(chunk_ids)
                ))
                
                # Insert chunks
                for chunk in doc.chunks:
                    conn.execute("""
                        INSERT OR REPLACE INTO chunks 
                        (id, document_id, collection_name, content, metadata, embedding_id, chunk_index)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        chunk.id,
                        doc.id,
                        collection_name,
                        chunk.content,
                        json.dumps(chunk.metadata),
                        embedding_id,  # Same embedding for document and chunks
                        chunk.metadata.get("chunk_index", 0)
                    ))
                
                document_ids.append(doc.id)
        
        return document_ids
    
    async def add_document_chunks(self, chunks: List[Chunk], **kwargs: Any) -> List[str]:
        """Add document chunks to the vector store."""
        collection_name = self._get_collection_name(**kwargs)
        
        if not chunks:
            return []
        
        # Generate embeddings for chunks
        texts = [chunk.content for chunk in chunks]
        embeddings = await self.embeddings.vectorize_documents(texts)
        
        if not embeddings:
            raise RuntimeError("Failed to generate embeddings")
        
        dimensions = len(embeddings[0])
        
        # Load or create index
        index = self._load_index(collection_name)
        if index is None:
            index = self._create_index(dimensions, collection_name)
        
        # Add embeddings to index
        embedding_ids = []
        for embedding in embeddings:
            embedding_id = index.ntotal
            index.add(np.array([embedding], dtype=np.float32))
            embedding_ids.append(embedding_id)
        
        # Save index
        self._save_index(collection_name, index)
        
        # Save to SQLite
        with sqlite3.connect(self.db_path) as conn:
            # Update collection
            conn.execute("""
                INSERT OR REPLACE INTO collections 
                (name, description, embedding_model, embedding_dimensions, document_count, index_type, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                collection_name,
                f"Collection with {len(chunks)} chunks",
                self.embeddings.config.model_name,
                dimensions,
                index.ntotal,
                self.config.index_type or "IndexFlatIP"
            ))
            
            # Insert chunks
            chunk_ids = []
            for chunk, embedding_id in zip(chunks, embedding_ids):
                conn.execute("""
                    INSERT OR REPLACE INTO chunks 
                    (id, document_id, collection_name, content, metadata, embedding_id, chunk_index)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    chunk.id,
                    chunk.document_id,
                    collection_name,
                    chunk.content,
                    json.dumps(chunk.metadata),
                    embedding_id,
                    chunk.metadata.get("chunk_index", 0)
                ))
                
                chunk_ids.append(chunk.id)
        
        return chunk_ids
    
    async def add_document_text_chunks(self, document_texts: List[str], **kwargs: Any) -> List[str]:
        """Add document text chunks to the vector store (legacy method)."""
        # Convert texts to chunks
        chunks = []
        for i, text in enumerate(document_texts):
            chunk = Chunk(
                id=f"chunk_{i}_{hashlib.md5(text.encode()).hexdigest()[:8]}",
                content=text,
                document_id=kwargs.get("document_id", f"doc_{i}"),
                metadata={"chunk_index": i}
            )
            chunks.append(chunk)
        
        return await self.add_document_chunks(chunks, **kwargs)
    
    async def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from the vector store."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get document info
                cursor = conn.execute("""
                    SELECT collection_name, embedding_id FROM documents 
                    WHERE id IN ({})
                """.format(','.join('?' * len(document_ids))), document_ids)
                
                documents_to_delete = cursor.fetchall()
                
                if not documents_to_delete:
                    return True
                
                # Group by collection
                collections = {}
                for collection_name, embedding_id in documents_to_delete:
                    if collection_name not in collections:
                        collections[collection_name] = []
                    collections[collection_name].append(embedding_id)
                
                # Remove from FAISS indices
                for collection_name, embedding_ids in collections.items():
                    index = self._load_index(collection_name)
                    if index:
                        # FAISS doesn't support direct deletion, so we need to rebuild
                        # This is a simplified approach - in practice, you'd want more sophisticated handling
                        pass
                
                # Delete from SQLite
                conn.execute("""
                    DELETE FROM chunks WHERE document_id IN ({})
                """.format(','.join('?' * len(document_ids))), document_ids)
                
                conn.execute("""
                    DELETE FROM documents WHERE id IN ({})
                """.format(','.join('?' * len(document_ids))), document_ids)
                
                # Update collection counts
                for collection_name in collections.keys():
                    conn.execute("""
                        UPDATE collections SET document_count = (
                            SELECT COUNT(*) FROM documents WHERE collection_name = ?
                        ), updated_at = CURRENT_TIMESTAMP WHERE name = ?
                    """, (collection_name, collection_name))
            
            return True
            
        except Exception as e:
            print(f"Error deleting documents: {e}")
            return False
    
    async def search(
        self, 
        query: str, 
        search_type: SearchType, 
        retrieval_query: Optional[str] = None,
        **kwargs: Any
    ) -> List[SearchResult]:
        """Search for documents using text query."""
        collection_name = self._get_collection_name(**kwargs)
        top_k = kwargs.get("top_k", 5)
        score_threshold = kwargs.get("score_threshold")
        filters = kwargs.get("filters", {})
        
        # Generate query embedding
        query_embedding = await self.embeddings.vectorize_query(query)
        
        return await self.search_with_embeddings(
            query_embedding, search_type, retrieval_query,
            collection_name=collection_name,
            top_k=top_k,
            score_threshold=score_threshold,
            filters=filters
        )
    
    async def search_with_embeddings(
        self, 
        query_embedding: List[float], 
        search_type: SearchType, 
        retrieval_query: Optional[str] = None,
        **kwargs: Any
    ) -> List[SearchResult]:
        """Search for documents using embedding vector."""
        collection_name = self._get_collection_name(**kwargs)
        top_k = kwargs.get("top_k", 5)
        score_threshold = kwargs.get("score_threshold")
        filters = kwargs.get("filters", {})
        
        # Load index
        index = self._load_index(collection_name)
        if index is None:
            return []
        
        # Search
        scores, indices = index.search(np.array([query_embedding], dtype=np.float32), top_k)
        
        # Get documents from SQLite
        results = []
        with sqlite3.connect(self.db_path) as conn:
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # No more results
                    break
                
                if score_threshold and score < score_threshold:
                    continue
                
                # Get document info
                cursor = conn.execute("""
                    SELECT id, content, metadata FROM documents 
                    WHERE collection_name = ? AND embedding_id = ?
                """, (collection_name, int(idx)))
                
                row = cursor.fetchone()
                if row:
                    doc_id, content, metadata = row
                    metadata_dict = json.loads(metadata) if metadata else {}
                    
                    # Apply filters
                    if filters and not self._matches_filters(metadata_dict, filters):
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
        
        return results
    
    async def get_document(self, document_id: str) -> Optional[Document]:
        """Retrieve a document by its ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT id, content, metadata, chunk_ids FROM documents WHERE id = ?
            """, (document_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            doc_id, content, metadata, chunk_ids = row
            metadata_dict = json.loads(metadata) if metadata else {}
            chunk_id_list = json.loads(chunk_ids) if chunk_ids else []
            
            # Get chunks
            chunks = []
            if chunk_id_list:
                cursor = conn.execute("""
                    SELECT id, content, metadata FROM chunks WHERE id IN ({})
                """.format(','.join('?' * len(chunk_id_list))), chunk_id_list)
                
                for chunk_row in cursor.fetchall():
                    chunk_id, chunk_content, chunk_metadata = chunk_row
                    chunk_metadata_dict = json.loads(chunk_metadata) if chunk_metadata else {}
                    
                    chunk = Chunk(
                        id=chunk_id,
                        content=chunk_content,
                        document_id=doc_id,
                        metadata=chunk_metadata_dict
                    )
                    chunks.append(chunk)
            
            return Document(
                id=doc_id,
                content=content,
                chunks=chunks,
                metadata=metadata_dict
            )
    
    async def update_document(self, document: Document) -> bool:
        """Update an existing document."""
        try:
            # Delete old document
            await self.delete_documents([document.id])
            
            # Add updated document
            await self.add_documents([document])
            
            return True
            
        except Exception as e:
            print(f"Error updating document: {e}")
            return False
    
    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if metadata matches filters."""
        for key, value in filters.items():
            if key not in metadata:
                return False
            
            # Handle different filter types
            if isinstance(value, dict):
                if "$in" in value:
                    if metadata[key] not in value["$in"]:
                        return False
                elif "$gte" in value:
                    if metadata[key] < value["$gte"]:
                        return False
                elif "$lte" in value:
                    if metadata[key] > value["$lte"]:
                        return False
                elif "$gt" in value:
                    if metadata[key] <= value["$gt"]:
                        return False
                elif "$lt" in value:
                    if metadata[key] >= value["$lt"]:
                        return False
                else:
                    if metadata[key] != value:
                        return False
            else:
                if metadata[key] != value:
                    return False
        
        return True
    
    async def get_collection_info(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a collection."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT name, description, embedding_model, embedding_dimensions, 
                       document_count, index_type, created_at, updated_at
                FROM collections WHERE name = ?
            """, (collection_name,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            return {
                "name": row[0],
                "description": row[1],
                "embedding_model": row[2],
                "embedding_dimensions": row[3],
                "document_count": row[4],
                "index_type": row[5],
                "created_at": row[6],
                "updated_at": row[7]
            }
    
    async def list_collections(self) -> List[Dict[str, Any]]:
        """List all collections."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT name, description, embedding_model, embedding_dimensions, 
                       document_count, index_type, created_at, updated_at
                FROM collections ORDER BY created_at DESC
            """)
            
            collections = []
            for row in cursor.fetchall():
                collections.append({
                    "name": row[0],
                    "description": row[1],
                    "embedding_model": row[2],
                    "embedding_dimensions": row[3],
                    "document_count": row[4],
                    "index_type": row[5],
                    "created_at": row[6],
                    "updated_at": row[7]
                })
            
            return collections
    
    async def clear_collection(self, collection_name: str) -> bool:
        """Clear all documents from a collection."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Delete chunks
                conn.execute("DELETE FROM chunks WHERE collection_name = ?", (collection_name,))
                
                # Delete documents
                conn.execute("DELETE FROM documents WHERE collection_name = ?", (collection_name,))
                
                # Update collection
                conn.execute("""
                    UPDATE collections SET document_count = 0, updated_at = CURRENT_TIMESTAMP 
                    WHERE name = ?
                """, (collection_name,))
            
            # Remove index from cache and delete file
            if collection_name in self.index_cache:
                del self.index_cache[collection_name]
            
            index_path = self._get_index_path(collection_name)
            if index_path.exists():
                index_path.unlink()
            
            return True
            
        except Exception as e:
            print(f"Error clearing collection: {e}")
            return False


class VectorStoreFactory:
    """Factory for creating vector store instances."""
    
    @staticmethod
    def create_vector_store(
        config: VectorStoreConfig, 
        embeddings: Embeddings
    ) -> VectorStore:
        """Create a vector store instance based on configuration."""
        if config.store_type == VectorStoreType.FAISS:
            return FAISSVectorStore(config, embeddings)
        else:
            raise ValueError(f"Unsupported vector store type: {config.store_type}")
    
    @staticmethod
    def create_faiss_vector_store(
        data_dir: str = "./data/vector_store",
        collection_name: str = "default",
        index_type: str = "IndexFlatIP",
        embeddings: Optional[Embeddings] = None
    ) -> FAISSVectorStore:
        """Create a FAISS vector store with default configuration."""
        config = VectorStoreConfig(
            store_type=VectorStoreType.FAISS,
            connection_string=data_dir,
            collection_name=collection_name,
            index_type=index_type,
            embedding_dimension=embeddings.num_dimensions if embeddings else 384
        )
        
        if embeddings is None:
            # Create default embeddings
            from .vllm_integration import VLLMEmbeddings
            from .rag import EmbeddingsConfig, EmbeddingModelType
            
            embeddings_config = EmbeddingsConfig(
                model_type=EmbeddingModelType.CUSTOM,
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                num_dimensions=384
            )
            embeddings = VLLMEmbeddings(embeddings_config)
        
        return FAISSVectorStore(config, embeddings)


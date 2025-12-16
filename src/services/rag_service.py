"""
RAG Service - Handles document processing, embeddings, and vector search
"""

import logging
import os
import json
import re
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGService:
    """
    RAG Service for document processing and vector search
    
    Features:
    - Document chunking with overlap
    - Embedding generation using sentence-transformers
    - FAISS local indexing
    - Qdrant cloud/local storage
    - Hybrid search (semantic + keyword)
    """
    
    def __init__(self, 
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 chunk_size: int = 512,
                 chunk_overlap: int = 50,
                 qdrant_url: Optional[str] = None,
                 qdrant_api_key: Optional[str] = None):
        """
        Initialize RAG Service
        
        Args:
            embedding_model: HuggingFace model name for embeddings
            chunk_size: Number of tokens per chunk
            chunk_overlap: Overlap tokens between chunks
            qdrant_url: Qdrant server URL (optional)
            qdrant_api_key: Qdrant API key (optional)
        """
        logger.info("Initializing RAG Service...")
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model_name = embedding_model
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")
        
        # Initialize FAISS index (local storage)
        self.faiss_index = None
        self.faiss_metadata = {}  # Store metadata for FAISS results
        # Use absolute path
        project_root = Path(__file__).parent.parent.parent
        self.faiss_index_path = project_root / "data" / "faiss_index"
        self.faiss_index_path.mkdir(parents=True, exist_ok=True)
        self._load_faiss_index()
        
        # Initialize Qdrant client
        self.qdrant_client = None
        if qdrant_url:
            logger.info(f"Connecting to Qdrant at {qdrant_url}")
            self.qdrant_client = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key
            )
        else:
            logger.info("Using Qdrant in-memory mode")
            self.qdrant_client = QdrantClient(":memory:")
        
        self.qdrant_collection_name = "financial_documents"
        self._init_qdrant_collection()
        
        logger.info("RAG Service initialized successfully!")
    
    def _load_faiss_index(self):
        """Load FAISS index from disk if exists, otherwise create new"""
        index_file = self.faiss_index_path / "index.faiss"
        metadata_file = self.faiss_index_path / "metadata.json"
        
        if index_file.exists() and metadata_file.exists():
            try:
                logger.info("Loading existing FAISS index...")
                self.faiss_index = faiss.read_index(str(index_file))
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    self.faiss_metadata = json.load(f)
                logger.info(f"Loaded FAISS index with {self.faiss_index.ntotal} vectors")
            except Exception as e:
                logger.warning(f"Failed to load FAISS index: {e}, creating new one")
                self._create_faiss_index()
        else:
            self._create_faiss_index()
    
    def _create_faiss_index(self):
        """Create new FAISS index"""
        logger.info("Creating new FAISS index...")
        self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
        self.faiss_metadata = {}
    
    def _save_faiss_index(self):
        """Save FAISS index to disk"""
        try:
            index_file = self.faiss_index_path / "index.faiss"
            metadata_file = self.faiss_index_path / "metadata.json"
            
            faiss.write_index(self.faiss_index, str(index_file))
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.faiss_metadata, f, ensure_ascii=False, indent=2)
            
            logger.info("FAISS index saved to disk")
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
    
    def _init_qdrant_collection(self):
        """Initialize Qdrant collection"""
        try:
            # Check if collection exists
            try:
                self.qdrant_client.get_collection(self.qdrant_collection_name)
                logger.info(f"Using existing Qdrant collection: {self.qdrant_collection_name}")
            except Exception:
                # Collection doesn't exist, create it
                logger.info(f"Creating new Qdrant collection: {self.qdrant_collection_name}")
                self.qdrant_client.create_collection(
                    collection_name=self.qdrant_collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant collection: {e}")
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Text to chunk
            metadata: Metadata to attach to each chunk
            
        Returns:
            List of chunks with metadata
        """
        # Split by sentences first for better boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            words = sentence.split()
            sentence_length = len(words)
            
            # If adding this sentence exceeds chunk size, save current chunk
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'metadata': metadata or {},
                    'length': current_length
                })
                
                # Start new chunk with overlap
                overlap_sentences = []
                overlap_length = 0
                for prev_sentence in reversed(current_chunk):
                    prev_length = len(prev_sentence.split())
                    if overlap_length + prev_length <= self.chunk_overlap:
                        overlap_sentences.insert(0, prev_sentence)
                        overlap_length += prev_length
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_length = overlap_length
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'metadata': metadata or {},
                'length': current_length
            })
        
        logger.info(f"Text chunked into {len(chunks)} chunks")
        return chunks
    
    def add_document(self, 
                    doc_id: str,
                    text: str,
                    title: str = "",
                    source: str = "",
                    user_id: str = "") -> int:
        """
        Add document to both FAISS and Qdrant
        
        Args:
            doc_id: Unique document ID
            text: Document text content
            title: Document title
            source: Document source/filename
            user_id: User who uploaded the document
            
        Returns:
            Number of chunks added
        """
        logger.info(f"Adding document {doc_id}: {title}")
        
        # Chunk the document
        chunks = self.chunk_text(text, {
            'doc_id': doc_id,
            'title': title,
            'source': source,
            'user_id': user_id,
            'added_at': datetime.now().isoformat()
        })
        
        # Generate embeddings
        chunk_texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embedding_model.encode(chunk_texts, convert_to_numpy=True)
        
        # Add to FAISS
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{doc_id}_{i}"
            chunk_meta = {
                **chunk['metadata'],
                'text': chunk['text'],  # Store the chunk text
                'chunk_id': chunk_id,
                'chunk_index': i
            }
            
            # Add to FAISS
            self.faiss_index.add(np.array([embedding], dtype=np.float32))
            self.faiss_metadata[str(self.faiss_index.ntotal - 1)] = chunk_meta
            
            # Add to Qdrant
            point = PointStruct(
                id=hash(chunk_id) % (2**31),  # Convert to valid ID
                vector=embedding.tolist(),
                payload=chunk_meta
            )
            try:
                self.qdrant_client.upsert(
                    collection_name=self.qdrant_collection_name,
                    points=[point]
                )
            except Exception as e:
                logger.warning(f"Failed to add to Qdrant: {e}")
        
        # Save FAISS index
        self._save_faiss_index()
        
        logger.info(f"Added {len(chunks)} chunks from document {doc_id}")
        return len(chunks)
    
    def search(self, 
               query: str,
               top_k: int = 5,
               user_id: str = "") -> List[Dict]:
        """
        Search for relevant document chunks
        
        Args:
            query: Search query
            top_k: Number of results to return
            user_id: Filter by user (optional)
            
        Returns:
            List of relevant chunks with similarity scores
        """
        logger.info(f"Searching for: {query}")
        
        if not self.faiss_index or self.faiss_index.ntotal == 0:
            logger.warning("FAISS index is empty")
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
        
        # Search FAISS
        distances, indices = self.faiss_index.search(
            np.array([query_embedding], dtype=np.float32),
            min(top_k * 2, self.faiss_index.ntotal)  # Get more to filter by user
        )
        
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx == -1:  # Invalid index
                continue
            
            meta = self.faiss_metadata.get(str(idx))
            if not meta:
                continue
            
            # Filter by user if specified (only filter if user_id is provided and non-empty)
            if user_id:
                doc_user = meta.get('user_id', '')
                if doc_user != user_id:
                    continue
            
            # Convert L2 distance to similarity (lower distance = higher similarity)
            similarity = 1 / (1 + distance)
            
            results.append({
                'text': meta.get('text', ''),
                'title': meta.get('title', ''),
                'source': meta.get('source', ''),
                'doc_id': meta.get('doc_id', ''),
                'chunk_id': meta.get('chunk_id', ''),
                'similarity': float(similarity),
                'metadata': meta
            })
            
            if len(results) >= top_k:
                break
        
        logger.info(f"Found {len(results)} relevant chunks")
        return results
    
    def remove_document(self, doc_id: str) -> bool:
        """
        Remove all chunks of a document from indexes
        
        Args:
            doc_id: Document ID to remove
            
        Returns:
            Success status
        """
        logger.info(f"Removing document {doc_id}")
        
        try:
            # Remove from FAISS (rebuild index without this document)
            to_keep = []
            for idx_str, meta in self.faiss_metadata.items():
                if meta.get('doc_id') != doc_id:
                    to_keep.append(int(idx_str))
            
            if to_keep:
                # Reconstruct index without deleted document
                new_index = faiss.IndexFlatL2(self.embedding_dim)
                new_metadata = {}
                
                for new_idx, old_idx in enumerate(to_keep):
                    old_vector = self.faiss_index.reconstruct(old_idx)
                    new_index.add(np.array([old_vector], dtype=np.float32))
                    new_metadata[str(new_idx)] = self.faiss_metadata[str(old_idx)]
                
                self.faiss_index = new_index
                self.faiss_metadata = new_metadata
            else:
                self._create_faiss_index()
            
            self._save_faiss_index()
            
            # Remove from Qdrant
            try:
                self.qdrant_client.delete(
                    collection_name=self.qdrant_collection_name,
                    points_selector={
                        "filter": {
                            "must": [
                                {
                                    "key": "doc_id",
                                    "match": {"value": doc_id}
                                }
                            ]
                        }
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to remove from Qdrant: {e}")
            
            logger.info(f"Document {doc_id} removed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove document {doc_id}: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """Get RAG service statistics"""
        return {
            'embedding_model': self.embedding_model_name,
            'embedding_dimension': self.embedding_dim,
            'faiss_vectors': self.faiss_index.ntotal if self.faiss_index else 0,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap
        }
    
    def get_document_chunks(self, doc_id: str) -> List[Dict]:
        """
        Get all chunks for a specific document
        
        Args:
            doc_id: Document ID
            
        Returns:
            List of chunk dictionaries with text, index, and token count
        """
        chunks = []
        chunk_idx = 0
        
        for idx_str, meta in self.faiss_metadata.items():
            if meta.get('doc_id') == doc_id:
                chunk_text = meta.get('text', '')
                token_count = len(chunk_text.split())  # Simple token approximation
                
                chunks.append({
                    'text': chunk_text,
                    'index': chunk_idx,
                    'token_count': token_count,
                    'source': meta.get('source', ''),
                    'title': meta.get('title', '')
                })
                chunk_idx += 1
        
        logger.info(f"Retrieved {len(chunks)} chunks for document {doc_id}")
        return chunks
    
    def regenerate_embeddings(self, doc_id: str) -> Tuple[bool, str, int]:
        """
        Regenerate embeddings for a document
        Removes old embeddings and re-embeds the chunks
        
        Args:
            doc_id: Document ID
            
        Returns:
            Tuple of (success, message, new_chunk_count)
        """
        try:
            # Get the old chunks
            old_chunks = self.get_document_chunks(doc_id)
            if not old_chunks:
                return False, "Document not found", 0
            
            # Remove old embeddings
            self.remove_document(doc_id)
            
            # Re-embed the chunks
            chunks_to_add = [
                {
                    'text': chunk['text'],
                    'source': chunk['source'],
                    'title': chunk['title']
                }
                for chunk in old_chunks
            ]
            
            # Add back with new embeddings
            embeddings = self.embedding_model.encode(
                [c['text'] for c in chunks_to_add],
                convert_to_numpy=True
            )
            
            new_idx_start = self.faiss_index.ntotal if self.faiss_index else 0
            
            # Add to FAISS
            self.faiss_index.add(embeddings.astype(np.float32))
            
            # Add metadata
            for i, chunk in enumerate(chunks_to_add):
                idx = new_idx_start + i
                self.faiss_metadata[str(idx)] = {
                    'doc_id': doc_id,
                    'text': chunk['text'],
                    'source': chunk['source'],
                    'title': chunk['title'],
                    'chunk_index': i
                }
            
            # Save FAISS index
            self._save_faiss_index()
            
            logger.info(f"Regenerated embeddings for document {doc_id} ({len(chunks_to_add)} chunks)")
            return True, f"Embeddings regenerated successfully", len(chunks_to_add)
            
        except Exception as e:
            logger.error(f"Failed to regenerate embeddings for {doc_id}: {e}")
            return False, f"Error regenerating embeddings: {str(e)}", 0


# Global RAG service instance
_rag_service = None


def get_rag_service() -> RAGService:
    """Get or create RAG service singleton"""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service

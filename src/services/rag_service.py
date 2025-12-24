"""
RAG Service - Handles document processing, embeddings, and vector search
QDRANT-FIRST ARCHITECTURE: Qdrant is primary vector database (Cloud or Local)
"""

import logging
import os
import json
import re
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime
import uuid

import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, PointIdsList
from qdrant_client import models

# Import settings
from ..core.config import settings

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGService:
    """
    RAG Service for document processing and vector search
    
    QDRANT-FIRST ARCHITECTURE:
    - Primary: Qdrant for all vector operations
    - Features:
      - Document chunking with overlap
      - Embedding generation using sentence-transformers
      - Qdrant cloud/local vector storage (PRIMARY)
      - Semantic search with similarity scoring
      - User isolation and privacy filtering
    """
    
    def __init__(self, 
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 chunk_size: int = 512,
                 chunk_overlap: int = 50,
                 qdrant_url: Optional[str] = None,
                 qdrant_api_key: Optional[str] = None):
        """
        Initialize RAG Service with Qdrant as primary vector database
        
        Args:
            embedding_model: HuggingFace model name for embeddings
            chunk_size: Number of tokens per chunk
            chunk_overlap: Overlap tokens between chunks
            qdrant_url: Qdrant server URL (optional, uses config if not provided)
                       Examples:
                       - "https://your-instance.qdrant.io" (Qdrant Cloud)
                       - "http://localhost:6333" (local)
                       - None (uses QDRANT_MODE from config)
            qdrant_api_key: Qdrant API key (optional, uses config if not provided)
        """
        logger.info("Initializing RAG Service (Qdrant-First Architecture)...")
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model_name = embedding_model
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")
        
        # Determine Qdrant connection parameters
        # Priority: 1) params provided 2) config settings 3) fallback to in-memory
        if qdrant_url is None:
            # Use cloud or local settings from config
            if settings.QDRANT_MODE == "cloud" and settings.QDRANT_CLOUD_URL:
                qdrant_url = settings.QDRANT_CLOUD_URL
                qdrant_api_key = settings.QDRANT_CLOUD_API_KEY
                logger.info("Using Qdrant Cloud from config")
            elif settings.QDRANT_URL:
                # Fallback to legacy QDRANT_URL setting
                qdrant_url = settings.QDRANT_URL
                qdrant_api_key = settings.QDRANT_API_KEY
                logger.info("Using legacy QDRANT_URL from config")
        
        # Initialize Qdrant client (PRIMARY vector database)
        logger.info(f"Connecting to Qdrant: {qdrant_url or 'in-memory mode (development)'}")
        try:
            if qdrant_url:
                # Production: Connect to remote Qdrant server (Cloud or Local)
                self.qdrant_client = QdrantClient(
                    url=qdrant_url,
                    api_key=qdrant_api_key
                )
                mode = "Cloud" if settings.QDRANT_MODE == "cloud" else "Remote"
                logger.info(f"✓ Connected to Qdrant {mode} at {qdrant_url}")
            else:
                # Development: In-memory Qdrant
                self.qdrant_client = QdrantClient(":memory:")
                logger.info("✓ Using Qdrant in-memory mode (development only)")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise
        
        self.qdrant_collection_name = "financial_documents"
        self._init_qdrant_collection()
        self._init_next_point_id()  # Initialize next_point_id after collection exists
        
        logger.info("✓ RAG Service initialized successfully (Qdrant-First Architecture)")
    
    def _init_qdrant_collection(self):
        """Initialize Qdrant collection for vector storage"""
        try:
            # Check if collection exists
            try:
                collection_info = self.qdrant_client.get_collection(self.qdrant_collection_name)
                logger.info(f"✓ Using existing Qdrant collection: {self.qdrant_collection_name}")
                logger.info(f"  Vectors in collection: {collection_info.points_count}")
            except Exception:
                # Collection doesn't exist, create it
                logger.info(f"Creating new Qdrant collection: {self.qdrant_collection_name}")
                self.qdrant_client.create_collection(
                    collection_name=self.qdrant_collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE  # Cosine similarity
                    )
                )
                # Create payload index for faster filtering
                self.qdrant_client.create_payload_index(
                    collection_name=self.qdrant_collection_name,
                    field_name="user_id",
                    field_schema="keyword"
                )
                self.qdrant_client.create_payload_index(
                    collection_name=self.qdrant_collection_name,
                    field_name="doc_id",
                    field_schema="keyword"
                )
                logger.info(f"✓ Created new Qdrant collection: {self.qdrant_collection_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant collection: {e}")
            raise
    
    def _init_next_point_id(self):
        """Initialize next_point_id by getting the maximum ID from existing points"""
        try:
            # Get collection info first
            collection_info = self.qdrant_client.get_collection(self.qdrant_collection_name)
            point_count = collection_info.points_count
            logger.info(f"Collection has {point_count} points")
            
            if point_count == 0:
                self.next_point_id = 1
                logger.info("Collection is empty, initializing next_point_id to 1")
                return
            
            # Scroll through all points to find the maximum ID
            max_id = 0
            points_scanned = 0
            offset = None
            
            while True:
                # scroll() returns a tuple: (points, next_page_offset)
                scroll_result = self.qdrant_client.scroll(
                    collection_name=self.qdrant_collection_name,
                    offset=offset,
                    limit=100,
                    with_payload=False
                )
                
                points, next_offset = scroll_result
                
                if not points:
                    break
                
                points_scanned += len(points)
                current_max = max(point.id for point in points)
                max_id = max(max_id, current_max)
                logger.debug(f"Scanned {points_scanned} points, current max ID: {max_id}")
                
                # Check if there are more points to scroll
                if next_offset is None:
                    break
                    
                offset = next_offset
            
            self.next_point_id = max_id + 1
            logger.info(f"✓ Initialized next_point_id to {self.next_point_id} (scanned {points_scanned} points, max ID: {max_id})")
        except Exception as e:
            logger.error(f"Error determining max point ID: {e}")
            self.next_point_id = 1
            logger.info("Fallback: Starting next_point_id from 1")
    
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
        Add document to Qdrant vector database
        
        Args:
            doc_id: Unique document ID
            text: Document text content
            title: Document title
            source: Document source/filename
            user_id: User who uploaded the document (for privacy filtering)
            
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
        
        if not chunks:
            logger.warning(f"No chunks generated for document {doc_id}")
            return 0
        
        # Generate embeddings for all chunks
        chunk_texts = [chunk['text'] for chunk in chunks]
        logger.info(f"Generating embeddings for {len(chunk_texts)} chunks...")
        embeddings = self.embedding_model.encode(chunk_texts, convert_to_numpy=True)
        
        # Add to Qdrant
        points = []
        point_ids = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{doc_id}_{i}"
            
            # Create point for Qdrant
            point = PointStruct(
                id=self.next_point_id,  # Unique sequential ID
                vector=embedding.tolist(),
                payload={
                    **chunk['metadata'],
                    'text': chunk['text'],  # Store chunk text as payload
                    'chunk_id': chunk_id,
                    'chunk_index': i
                }
            )
            points.append(point)
            point_ids.append(self.next_point_id)
            self.next_point_id += 1
        
        # Batch upsert to Qdrant (more efficient)
        try:
            logger.info(f"Upserting {len(points)} points with IDs: {point_ids}")
            self.qdrant_client.upsert(
                collection_name=self.qdrant_collection_name,
                points=points
            )
            logger.info(f"✓ Added {len(chunks)} chunks to Qdrant for document {doc_id} with point IDs: {point_ids}")
        except Exception as e:
            logger.error(f"Failed to add document {doc_id} to Qdrant: {e}")
            raise
        
        return len(chunks)
    
    def search(self, 
               query: str,
               top_k: int = 5,
               user_id: str = "",
               session_id: Optional[str] = None,
               include_global: bool = True) -> List[Dict]:
        """
        Search for relevant document chunks using Qdrant
        Prioritizes user's personal collection, optionally falls back to global
        
        Args:
            query: Search query string
            top_k: Number of results to return (default: 5, max: 50)
            user_id: Filter by user ID (for privacy isolation)
            session_id: Conversation session ID for isolation (optional)
            include_global: Fall back to global docs if personal results empty
            
        Returns:
            List of relevant chunks with similarity scores, sorted by relevance
        """
        logger.info(f"Searching Qdrant: '{query}' (top_k={top_k}, user_id={user_id}, session_id={session_id})")
        
        if not self.qdrant_client:
            logger.error("Qdrant client not initialized")
            return []
        
        try:
            # Check if collection has any data
            collection_info = self.qdrant_client.get_collection(self.qdrant_collection_name)
            if collection_info.points_count == 0:
                logger.warning("Qdrant collection is empty")
                return []
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
            
            # Build filter for personal collection search
            personal_filter = None
            if user_id:
                must_conditions = [
                    models.FieldCondition(
                        key="user_id",
                        match=models.MatchValue(value=user_id)
                    )
                ]
                # If session_id provided, isolate to this conversation
                if session_id:
                    must_conditions.append(
                        models.FieldCondition(
                            key="chat_session_id",
                            match=models.MatchValue(value=session_id)
                        )
                    )
                personal_filter = models.Filter(must=must_conditions)
            
            # Search personal collection first
            logger.info(f"Searching personal collection (session: {session_id})")
            personal_result = self.qdrant_client.query_points(
                collection_name=self.qdrant_collection_name,
                query=query_embedding.tolist(),
                query_filter=personal_filter,
                limit=top_k,
                with_payload=True
            )
            
            personal_results = personal_result.points if personal_result and personal_result.points else []
            logger.info(f"Found {len(personal_results)} results in personal collection")
            
            # Fall back to global if no personal results and include_global=True
            global_results = []
            if not personal_results and include_global:
                logger.info("No personal results found, searching global collection")
                global_result = self.qdrant_client.query_points(
                    collection_name=self.qdrant_collection_name,
                    query=query_embedding.tolist(),
                    query_filter=None,  # No user filter for global search
                    limit=top_k,
                    with_payload=True
                )
                global_results = global_result.points if global_result and global_result.points else []
                logger.info(f"Found {len(global_results)} results in global collection")
            
            # Use personal results if available, otherwise global
            search_points = personal_results if personal_results else global_results
            
            # Format results
            results = []
            for scored_point in search_points:
                payload = scored_point.payload
                text = payload.get('text', '')
                
                # Create RAG document summary with metrics extraction
                try:
                    from ..utils.summarization import create_rag_summary
                    doc_summary = create_rag_summary(
                        text,
                        float(scored_point.score) if hasattr(scored_point, 'score') else 0.0
                    )
                    
                    results.append({
                        'text': doc_summary.get('full_text', text),
                        'title': payload.get('title', ''),
                        'source': payload.get('source', ''),
                        'doc_id': payload.get('doc_id', ''),
                        'chunk_id': payload.get('chunk_id', ''),
                        'similarity': doc_summary.get('relevance_score', 0.0),
                        'metrics': doc_summary.get('metrics_summary', {}),
                        'summary': doc_summary.get('summary', None),
                        'metadata': payload
                    })
                except Exception as sum_error:
                    logger.warning(f"RAG summarization failed for doc {payload.get('doc_id')}: {sum_error}")
                    # Fallback: return result without summary
                    results.append({
                        'text': text,
                        'title': payload.get('title', ''),
                        'source': payload.get('source', ''),
                        'doc_id': payload.get('doc_id', ''),
                        'chunk_id': payload.get('chunk_id', ''),
                        'similarity': float(scored_point.score) if hasattr(scored_point, 'score') else 0.0,
                        'metrics': {},
                        'summary': None,
                        'metadata': payload
                    })
            
            logger.info(f"✓ Found {len(results)} relevant chunks")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    
    def remove_document(self, doc_id: str) -> bool:
        """
        Remove all chunks of a document from Qdrant
        
        Args:
            doc_id: Document ID to remove
            
        Returns:
            Success status
        """
        logger.info(f"Removing document {doc_id} from Qdrant")
        
        try:
            # Delete all points (chunks) with matching doc_id
            self.qdrant_client.delete(
                collection_name=self.qdrant_collection_name,
                points_selector=PointIdsList(
                    points=[],  # Empty by default, using filter instead
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="doc_id",
                                match=models.MatchValue(value=doc_id)
                            )
                        ]
                    )
                )
            )
            
            logger.info(f"✓ Document {doc_id} removed from Qdrant")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove document {doc_id}: {e}")
            return False
    
    def delete_documents(self, doc_id: str) -> bool:
        """
        Delete document(s) from Qdrant vector database
        Alias for remove_document() for API compatibility
        Permanently removes all chunks/vectors associated with the document
        
        Args:
            doc_id: Document ID to delete
            
        Returns:
            Success status
        """
        logger.info(f"Hard deleting document {doc_id} from Qdrant vectorDB")
        return self.remove_document(doc_id)
    
    def get_stats(self) -> Dict:
        """
        Get RAG service statistics from Qdrant
        
        Returns:
            Dictionary with service stats
        """
        try:
            collection_info = self.qdrant_client.get_collection(self.qdrant_collection_name)
            vector_count = collection_info.points_count
        except Exception as e:
            logger.warning(f"Failed to get collection stats: {e}")
            vector_count = 0
        
        return {
            'vector_database': 'qdrant',
            'embedding_model': self.embedding_model_name,
            'embedding_dimension': self.embedding_dim,
            'total_vectors': vector_count,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'next_point_id': self.next_point_id
        }
    
    
    def get_document_chunks(self, doc_id: str, user_id: str = "") -> List[Dict]:
        """
        Get all chunks for a specific document from Qdrant
        
        Args:
            doc_id: Document ID
            user_id: Optional user filter for privacy
            
        Returns:
            List of chunk dictionaries with text, index, and token count
        """
        logger.info(f"Retrieving chunks for document {doc_id}")
        
        try:
            # Build filter
            filters = [models.FieldCondition(
                key="doc_id",
                match=models.MatchValue(value=doc_id)
            )]
            
            if user_id:
                filters.append(models.FieldCondition(
                    key="user_id",
                    match=models.MatchValue(value=user_id)
                ))
            
            query_filter = models.Filter(must=filters) if filters else None
            
            # Scroll through all matching points
            points, _ = self.qdrant_client.scroll(
                collection_name=self.qdrant_collection_name,
                limit=1000,  # Get up to 1000 chunks
                scroll_filter=query_filter,
                with_payload=True
            )
            
            chunks = []
            for idx, point in enumerate(sorted(points, key=lambda p: p.payload.get('chunk_index', 0))):
                payload = point.payload
                chunk_text = payload.get('text', '')
                token_count = len(chunk_text.split())
                
                chunks.append({
                    'text': chunk_text,
                    'index': idx,
                    'token_count': token_count,
                    'source': payload.get('source', ''),
                    'title': payload.get('title', ''),
                    'chunk_id': payload.get('chunk_id', '')
                })
            
            logger.info(f"✓ Retrieved {len(chunks)} chunks for document {doc_id}")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to retrieve chunks for {doc_id}: {e}")
            return []
    
    def regenerate_embeddings(self, doc_id: str, user_id: str = "") -> Tuple[bool, str, int]:
        """
        Regenerate embeddings for a document in Qdrant
        Removes old embeddings and re-embeds the chunks
        
        Args:
            doc_id: Document ID
            user_id: Optional user context
            
        Returns:
            Tuple of (success, message, new_chunk_count)
        """
        logger.info(f"Regenerating embeddings for document {doc_id}")
        
        try:
            # Get the old chunks
            old_chunks = self.get_document_chunks(doc_id, user_id)
            if not old_chunks:
                return False, "Document not found", 0
            
            # Remove old embeddings
            success = self.remove_document(doc_id)
            if not success:
                return False, "Failed to remove old embeddings", 0
            
            # Re-embed the chunks
            chunks_texts = [c['text'] for c in old_chunks]
            embeddings = self.embedding_model.encode(chunks_texts, convert_to_numpy=True)
            
            # Prepare new points with updated embeddings
            points = []
            for i, (chunk, embedding) in enumerate(zip(old_chunks, embeddings)):
                chunk_id = f"{doc_id}_{i}"
                point = PointStruct(
                    id=self.next_point_id,
                    vector=embedding.tolist(),
                    payload={
                        'doc_id': doc_id,
                        'text': chunk['text'],
                        'source': chunk['source'],
                        'title': chunk['title'],
                        'user_id': user_id,
                        'chunk_id': chunk_id,
                        'chunk_index': i,
                        'added_at': datetime.now().isoformat()
                    }
                )
                points.append(point)
                self.next_point_id += 1
            
            # Upsert to Qdrant
            self.qdrant_client.upsert(
                collection_name=self.qdrant_collection_name,
                points=points
            )
            
            logger.info(f"✓ Regenerated embeddings for document {doc_id} ({len(old_chunks)} chunks)")
            return True, "Embeddings regenerated successfully", len(old_chunks)
            
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

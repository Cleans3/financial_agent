import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from src.core.config import settings
from src.core.embeddings import get_embedding_strategy

logger = logging.getLogger(__name__)


class QdrantCollectionManager:
    
    FINANCIAL_KEYWORDS = [
        "revenue", "profit", "earnings", "ebit", "net_income", "margin",
        "ratio", "debt", "equity", "asset", "liability", "cash_flow",
        "roa", "roe", "eps", "dividend", "stock", "share", "financial",
        "fiscal", "quarterly", "annual", "report", "statement", "balance",
        "income", "cash", "liquidity", "solvency", "dividend", "valuation",
        "pe_ratio", "pb_ratio", "dividend_yield", "payout", "fcf",
        "ebitda", "gross_margin", "operating_margin", "net_margin"
    ]
    
    def __init__(self):
        self.client = self._init_client()
        self.embedding_strategy = get_embedding_strategy()
        self.embedding_dim = self.embedding_strategy.embedding_dimension
    
    def _init_client(self) -> QdrantClient:
        if settings.QDRANT_MODE == "cloud" and settings.QDRANT_CLOUD_URL:
            client = QdrantClient(
                url=settings.QDRANT_CLOUD_URL,
                api_key=settings.QDRANT_CLOUD_API_KEY
            )
            logger.info(f"Connected to Qdrant Cloud")
        elif settings.QDRANT_URL:
            client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
            logger.info(f"Connected to Qdrant at {settings.QDRANT_URL}")
        else:
            client = QdrantClient(":memory:")
            logger.info("Using Qdrant in-memory mode")
        
        return client
    
    def _get_user_collection_name(self, user_id: str) -> str:
        return f"user_{user_id}".replace("-", "_")
    
    def _get_global_collection_name(self) -> str:
        return "global_admin"
    
    def _detect_content_type(self, text: str) -> str:
        text_lower = text.lower()
        financial_count = sum(1 for kw in self.FINANCIAL_KEYWORDS if kw in text_lower)
        financial_ratio = financial_count / len(text_lower.split()) if text_lower.split() else 0
        
        return "financial" if financial_ratio > 0.1 else "general"
    
    def _ensure_collection_exists(self, collection_name: str):
        """Ensure collection exists, create if not found (but don't swallow other errors)"""
        try:
            self.client.get_collection(collection_name)
            logger.debug(f"Collection exists: {collection_name}")
            # Ensure indexes exist even for existing collections
            self._create_payload_indexes(collection_name)
            return
        except Exception as e:
            # Check if it's a "not found" error vs other issues
            error_msg = str(e).lower()
            if "not found" in error_msg or "does not exist" in error_msg:
                logger.info(f"Collection not found, creating: {collection_name}")
                try:
                    self.client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(size=self.embedding_dim, distance=Distance.COSINE)
                    )
                    logger.info(f"Collection created: {collection_name}")
                    self._create_payload_indexes(collection_name)
                except Exception as create_err:
                    logger.error(f"Failed to create collection: {create_err}")
                    raise
            else:
                # Other errors (timeout, connection issues, etc) should be raised
                logger.error(f"Error checking collection {collection_name}: {e}")
                raise
    
    def _create_payload_indexes(self, collection_name: str):
        """Create indexes for payload fields to enable filtering"""
        for field in ["user_id", "chat_id", "chat_session_id", "file_id", "content_type"]:
            try:
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field,
                    field_schema="keyword"
                )
                logger.debug(f"Created/verified index on {field} for {collection_name}")
            except Exception as e:
                # Log but don't fail - index might already exist
                logger.debug(f"Could not create index on {field}: {e}")
    
    def add_document_chunks(self, 
                           user_id: str, 
                           chat_session_id: str,
                           file_id: str,
                           chunks: List[Dict],
                           metadata: Optional[Dict] = None) -> int:
        """Add document chunks to user's collection using appropriate embedding model
        
        Args:
            user_id: User ID
            chat_session_id: Chat session ID (from PostgreSQL ChatSession.id)
            file_id: Document file ID
            chunks: List of text chunks
            metadata: Additional metadata
        """
        collection_name = self._get_user_collection_name(user_id)
        self._ensure_collection_exists(collection_name)
        
        points = []
        chunk_texts = [c['text'] for c in chunks]
        
        content_type = self._detect_content_type(" ".join(chunk_texts[:3]))
        embedding_strategy = get_embedding_strategy(model_type=content_type)
        
        logger.info(f"Using {content_type} embedding model for document {file_id}")
        
        # Get next sequential point ID
        try:
            collection_info = self.client.get_collection(collection_name)
            next_point_id = collection_info.points_count + 1
        except:
            next_point_id = 1
        
        embeddings = embedding_strategy.embed(chunk_texts)
        
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point_id = next_point_id + idx
            payload = {
                'user_id': user_id,
                'chat_session_id': chat_session_id,
                'file_id': file_id,
                'chunk_index': idx,
                'text': chunk['text'],
                'title': chunk.get('title', ''),
                'source': chunk.get('source', ''),
                'content_type': content_type,
                'timestamp': datetime.now().isoformat(),
                **(metadata or {})
            }
            points.append(PointStruct(id=point_id, vector=embedding, payload=payload))
        
        self.client.upsert(collection_name=collection_name, points=points)
        logger.info(f"Added {len(points)} chunks to {collection_name} using {content_type} model")
        return len(points)
    
    def add_document_chunks_to_global(self,
                                     file_id: str,
                                     chunks: List[Dict],
                                     metadata: Optional[Dict] = None) -> int:
        """Add document chunks to GLOBAL admin collection (shared across all users)
        
        Args:
            file_id: Document file ID
            chunks: List of text chunks
            metadata: Additional metadata
        
        Returns:
            Number of chunks added
        """
        collection_name = self._get_global_collection_name()
        self._ensure_collection_exists(collection_name)
        
        points = []
        chunk_texts = [c['text'] for c in chunks]
        
        content_type = self._detect_content_type(" ".join(chunk_texts[:3]))
        embedding_strategy = get_embedding_strategy(model_type=content_type)
        
        logger.info(f"Using {content_type} embedding model for global document {file_id}")
        
        # Get next sequential point ID
        try:
            collection_info = self.client.get_collection(collection_name)
            next_point_id = collection_info.points_count + 1
        except:
            next_point_id = 1
        
        embeddings = embedding_strategy.embed(chunk_texts)
        
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point_id = next_point_id + idx
            payload = {
                'file_id': file_id,
                'chunk_index': idx,
                'text': chunk['text'],
                'title': chunk.get('title', ''),
                'source': chunk.get('source', ''),
                'content_type': content_type,
                'timestamp': datetime.now().isoformat(),
                **(metadata or {})
            }
            points.append(PointStruct(id=point_id, vector=embedding, payload=payload))
        
        self.client.upsert(collection_name=collection_name, points=points)
        logger.info(f"Added {len(points)} chunks to {collection_name} (GLOBAL) using {content_type} model")
        return len(points)
    
    def search(self, 
               user_id: str,
               query_embedding: List[float],
               query_text: str = "",
               chat_session_id: Optional[str] = None,
               limit: Optional[int] = None,
               include_global: bool = False) -> List[Dict]:
        """Search user's collection with conversation isolation
        
        Args:
            user_id: User ID
            query_embedding: Query embedding vector
            query_text: Query text for detection
            chat_session_id: Chat session ID for conversation isolation (required for isolation)
            limit: Max results to return
            include_global: Whether to fall back to global collection if no personal results
        """
        limit = limit or settings.RAG_TOP_K_RESULTS
        threshold = settings.RAG_SIMILARITY_THRESHOLD
        
        content_type = self._detect_content_type(query_text) if query_text else "general"
        embedding_strategy = get_embedding_strategy(model_type=content_type)
        
        if query_text:
            query_embedding = embedding_strategy.embed_single(query_text)
        
        # ALWAYS search personal collection first
        personal_results = self._search_collection(
            self._get_user_collection_name(user_id),
            query_embedding,
            chat_session_id=chat_session_id,
            limit=limit,
            threshold=threshold
        )
        
        # Only fall back to global if explicitly enabled AND personal results are empty
        if include_global and not personal_results:
            logger.info(f"No personal results for {user_id}, searching global collection")
            global_results = self._search_collection(
                self._get_global_collection_name(),
                query_embedding,
                chat_session_id=None,  # Global search is NOT conversation-isolated
                limit=limit,
                threshold=threshold
            )
            return global_results
        
        return personal_results
    
    def _search_collection(self,
                          collection_name: str,
                          query_embedding: List[float],
                          chat_session_id: Optional[str] = None,
                          limit: int = 5,
                          threshold: float = 0.3) -> List[Dict]:
        """Search single collection with optional conversation isolation
        
        Args:
            collection_name: Qdrant collection to search
            query_embedding: Query embedding vector
            chat_session_id: If provided, only search points from this chat session (conversation isolation)
            limit: Max results
            threshold: Minimum similarity score
        """
        try:
            search_filter = None
            if chat_session_id:
                search_filter = Filter(
                    must=[
                        FieldCondition(
                            key="chat_session_id",
                            match=MatchValue(value=chat_session_id)
                        )
                    ]
                )
                logger.debug(f"Searching {collection_name} with chat_session_id filter: {chat_session_id}")
            
            results = self.client.query_points(
                collection_name=collection_name,
                query=query_embedding,
                query_filter=search_filter,
                limit=limit,
                score_threshold=threshold
            )
            
            return [
                {
                    'text': r.payload.get('text', ''),
                    'title': r.payload.get('title', ''),
                    'source': r.payload.get('source', ''),
                    'score': r.score,
                    'metadata': {k: v for k, v in r.payload.items() if k != 'text'}
                }
                for r in results.points
            ]
        except Exception as e:
            logger.warning(f"Search in {collection_name} failed: {e}")
            return []
    
    def search_by_metadata(self,
                          collection_name: str,
                          metadata_filters: Dict,
                          chat_session_id: Optional[str] = None,
                          limit: int = 10) -> List[Dict]:
        """Search collection by metadata filters (source, title, file_id, etc.)
        
        Args:
            collection_name: Qdrant collection to search
            metadata_filters: Dict of field:value to filter by
            chat_session_id: If provided, restrict to this chat session
            limit: Max results
        """
        try:
            must_conditions = []
            
            # Add chat_session_id filter if provided (ALWAYS enforce conversation isolation)
            if chat_session_id:
                must_conditions.append(
                    FieldCondition(
                        key="chat_session_id",
                        match=MatchValue(value=chat_session_id)
                    )
                )
            
            # Add metadata filters
            for field, value in metadata_filters.items():
                must_conditions.append(
                    FieldCondition(
                        key=field,
                        match=MatchValue(value=value)
                    )
                )
            
            if not must_conditions:
                return []
            
            search_filter = Filter(must=must_conditions) if must_conditions else None
            
            logger.debug(f"Searching {collection_name} by metadata: {metadata_filters}, session={chat_session_id}")
            
            # Use scroll to get all matching points (not similarity-based)
            results = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=search_filter,
                limit=limit
            )
            
            return [
                {
                    'text': r.payload.get('text', ''),
                    'title': r.payload.get('title', ''),
                    'source': r.payload.get('source', ''),
                    'score': 1.0,  # Metadata match is a full match
                    'metadata': {k: v for k, v in r.payload.items() if k != 'text'}
                }
                for r in results[0]
            ]
        except Exception as e:
            logger.warning(f"Metadata search in {collection_name} failed: {e}")
            return []
    
    def delete_by_chat_id(self, user_id: str, chat_id: str):
        """Delete all points with matching chat_id from user's collection"""
        collection_name = self._get_user_collection_name(user_id)
        try:
            self.client.delete(
                collection_name=collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="chat_id",
                            match=MatchValue(value=chat_id)
                        )
                    ]
                )
            )
            logger.info(f"Deleted points with chat_id={chat_id} from {collection_name}")
        except Exception as e:
            logger.error(f"Failed to delete by chat_id: {e}")
    
    def delete_by_file_id(self, user_id: str, file_id: str):
        """Delete all points with matching file_id from user's collection"""
        collection_name = self._get_user_collection_name(user_id)
        try:
            self.client.delete(
                collection_name=collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="file_id",
                            match=MatchValue(value=file_id)
                        )
                    ]
                )
            )
            logger.info(f"Deleted points with file_id={file_id} from {collection_name}")
        except Exception as e:
            logger.error(f"Failed to delete by file_id: {e}")
    
    def list_collections(self) -> List[str]:
        """List all collections"""
        try:
            collections = self.client.get_collections()
            return [c.name for c in collections.collections]
        except:
            return []
    
    def collection_stats(self, collection_name: str) -> Dict:
        """Get collection statistics"""
        try:
            info = self.client.get_collection(collection_name)
            return {
                'points_count': info.points_count,
                'vector_size': info.config.params.vectors.size,
                'distance': str(info.config.params.vectors.distance)
            }
        except:
            return {}

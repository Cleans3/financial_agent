import logging
import time
from typing import List, Dict, Optional, Tuple, Any, Callable
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
        self.timeout = settings.QDRANT_TIMEOUT_SECONDS
        self.retry_attempts = settings.QDRANT_RETRY_ATTEMPTS
        self.retry_delay = settings.QDRANT_RETRY_DELAY_SECONDS
    
    def _init_client(self) -> QdrantClient:
        if settings.QDRANT_MODE == "cloud" and settings.QDRANT_CLOUD_URL:
            client = QdrantClient(
                url=settings.QDRANT_CLOUD_URL,
                api_key=settings.QDRANT_CLOUD_API_KEY,
                timeout=settings.QDRANT_TIMEOUT_SECONDS
            )
            logger.info(f"Connected to Qdrant Cloud (timeout: {settings.QDRANT_TIMEOUT_SECONDS}s)")
        elif settings.QDRANT_URL:
            client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY,
                timeout=settings.QDRANT_TIMEOUT_SECONDS
            )
            logger.info(f"Connected to Qdrant at {settings.QDRANT_URL} (timeout: {settings.QDRANT_TIMEOUT_SECONDS}s)")
        else:
            client = QdrantClient(":memory:")
            logger.info("Using Qdrant in-memory mode")
        
        return client
    
    def _get_user_collection_name(self, user_id: str) -> str:
        return f"user_{user_id}".replace("-", "_")
    
    def _get_global_collection_name(self) -> str:
        return "global_admin"
    
    def _retry_with_backoff(self, func: Callable[..., Any], *args, **kwargs) -> Any:
        """Execute function with exponential backoff retry logic for timeout errors
        
        Args:
            func: Function to retry
            *args: Function arguments
            **kwargs: Function keyword arguments
        
        Returns:
            Function result
        
        Raises:
            Exception: If all retries fail
        """
        attempt = 0
        delay = self.retry_delay
        last_error = None
        
        while attempt < self.retry_attempts:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                
                # Retry on timeout errors
                if any(x in error_str for x in ["timeout", "timed out", "connection reset"]):
                    attempt += 1
                    if attempt < self.retry_attempts:
                        logger.warning(
                            f"Qdrant timeout (attempt {attempt}/{self.retry_attempts}): {e}. "
                            f"Retrying in {delay}s..."
                        )
                        time.sleep(delay)
                        delay *= 2  # Exponential backoff
                    else:
                        logger.error(f"All {self.retry_attempts} retry attempts exhausted")
                        raise
                else:
                    # Non-timeout errors should fail immediately
                    raise
        
        raise last_error
    
    def _detect_content_type(self, text: str) -> str:
        text_lower = text.lower()
        financial_count = sum(1 for kw in self.FINANCIAL_KEYWORDS if kw in text_lower)
        financial_ratio = financial_count / len(text_lower.split()) if text_lower.split() else 0
        
        return "financial" if financial_ratio > 0.1 else "general"
    
    def _ensure_collection_exists(self, collection_name: str):
        """Ensure collection exists, create if not found with retry logic"""
        try:
            # Use retry logic for get_collection
            self._retry_with_backoff(self.client.get_collection, collection_name)
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
                    # Use retry logic for create_collection
                    self._retry_with_backoff(
                        self.client.create_collection,
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
        # Fields that need keyword indexes for metadata filtering
        indexed_fields = [
            "user_id",           # For user isolation
            "chat_id",           # For conversation isolation
            "chat_session_id",   # For session-based filtering
            "file_id",           # For document identification
            "content_type",      # For content type filtering
            "source_file",       # For filename-based search (uploaded files)
            "filename"           # For title/filename filtering
        ]
        
        for field in indexed_fields:
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
                'content_type': content_type,
                'timestamp': datetime.now().isoformat(),
                'chunk_type': chunk.get('chunk_type', 'structural'),  # Use chunk's type, default to structural
                **(metadata or {})
            }
            # Only add non-empty metadata fields
            if chunk.get('filename') and chunk['filename'].strip():
                payload['filename'] = chunk['filename']
            if chunk.get('source_file') and chunk['source_file'].strip():
                payload['source_file'] = chunk['source_file']
            # Remove any empty string values from metadata
            payload = {k: v for k, v in payload.items() if v is not None and v != ''}
            
            points.append(PointStruct(id=point_id, vector=embedding, payload=payload))
        
        # Use retry logic for upsert operation (critical write operation)
        try:
            self._retry_with_backoff(
                self.client.upsert,
                collection_name=collection_name,
                points=points
            )
            logger.info(f"Added {len(points)} chunks to {collection_name} using {content_type} model")
            return len(points)
        except Exception as e:
            logger.error(f"Failed to add chunks to {collection_name} after {self.retry_attempts} attempts: {e}")
            raise
    
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
                'content_type': content_type,
                'timestamp': datetime.now().isoformat(),
                'chunk_type': 'structural',  # Mark as structural chunk
                **(metadata or {})
            }
            # Only add non-empty metadata fields
            if chunk.get('filename') and chunk['filename'].strip():
                payload['filename'] = chunk['filename']
            if chunk.get('source_file') and chunk['source_file'].strip():
                payload['source_file'] = chunk['source_file']
            # Remove any empty string values from metadata
            payload = {k: v for k, v in payload.items() if v is not None and v != ''}
            
            points.append(PointStruct(id=point_id, vector=embedding, payload=payload))
        
        # Use retry logic for upsert operation (critical write operation)
        try:
            self._retry_with_backoff(
                self.client.upsert,
                collection_name=collection_name,
                points=points
            )
            logger.info(f"Added {len(points)} chunks to {collection_name} (GLOBAL) using {content_type} model")
            return len(points)
        except Exception as e:
            logger.error(f"Failed to add global chunks to {collection_name} after {self.retry_attempts} attempts: {e}")
            raise
    
    def search(self, 
               user_id: str,
               query_embedding: List[float],
               query_text: str = "",
               chat_session_id: Optional[str] = None,
               limit: Optional[int] = None,
               include_global: bool = False,
               chunk_type_filter: Optional[str] = None) -> List[Dict]:
        """Search user's collection with conversation isolation and optional chunk type filtering
        
        Args:
            user_id: User ID
            query_embedding: Query embedding vector
            query_text: Query text for detection
            chat_session_id: Chat session ID for conversation isolation (required for isolation)
            limit: Max results to return
            include_global: Whether to fall back to global collection if no personal results
            chunk_type_filter: Optional filter for chunk type ("structural", "metric_centric")
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
            threshold=threshold,
            chunk_type_filter=chunk_type_filter
        )
        
        # Only fall back to global if explicitly enabled AND personal results are empty
        if include_global and not personal_results:
            logger.info(f"No personal results for {user_id}, searching global collection")
            global_results = self._search_collection(
                self._get_global_collection_name(),
                query_embedding,
                chat_session_id=None,  # Global search is NOT conversation-isolated
                limit=limit,
                threshold=threshold,
                chunk_type_filter=chunk_type_filter
            )
            return global_results
        
        return personal_results
    
    def _search_collection(self,
                          collection_name: str,
                          query_embedding: Optional[List[float]],
                          chat_session_id: Optional[str] = None,
                          limit: int = 5,
                          threshold: float = None,
                          chunk_type_filter: Optional[str] = None) -> List[Dict]:
        """Search single collection with optional conversation isolation, chunk type filtering, and retry logic
        
        Args:
            collection_name: Qdrant collection to search
            query_embedding: Query embedding vector (None for metadata-only retrieval)
            chat_session_id: If provided, only search points from this chat session (conversation isolation)
            limit: Max results
            threshold: Minimum similarity score (defaults to RAG_SIMILARITY_THRESHOLD setting)
            chunk_type_filter: Optional filter for chunk type ("structural", "metric_centric")
        """
        # Use settings default if no threshold provided
        if threshold is None:
            threshold = settings.RAG_SIMILARITY_THRESHOLD
        
        # For metadata-only retrieval (query_embedding=None), return all points from session
        if query_embedding is None:
            logger.info(f"[QDRANT:SEARCH] Performing metadata-only retrieval from {collection_name} "
                       f"for chat_session_id={chat_session_id}")
            try:
                # Build filter for session only
                filter_conditions = []
                if chat_session_id:
                    filter_conditions.append(
                        FieldCondition(
                            key="chat_session_id",
                            match=MatchValue(value=chat_session_id)
                        )
                    )
                
                search_filter = None
                if filter_conditions:
                    search_filter = Filter(must=filter_conditions)
                
                # Search with dummy embedding (all zeros) and NO score threshold
                dummy_embedding = [0.0] * 384  # Match embedding dimension (sentence-transformers/all-MiniLM-L6-v2)
                results = self._retry_with_backoff(
                    self.client.query_points,
                    collection_name=collection_name,
                    query=dummy_embedding,
                    query_filter=search_filter,
                    limit=limit,
                    score_threshold=None  # No filtering by score for metadata retrieval
                )
                
                logger.info(f"[QDRANT:SEARCH] Metadata retrieval returned {len(results.points)} results")
                
                # Extract and return points as dictionaries
                points_data = []
                for point in results.points:
                    point_dict = {
                        'id': point.id,
                        'text': point.payload.get('text', ''),
                        'chunk_type': point.payload.get('chunk_type', 'unknown'),
                        'chunk_index': point.payload.get('chunk_index'),
                        'metric_name': point.payload.get('metric_name'),
                        'score': point.score if hasattr(point, 'score') else 0.0,
                        'chat_session_id': point.payload.get('chat_session_id'),
                    }
                    points_data.append(point_dict)
                
                return points_data
            except Exception as e:
                logger.error(f"[QDRANT:SEARCH] Metadata retrieval failed: {e}")
                return []
        
        try:
            # Build filter conditions
            filter_conditions = []
            
            if chat_session_id:
                filter_conditions.append(
                    FieldCondition(
                        key="chat_session_id",
                        match=MatchValue(value=chat_session_id)
                    )
                )
            
            # Note: chunk_type filter requires keyword index in Qdrant
            # If index not available, we'll do post-search filtering instead
            should_postfilter_by_type = False
            if chunk_type_filter:
                filter_conditions.append(
                    FieldCondition(
                        key="chunk_type",
                        match=MatchValue(value=chunk_type_filter)
                    )
                )
            
            search_filter = None
            if filter_conditions:
                search_filter = Filter(must=filter_conditions)
                if chat_session_id:
                    logger.debug(f"Searching {collection_name} with filters: "
                                f"chat_session_id={chat_session_id}, chunk_type={chunk_type_filter}")

            
            # CRITICAL DEBUG: Log search parameters
            logger.info(f"[QDRANT:SEARCH] Searching {collection_name} with limit={limit}, threshold={threshold}, "
                       f"chat_session_id={chat_session_id}, chunk_type_filter={chunk_type_filter}")
            
            # Use retry logic for search operation
            try:
                results = self._retry_with_backoff(
                    self.client.query_points,
                    collection_name=collection_name,
                    query=query_embedding,
                    query_filter=search_filter,
                    limit=limit,
                    score_threshold=threshold
                )
                
                # CRITICAL DEBUG: Log what we got back
                logger.info(f"[QDRANT:SEARCH] Query returned {len(results.points)} results with threshold={threshold}")
            except Exception as filter_error:
                # If chunk_type filter fails (index missing), retry without chunk_type filter
                if "chunk_type" in str(filter_error) and chunk_type_filter:
                    logger.warning(f"chunk_type filter failed (index missing), retrying without type filter: {filter_error}")
                    should_postfilter_by_type = True
                    
                    # Remove chunk_type from filters and retry
                    filter_conditions = [fc for fc in filter_conditions if fc.key != "chunk_type"]
                    search_filter = None
                    if filter_conditions:
                        search_filter = Filter(must=filter_conditions)
                    
                    # CRITICAL: Retry WITHOUT score threshold to see all results
                    logger.warning(f"[QDRANT:SEARCH] Retrying WITHOUT score_threshold (was {threshold}) to capture all results")
                    results = self._retry_with_backoff(
                        self.client.query_points,
                        collection_name=collection_name,
                        query=query_embedding,
                        query_filter=search_filter,
                        limit=limit * 2,  # Get more results since we'll filter
                        score_threshold=None  # ← NO THRESHOLD for retry!
                    )
                    logger.info(f"[QDRANT:SEARCH] Retry without threshold returned {len(results.points)} results")
                else:
                    raise
            
            # Post-filter by chunk_type if needed and extract points
            if should_postfilter_by_type:
                # Filter the point objects
                filtered_points = [p for p in results.points if p.payload.get('chunk_type') == chunk_type_filter]
                # Reconstruct a results object with filtered points
                results.points = filtered_points[:limit]
            
            # CRITICAL DEBUG: Log scores of returned results
            if results.points:
                scores = [r.score for r in results.points]
                logger.info(f"[QDRANT:SEARCH] Result scores: min={min(scores):.4f}, max={max(scores):.4f}, avg={sum(scores)/len(scores):.4f}")
            
            return [
                {
                    'id': r.id,
                    'point_id': r.id,
                    'text': r.payload.get('text', ''),
                    'title': r.payload.get('title', ''),
                    'source': r.payload.get('source', ''),
                    'score': r.score,
                    'chunk_type': r.payload.get('chunk_type', 'structural'),
                    'chunk_index': r.payload.get('chunk_index'),
                    'file_id': r.payload.get('file_id'),
                    'metric_name': r.payload.get('metric_name'),
                    'source_chunk_ids': r.payload.get('source_chunk_ids', []),
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
            
            # Use scroll to get ALL matching points (not similarity-based, no deduplication)
            # Set limit=10000 to ensure we get all results in one page for this metadata search
            results_page, _ = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=search_filter,
                limit=limit if limit <= 10000 else 10000  # Use actual limit, max 10000 per page
            )
            
            # Convert results, preserving all chunks (no deduplication)
            formatted_results = []
            for r in results_page:
                formatted_results.append({
                    'text': r.payload.get('text', ''),
                    'title': r.payload.get('title', ''),
                    'source': r.payload.get('source', ''),
                    'filename': r.payload.get('filename', ''),
                    'score': 1.0,  # Metadata match is a full match
                    'id': r.id,
                    'metadata': {k: v for k, v in r.payload.items() if k != 'text'}
                })
            
            logger.debug(f"Metadata search returned {len(formatted_results)} results")
            return formatted_results
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

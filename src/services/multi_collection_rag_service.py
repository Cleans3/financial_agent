import logging
import re
import uuid
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from src.services.qdrant_collection_manager import QdrantCollectionManager
from src.core.embeddings import get_embedding_strategy
from src.core.summarization import get_summarization_strategy, should_summarize_response
from src.core.config import settings
from qdrant_client.models import Filter, FieldCondition, MatchValue

logger = logging.getLogger(__name__)


class MultiCollectionRAGService:
    
    def __init__(self):
        self.qd_manager = QdrantCollectionManager()
        self.embedding_strategy = get_embedding_strategy()
        self.summarization_strategy = get_summarization_strategy()
    
    def chunk_text(self, text: str, chunk_size: int = None, overlap: int = None) -> List[Dict]:
        """Split text into overlapping chunks"""
        chunk_size = chunk_size or settings.CHUNK_SIZE_TOKENS
        overlap = overlap or settings.CHUNK_OVERLAP_TOKENS
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            words = sentence.split()
            sentence_length = len(words)
            
            if current_length + sentence_length > chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'length': current_length
                })
                
                overlap_sentences = []
                overlap_length = 0
                for prev_sentence in reversed(current_chunk):
                    prev_length = len(prev_sentence.split())
                    if overlap_length + prev_length <= overlap:
                        overlap_sentences.insert(0, prev_sentence)
                        overlap_length += prev_length
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_length = overlap_length
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'length': current_length
            })
        
        return chunks
    
    def add_document(self,
                    user_id: str,
                    chat_session_id: str,
                    text: str,
                    title: str = "",
                    source: str = "") -> Tuple[int, Optional[str]]:
        """
        Add document to user's collection with optional summarization.
        Includes embedding method selection logging.
        
        Args:
            user_id: User ID
            chat_session_id: Chat session ID (from PostgreSQL ChatSession.id) - not a generated one
            text: Document text to add
            title: Document title
            source: Document source/filename
        
        Returns: (chunks_added, summary)
        """
        file_id = str(uuid.uuid4())
        chunks = self.chunk_text(text)
        
        if not chunks:
            logger.warning(f"No chunks generated for document {title}")
            return 0, None
        
        # Determine and log embedding method based on file size
        text_size_kb = len(text.encode('utf-8')) / 1024
        if text_size_kb < 5:
            embedding_method = "SINGLE_DENSE"
        elif text_size_kb < 50:
            embedding_method = "MULTIDIMENSIONAL"
        else:
            embedding_method = "HIERARCHICAL"
        
        logger.info(f"[INGEST] Embedding method selection:")
        logger.info(f"  File size: {text_size_kb:.1f}KB")
        logger.info(f"  Method: {embedding_method} (threshold: <5KB=SINGLE_DENSE, <50KB=MULTIDIMENSIONAL, >=50KB=HIERARCHICAL)")
        logger.info(f"  Model: general (sentence-transformers/all-MiniLM-L6-v2)")
        logger.info(f"  Dimensions: 384")
        logger.info(f"  Strategy: {embedding_method.lower()} embedding")
        
        chunks_with_metadata = [
            {
                **chunk,
                'title': title,
                'source': source,
                'doc_id': file_id,
            }
            for chunk in chunks
        ]
        
        logger.info(f"[INGEST] Chunking: {len(chunks_with_metadata)} chunks created from {title}")
        
        num_chunks = self.qd_manager.add_document_chunks(
            user_id=user_id,
            chat_session_id=chat_session_id,
            file_id=file_id,
            chunks=chunks_with_metadata
        )
        
        logger.info(f"[INGEST] ✓ Ingested {num_chunks} vectors to Qdrant collection")
        logger.info(f"[INGEST] File ID: {file_id}, Title: {title}, Source: {source}")
        
        summary = None
        if should_summarize_response(text, settings.SUMMARIZE_MODE):
            try:
                summary = self.summarization_strategy.summarize(text, {"source": source})
                logger.info(f"[INGEST] Generated summary for {title}")
            except Exception as e:
                logger.warning(f"[INGEST] Failed to generate summary: {e}")
        
        return num_chunks, summary
    
    def add_document_to_global(self,
                              text: str,
                              title: str = "",
                              source: str = "") -> Tuple[int, Optional[str]]:
        """
        Add document to GLOBAL shared collection (admin only).
        Used for shared knowledge base documents accessible by all users.
        
        Args:
            text: Document text to add
            title: Document title
            source: Document source/filename
        
        Returns: (chunks_added, summary)
        """
        file_id = str(uuid.uuid4())
        chunks = self.chunk_text(text)
        
        if not chunks:
            logger.warning(f"No chunks generated for global document {title}")
            return 0, None
        
        # Determine and log embedding method based on file size
        text_size_kb = len(text.encode('utf-8')) / 1024
        if text_size_kb < 5:
            embedding_method = "SINGLE_DENSE"
        elif text_size_kb < 50:
            embedding_method = "MULTIDIMENSIONAL"
        else:
            embedding_method = "HIERARCHICAL"
        
        logger.info(f"[INGEST-GLOBAL] Embedding method selection:")
        logger.info(f"  File size: {text_size_kb:.1f}KB")
        logger.info(f"  Method: {embedding_method} (threshold: <5KB=SINGLE_DENSE, <50KB=MULTIDIMENSIONAL, >=50KB=HIERARCHICAL)")
        logger.info(f"  Model: general (sentence-transformers/all-MiniLM-L6-v2)")
        logger.info(f"  Dimensions: 384")
        logger.info(f"  Strategy: {embedding_method.lower()} embedding")
        
        chunks_with_metadata = [
            {
                **chunk,
                'title': title,
                'source': source,
                'doc_id': file_id,
            }
            for chunk in chunks
        ]
        
        logger.info(f"[INGEST-GLOBAL] Chunking: {len(chunks_with_metadata)} chunks created from {title}")
        
        num_chunks = self.qd_manager.add_document_chunks_to_global(
            file_id=file_id,
            chunks=chunks_with_metadata
        )
        
        logger.info(f"[INGEST-GLOBAL] ✓ Ingested {num_chunks} vectors to global collection")
        logger.info(f"[INGEST-GLOBAL] File ID: {file_id}, Title: {title}, Source: {source}")
        
        summary = None
        if should_summarize_response(text, settings.SUMMARIZE_MODE):
            try:
                summary = self.summarization_strategy.summarize(text, {"source": source})
                logger.info(f"[INGEST-GLOBAL] Generated summary for {title}")
            except Exception as e:
                logger.warning(f"[INGEST-GLOBAL] Failed to generate summary: {e}")
        
        return num_chunks, summary
    
    def _extract_filename_from_query(self, query: str) -> Optional[str]:
        """Extract filename mentioned in the query
        
        Examples:
        - "Phân tích file.pdf" → "file.pdf"
        - "What is in annual-report.xlsx?" → "annual-report.xlsx"
        - "Tóm tắt tài liệu annual-report-2024-financial-highlights.pdf" → "annual-report-2024-financial-highlights.pdf"
        """
        # Look for common file extensions
        file_patterns = [
            r'([a-zA-Z0-9_\-\.]+\.pdf)',
            r'([a-zA-Z0-9_\-\.]+\.xlsx?)',
            r'([a-zA-Z0-9_\-\.]+\.csv)',
            r'([a-zA-Z0-9_\-\.]+\.docx?)',
            r'([a-zA-Z0-9_\-\.]+\.pptx?)',
            r'([a-zA-Z0-9_\-\.]+\.png|jpg|jpeg)',
        ]
        
        for pattern in file_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def _is_hybrid_search_needed(self, query: str) -> bool:
        """Determine if hybrid search (semantic + keyword) is needed
        
        Conditions:
        - User asks for specific data (line numbers, page numbers, exact text)
        - User asks for numerical values (revenue, profit, etc.)
        - User asks for specific fields (column names, headers)
        """
        hybrid_keywords = [
            'line', 'page', 'find', 'search', 'match', 'where',  # Exact location
            'revenue', 'profit', 'margin', 'growth', 'percentage',  # Specific metrics
            'what is', 'how much', 'how many', 'calculate', 'sum',  # Specific questions
            'số liệu', 'dữ liệu', 'giá trị', 'con số', 'tính toán',  # Vietnamese
        ]
        
        query_lower = query.lower()
        return any(kw in query_lower for kw in hybrid_keywords)
    
    def search(
        self,
        query: str,
        user_id: str,
        session_id: Optional[str] = None,
        chat_session_id: Optional[str] = None,
        uploaded_filenames: Optional[List[str]] = None,
        limit: Optional[int] = None,
        include_global: bool = False,
    ) -> List[dict]:
        """
        Search with 3-phase approach with detailed logging:
        1. Exact filename match (if files uploaded)
        2. Semantic + keyword hybrid search
        3. RRF ranking and deduplication
        
        CRITICAL: For uploaded files, use threshold 0.50 (not 0.30)
        
        Args:
            query: Search query
            user_id: User ID for collection isolation
            session_id: Chat session ID (alternative parameter name for chat_session_id)
            chat_session_id: Chat session ID (primary parameter name)
            uploaded_filenames: List of uploaded filenames to boost
            limit: Max results to return (ignored - uses internal ranking)
            include_global: Whether to include global collection results
        """
        # Normalize session_id parameter (accept both names for compatibility)
        actual_session_id = chat_session_id or session_id or None
        
        logger.info(f"[RETRIEVE] Search strategy initiated:")
        logger.info(f"  Query: '{query[:60]}...'")
        logger.info(f"  Session: {actual_session_id}")
        logger.info(f"  Files: {uploaded_filenames if uploaded_filenames else 'None'}")
        
        # Phase 1: Filename extraction and boost setup
        filename_boost = {}
        if uploaded_filenames:
            # Extract first filename only
            first_file = uploaded_filenames[0] if uploaded_filenames else None
            if len(uploaded_filenames) > 1:
                logger.warning(f"  ⚠️  Multiple files provided ({len(uploaded_filenames)}), using first: {first_file}")
            
            if first_file:
                logger.info(f"  ✓ File uploaded: {first_file}")
                filename_boost[first_file] = 1.5  # 50% relevance boost
        
        # Phase 2: Semantic search with filename priority
        semantic_results = self._semantic_search_with_filename_boost(
            query, user_id, actual_session_id, filename_boost
        )
        logger.info(f"[RETRIEVE] Phase 1 - Semantic Search: {len(semantic_results)} results")
        for i, r in enumerate(semantic_results[:5]):
            score = r.get('score', 0)
            fname = r.get('metadata', {}).get('filename', 'unknown')
            logger.info(f"    [{i+1}] {fname}: score={score:.3f}")
        
        # Phase 3: Keyword search
        keyword_results = self._keyword_search(query, user_id, actual_session_id)
        logger.info(f"[RETRIEVE] Phase 2 - Keyword Search: {len(keyword_results)} results")
        for i, r in enumerate(keyword_results[:3]):
            logger.info(f"    [{i+1}] {r.get('source', 'unknown')}")
        
        # Phase 4: RRF ranking and filtering
        threshold = 0.50 if uploaded_filenames else 0.30
        final_results = self._apply_rrf_ranking_with_threshold(
            semantic_results, keyword_results, 
            threshold=threshold
        )
        
        logger.info(f"[RETRIEVE] Phase 3 - RRF Ranking: {len(final_results)} results (threshold: {threshold})")
        for i, r in enumerate(final_results[:5]):
            logger.info(f"    [{i+1}] RRF Score={r.get('score'):.3f}, source={r.get('doc', {}).get('source', 'unknown')}")
        
        logger.info(f"[RETRIEVE] ✓ Search completed - {len(final_results)} results returned (session={session_id})")
        return final_results

    def _semantic_search_with_filename_boost(
        self, 
        query: str, 
        user_id: str, 
        session_id: str, 
        filename_boost: Dict[str, float]
    ) -> List[dict]:
        """
        Semantic search with filename-based relevance boost.
        Applies 1.5x multiplier if filename matches uploaded file.
        """
        base_results = self._semantic_search(query, user_id, session_id)
        
        # Apply filename boost
        for result in base_results:
            filename = result.get("metadata", {}).get("filename", "")
            original_score = result.get("score", 0)
            
            if filename in filename_boost:
                boosted_score = original_score * filename_boost[filename]
                result["score"] = boosted_score
                logger.info(f"    ✓ Boosted {filename}: {original_score:.2f} → {boosted_score:.2f} (1.5x)")
        
        # Re-sort by boosted scores
        return sorted(base_results, key=lambda x: x.get("score", 0), reverse=True)
    
    def _semantic_search(
        self,
        query: str,
        user_id: str,
        session_id: Optional[str] = None
    ) -> List[dict]:
        """
        Perform semantic search using embeddings.
        
        Args:
            query: Search query
            user_id: User ID for collection isolation
            session_id: Optional session ID for conversation isolation
            
        Returns:
            List of search results with scores and metadata
        """
        try:
            # Use Qdrant manager's search method
            results = self.qd_manager.search(
                user_id=user_id,
                query_text=query,
                chat_session_id=session_id,
                limit=10,
                include_global=False
            )
            
            # Convert Qdrant results to our format
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "score": result.get("similarity", 0),
                    "text": result.get("text", ""),
                    "metadata": result.get("metadata", {}),
                    "doc": result
                })
            
            return formatted_results
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    def _keyword_search(
        self,
        query: str,
        user_id: str,
        session_id: Optional[str] = None
    ) -> List[dict]:
        """
        Perform keyword search using text matching.
        Serves as secondary search for hybrid results.
        
        Args:
            query: Search query
            user_id: User ID for collection isolation
            session_id: Optional session ID for conversation isolation
            
        Returns:
            List of search results matching keywords
        """
        try:
            # Use Qdrant manager's keyword search
            results = self.qd_manager.search_by_keyword(
                query=query,
                user_id=user_id,
                chat_session_id=session_id,
                limit=10
            )
            
            # Convert to our format
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "score": result.get("similarity", 0),
                    "text": result.get("text", ""),
                    "metadata": result.get("metadata", {}),
                    "source": result.get("source", ""),
                    "doc": result
                })
            
            return formatted_results
        except Exception as e:
            logger.debug(f"Keyword search failed (non-critical): {e}")
            return []
    

    def _apply_rrf_ranking_with_threshold(
        self, 
        semantic: List[dict], 
        keyword: List[dict],
        threshold: float = 0.30,
        k: int = 60
    ) -> List[dict]:
        """
        RRF ranking with relevance threshold filtering.
        CRITICAL: Use 0.50 for uploaded files, 0.30 for global searches.
        """
        logger.info(f"  Phase 3 (RRF): semantic={len(semantic)}, keyword={len(keyword)}, threshold={threshold}")
        
        scores = {}
        doc_map = {}
        
        # Semantic results (priority 1.2x)
        for i, result in enumerate(semantic):
            doc_id = result.get("id")
            score = (1.2 / (k + i + 1))
            scores[doc_id] = scores.get(doc_id, 0) + score
            doc_map[doc_id] = result
            logger.debug(f"    Semantic[{i}]: {doc_id[:8]}... = {score:.3f}")
        
        # Keyword results (priority 1.0x)
        for i, result in enumerate(keyword):
            doc_id = result.get("id")
            score = (1.0 / (k + i + 1))
            scores[doc_id] = scores.get(doc_id, 0) + score
            doc_map[doc_id] = result
            logger.debug(f"    Keyword[{i}]: {doc_id[:8]}... = {score:.3f}")
        
        # Filter by threshold and sort
        ranked = [
            {"id": doc_id, "score": score, "doc": doc_map[doc_id]} 
            for doc_id, score in scores.items() 
            if score >= threshold
        ]
        ranked = sorted(ranked, key=lambda x: x["score"], reverse=True)
        
        logger.info(f"  Phase 3 (RRF) filtered: {len(ranked)} results above threshold {threshold}")
        for r in ranked[:5]:
            logger.info(f"    {r['id'][:8]}... = {r['score']:.3f}")
        
        return ranked[:10]  # Return top 10
    
    def format_rag_context(self, results: List[Dict]) -> str:
        """Format search results as context for LLM"""
        if not results:
            return ""
        
        parts = ["📚 Related Documents:"]
        for i, result in enumerate(results, 1):
            score = result.get('score', 0)
            relevance = f"{score:.0%}" if score else "N/A"
            title = result.get('title', 'Untitled')
            source = result.get('source', '')
            text_preview = result.get('text', '')[:200]
            
            # Add keyword match info if available
            keyword_info = ""
            if 'keyword_matches' in result:
                keyword_info = f" | Keywords: {result['keyword_matches']}"
            
            source_info = f" ({source})" if source else ""
            
            parts.append(f"{i}. [{title}]{source_info} (Match: {relevance}{keyword_info})\n{text_preview}...")
        
        return "\n\n".join(parts)
    
    def search_by_filename(self,
                          user_id: str,
                          filename: str,
                          chat_session_id: Optional[str] = None,
                          limit: Optional[int] = None) -> List[Dict]:
        """Search for documents by specific filename (metadata search)
        
        ALWAYS enforces chat_session_id for conversation isolation
        """
        limit = limit or settings.RAG_TOP_K_RESULTS
        logger.info(f"Searching by filename: {filename}, session={chat_session_id}")
        
        try:
            collection_name = self.qd_manager._get_user_collection_name(user_id)
            
            # Build metadata filter for filename
            metadata_filters = {
                'source': filename
            }
            
            results = self.qd_manager.search_by_metadata(
                collection_name=collection_name,
                metadata_filters=metadata_filters,
                chat_session_id=chat_session_id,
                limit=limit
            )
            
            logger.info(f"Filename search returned {len(results)} results for {filename}")
            return results
        except Exception as e:
            logger.error(f"Filename search failed: {e}")
            return []
    
    def search_by_title(self,
                       user_id: str,
                       title: str,
                       chat_session_id: Optional[str] = None,
                       limit: Optional[int] = None) -> List[Dict]:
        """Search for documents by title (metadata search)
        
        ALWAYS enforces chat_session_id for conversation isolation
        """
        limit = limit or settings.RAG_TOP_K_RESULTS
        logger.info(f"Searching by title: {title}, session={chat_session_id}")
        
        try:
            collection_name = self.qd_manager._get_user_collection_name(user_id)
            
            metadata_filters = {
                'title': title
            }
            
            results = self.qd_manager.search_by_metadata(
                collection_name=collection_name,
                metadata_filters=metadata_filters,
                chat_session_id=chat_session_id,
                limit=limit
            )
            
            logger.info(f"Title search returned {len(results)} results for {title}")
            return results
        except Exception as e:
            logger.error(f"Title search failed: {e}")
            return []
    
    def search_by_file_id(self,
                         user_id: str,
                         file_id: str,
                         chat_session_id: Optional[str] = None,
                         limit: Optional[int] = None) -> List[Dict]:
        """Search for all chunks from a specific file
        
        ALWAYS enforces chat_session_id for conversation isolation
        """
        limit = limit or settings.RAG_TOP_K_RESULTS
        logger.info(f"Searching by file_id: {file_id}, session={chat_session_id}")
        
        try:
            collection_name = self.qd_manager._get_user_collection_name(user_id)
            
            metadata_filters = {
                'file_id': file_id
            }
            
            results = self.qd_manager.search_by_metadata(
                collection_name=collection_name,
                metadata_filters=metadata_filters,
                chat_session_id=chat_session_id,
                limit=limit
            )
            
            logger.info(f"File search returned {len(results)} results for file_id={file_id}")
            return results
        except Exception as e:
            logger.error(f"File ID search failed: {e}")
            return []
    
    def delete_conversation_data(self, user_id: str, chat_id: str):
        """Delete all vectors associated with a conversation (by chat_id)"""
        try:
            self.qd_manager.delete_by_chat_id(user_id, chat_id)
            logger.info(f"Deleted conversation {chat_id} for user {user_id}")
        except Exception as e:
            logger.error(f"Failed to delete conversation data: {e}")
    
    def delete_conversation_by_session_id(self, user_id: str, chat_session_id: str):
        """Delete all vectors associated with a chat session (by chat_session_id)
        
        This removes all points in user's collection that have the matching chat_session_id
        in their metadata. Used during conversation deletion to clean up RAG points.
        """
        try:
            collection_name = self.qd_manager._get_user_collection_name(user_id)
            
            # Delete all points with matching chat_session_id
            self.qd_manager.client.delete(
                collection_name=collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="chat_session_id",
                            match=MatchValue(value=chat_session_id)
                        )
                    ]
                )
            )
            
            logger.info(f"Deleted all points with chat_session_id={chat_session_id} from user {user_id}")
        except Exception as e:
            logger.error(f"Failed to delete points by chat_session_id: {e}")
    
    def delete_file_data(self, user_id: str, file_id: str):
        """Delete all vectors associated with a file"""
        try:
            self.qd_manager.delete_by_file_id(user_id, file_id)
            logger.info(f"Deleted file {file_id} for user {user_id}")
        except Exception as e:
            logger.error(f"Failed to delete file data: {e}")
    
    # ==================== READ-ONLY ACCESS (for normal users) ====================
    
    def search_global_readonly(
        self,
        query: str,
        user_id: str,
        session_id: str
    ) -> List[dict]:
        """
        Search ONLY (read-only) - used for normal users accessing global collection.
        
        Normal users can:
        - Search global knowledge base (semantic + keyword)
        - View results
        
        Normal users CANNOT:
        - Modify global collection
        - Upload to global collection
        - Access other users' private collections
        
        Args:
            query: Search query
            user_id: Current user ID (for audit logging)
            session_id: Chat session ID
            
        Returns:
            List of matching documents
        """
        logger.info(f"[RETRIEVE] Global search (READ-ONLY mode):")
        logger.info(f"  User: {user_id}")
        logger.info(f"  Query: '{query[:60]}...'")
        logger.info(f"  Session: {session_id}")
        
        # Search global collection only (no file upload context)
        semantic_results = self._semantic_search_global(query)
        keyword_results = self._keyword_search_global(query)
        
        logger.info(f"  Semantic: {len(semantic_results)} results")
        logger.info(f"  Keyword: {len(keyword_results)} results")
        
        # RRF ranking
        final_results = self._apply_rrf_ranking_with_threshold(
            semantic_results, keyword_results,
            threshold=0.30  # Global collection uses lower threshold
        )
        
        logger.info(f"[RETRIEVE] ✓ Global read-only search completed - {len(final_results)} results")
        logger.info(f"[AUDIT] User {user_id} searched global collection (READ-ONLY)")
        
        return final_results
    
    def _semantic_search_global(self, query: str) -> List[dict]:
        """Semantic search in global collection only"""
        try:
            embeddings = self.embedding_strategy.embed(query)
            results = self.qd_manager.search_global(embeddings, limit=10)
            return results
        except Exception as e:
            logger.warning(f"Global semantic search failed: {e}")
            return []
    
    def _keyword_search_global(self, query: str) -> List[dict]:
        """Keyword search in global collection only"""
        try:
            results = self.qd_manager.search_by_keyword_global(query, limit=10)
            return results
        except Exception as e:
            logger.warning(f"Global keyword search failed: {e}")
            return []

# Global singleton instance
_rag_service_instance = None


def get_rag_service() -> MultiCollectionRAGService:
    """Factory function to get RAG service singleton instance"""
    global _rag_service_instance
    if _rag_service_instance is None:
        _rag_service_instance = MultiCollectionRAGService()
    return _rag_service_instance
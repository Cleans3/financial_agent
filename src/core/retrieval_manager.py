"""
Retrieval Manager - Handles personal/global RAG search with fallback
Implements: RETRIEVE_PERSONAL and RETRIEVE_GLOBAL node logic
"""

import logging
from typing import List, Dict, Tuple, Optional, Any
import time

logger = logging.getLogger(__name__)


class RetrievalManager:
    """
    Manages dual RAG retrieval with personal-first, global-fallback strategy.
    
    - RETRIEVE_PERSONAL: Search user's personal vectordb
    - RETRIEVE_GLOBAL: Search shared global vectordb (fallback)
    - retrieve_with_fallback(): Orchestrates both
    
    Each search returns both semantic and keyword results.
    """
    
    def __init__(self, rag_service, embeddings=None):
        """
        Initialize retrieval manager.
        
        Args:
            rag_service: RAG service instance (handles actual search)
            embeddings: Optional embeddings model for semantic search
        """
        self.rag_service = rag_service
        self.embeddings = embeddings
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.relevance_threshold = 0.30  # Minimum similarity score to include result
        self.max_results_per_source = 10  # Max results per search type
        self.personal_fallback_threshold = 2  # Trigger global search if < N results (0 or 1)
    
    def retrieve_personal(self, query: str, user_id: str, 
                                 session_id: Optional[str] = None,
                                 uploaded_filenames: Optional[List[str]] = None) -> Tuple[List[Dict], List[Dict]]:
        """
        Search personal vectordb (user-specific collection).
        
        Args:
            query: Search query text
            user_id: User identifier
            session_id: Optional session identifier for conversation isolation
            uploaded_filenames: Optional list of uploaded filenames to prioritize
            
        Returns:
            Tuple of (semantic_results, keyword_results)
            Each result dict contains:
            {
                'id': str,
                'content': str,
                'score': float (0-1),
                'source': str,
                'doc_id': str,
                'session_id': str,
                'search_type': 'semantic' or 'keyword'
            }
        """
        try:
            start_time = time.time()
            self.logger.info(f"Retrieving from personal vectordb for query: {query}")
            
            # Search personal collection using rag_service.search()
            # Note: rag_service.search() is NOT async, so we don't await it
            if self.rag_service and hasattr(self.rag_service, 'search'):
                try:
                    # rag_service.search() returns a hybrid search (semantic + keyword combined)
                    results = self.rag_service.search(
                        query=query,
                        user_id=user_id,
                        session_id=session_id,
                        uploaded_filenames=uploaded_filenames,
                        limit=self.max_results_per_source
                    )
                    
                    # Handle None or empty results
                    if not results:
                        self.logger.info(f"Personal retrieval returned 0 total results")
                        return ([], [])
                    
                    # Filter results by relevance threshold
                    if isinstance(results, list):
                        personal_results = [
                            {
                                **result,
                                'retrieval_source': 'personal',
                                # Ensure both 'text' and 'content' fields exist for compatibility
                                'content': result.get('text') or result.get('content', ''),
                                'text': result.get('text') or result.get('content', '')
                            }
                            for result in results
                            if isinstance(result, dict) and result.get('score', 0) >= self.relevance_threshold
                        ]
                    else:
                        self.logger.error(f"Unexpected results type: {type(results)}, expected list")
                        return ([], [])
                    
                    self.logger.info(f"Personal retrieval returned {len(personal_results)} results")
                    return (personal_results, [])
                        
                except (TypeError, AttributeError) as err:
                    # Handle unexpected keyword arguments or attribute errors
                    self.logger.error(f"Personal retrieval failed: {err}")
                    return ([], [])
                    
            return ([], [])
            
        except Exception as e:
            self.logger.error(f"Personal retrieval failed: {e}")
            return ([], [])
    
    def retrieve_global(self, query: str, user_id: str, session_id: Optional[str] = None) -> Tuple[List[Dict], List[Dict]]:
        """
        Search global vectordb (shared/admin-added documents).
        
        Used as fallback when personal retrieval returns insufficient results.
        
        Args:
            query: Search query text
            user_id: User identifier (required for search_global_readonly)
            session_id: Optional session identifier
            
        Returns:
            Tuple of (semantic_results, keyword_results)
        """
        try:
            start_time = time.time()
            self.logger.info(f"Retrieving from global vectordb for query: {query}")
            
            # Search global collection
            if self.rag_service and hasattr(self.rag_service, 'search_global_readonly'):
                try:
                    # search_global_readonly requires user_id and session_id parameters
                    results = self.rag_service.search_global_readonly(
                        query=query,
                        user_id=user_id,
                        session_id=session_id or ""
                    )
                    
                    # Handle None or empty results
                    if not results:
                        self.logger.info(f"Global retrieval returned 0 results")
                        return ([], [])
                    
                    # Filter results by relevance threshold
                    if isinstance(results, list):
                        global_results = [
                            {
                                **result,
                                'retrieval_source': 'global',
                                # Ensure both 'text' and 'content' fields exist for compatibility
                                'content': result.get('text') or result.get('content', ''),
                                'text': result.get('text') or result.get('content', '')
                            }
                            for result in results
                            if isinstance(result, dict) and result.get('score', 0) >= self.relevance_threshold
                        ]
                    else:
                        self.logger.error(f"Unexpected results type: {type(results)}, expected list")
                        return ([], [])
                    
                    self.logger.info(f"Global retrieval returned {len(global_results)} results")
                    return (global_results, [])
                        
                except (TypeError, AttributeError) as e:
                    self.logger.error(f"Global retrieval failed: {e}")
                    return ([], [])
            
            return ([], [])
        
        except Exception as e:
            self.logger.error(f"Global retrieval failed: {e}")
            return ([], [])
    
    async def retrieve_with_fallback(self, query: str, user_id: str, 
                                      session_id: Optional[str] = None,
                                      uploaded_filenames: Optional[List[str]] = None) -> Dict[str, List[Dict]]:
        """
        Retrieve from personal first, fall back to global if needed.
        
        INTELLIGENT DATABASE REASONING:
        1. Search personal vectordb (user's uploaded files + conversation context)
        2. Analyze personal results:
           - If >= threshold results AND good quality → STOP, don't query global (saves resources)
           - If < threshold results → Query global as fallback
        3. Return combined results with reasoning metadata
        
        Args:
            query: Search query text
            user_id: User identifier
            session_id: Optional session identifier
            uploaded_filenames: Optional list of filenames from current upload to prioritize
            
        Returns:
            Dict with keys:
            {
                'personal_semantic': List[Dict],
                'personal_keyword': List[Dict],
                'global_semantic': List[Dict],
                'global_keyword': List[Dict],
                'fallback_to_global': bool,
                'total_results': int,
                'db_reasoning': str  # Explanation of which DB was used and why
            }
        """
        start_time = time.time()
        fallback_triggered = False
        db_reasoning = ""
        
        # Step 1: Try personal retrieval first
        self.logger.info(f"\n[DATABASE REASONING] Starting retrieval for user: {user_id}")
        self.logger.info(f"[DATABASE REASONING] Query: {query[:80]}...")
        if uploaded_filenames:
            self.logger.info(f"[DATABASE REASONING] Uploaded files: {uploaded_filenames}")
        
        personal_semantic, personal_keyword = self.retrieve_personal(
            query, user_id, session_id, uploaded_filenames=uploaded_filenames
        )
        
        personal_total = len(personal_semantic) + len(personal_keyword)
        self.logger.info(f"[DATABASE REASONING] Personal DB returned: {personal_total} results")
        
        # Step 2: Analyze personal results and decide on global fallback
        global_semantic = []
        global_keyword = []
        
        if personal_total >= self.personal_fallback_threshold:
            # Personal DB has sufficient results
            db_reasoning = (
                f"✓ Using PERSONAL DB only ({personal_total} results >= threshold {self.personal_fallback_threshold}) "
                f"- Skipping global query to save resources"
            )
            self.logger.info(f"[DATABASE REASONING] {db_reasoning}")
            
        else:
            # Personal DB insufficient, trigger global fallback
            db_reasoning = (
                f"⚠ Personal DB insufficient ({personal_total} results < threshold {self.personal_fallback_threshold}) "
                f"- Attempting global DB fallback"
            )
            self.logger.info(f"[DATABASE REASONING] {db_reasoning}")
            
            fallback_triggered = True
            self.logger.info(f"[DATABASE REASONING] Querying global DB...")
            global_semantic, global_keyword = self.retrieve_global(query, user_id, session_id)
            
            global_total = len(global_semantic) + len(global_keyword)
            self.logger.info(f"[DATABASE REASONING] Global DB returned: {global_total} results")
            
            if global_total > 0:
                db_reasoning += f" - Global returned {global_total} results"
                self.logger.info(f"[DATABASE REASONING] Using combined results (personal + global)")
            else:
                db_reasoning += f" - Global returned 0 results"
                self.logger.info(f"[DATABASE REASONING] No results from either DB")
        
        total_results = personal_total + len(global_semantic) + len(global_keyword)
        elapsed = time.time() - start_time
        
        self.logger.info(f"[DATABASE REASONING] Final: {total_results} total results in {elapsed:.2f}s")
        self.logger.info(f"[DATABASE REASONING] Breakdown: Personal={personal_total}, Global={len(global_semantic) + len(global_keyword)}")
        
        result = {
            'personal_semantic': personal_semantic,
            'personal_keyword': personal_keyword,
            'global_semantic': global_semantic,
            'global_keyword': global_keyword,
            'fallback_to_global': fallback_triggered,
            'total_results': total_results,
            'elapsed_seconds': elapsed,
            'db_reasoning': db_reasoning  # Include reasoning explanation
        }
        
        self.logger.info(
            f"Retrieval complete: {total_results} total results "
            f"(personal: {personal_total}, global: {len(global_semantic) + len(global_keyword)}) "
            f"in {elapsed:.2f}s"
        )
        
        return result
    
    def set_relevance_threshold(self, threshold: float) -> None:
        """
        Set minimum relevance score for semantic results.
        
        Args:
            threshold: Value between 0.0 and 1.0
        """
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")
        
        self.relevance_threshold = threshold
        self.logger.info(f"Relevance threshold set to {threshold}")
    
    def set_fallback_threshold(self, threshold: int) -> None:
        """
        Set number of results needed to skip global fallback.
        
        Args:
            threshold: Minimum number of personal results to skip global search
        """
        if threshold < 0:
            raise ValueError(f"Fallback threshold must be >= 0, got {threshold}")
        
        self.personal_fallback_threshold = threshold
        self.logger.info(f"Personal fallback threshold set to {threshold}")

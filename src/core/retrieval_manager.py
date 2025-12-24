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
        self.personal_fallback_threshold = 3  # Trigger global search if < N results
    
    async def retrieve_personal(self, query: str, user_id: str, 
                                 session_id: Optional[str] = None) -> Tuple[List[Dict], List[Dict]]:
        """
        Search personal vectordb (user-specific collection).
        
        Args:
            query: Search query text
            user_id: User identifier
            session_id: Optional session identifier for conversation isolation
            
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
            
            semantic_results = []
            keyword_results = []
            
            # Search personal collection by semantic similarity
            if self.rag_service and hasattr(self.rag_service, 'search'):
                semantic = await self.rag_service.search(
                    query=query,
                    user_id=user_id,
                    session_id=session_id,
                    collection_type='personal',
                    search_type='semantic',
                    limit=self.max_results_per_source,
                    threshold=self.relevance_threshold
                )
                
                if semantic:
                    semantic_results = [
                        {
                            **result,
                            'search_type': 'semantic',
                            'retrieval_source': 'personal'
                        }
                        for result in semantic
                        if result.get('score', 0) >= self.relevance_threshold
                    ]
            
            # Search personal collection by keyword
            if self.rag_service and hasattr(self.rag_service, 'keyword_search'):
                keyword = await self.rag_service.keyword_search(
                    query=query,
                    user_id=user_id,
                    session_id=session_id,
                    collection_type='personal',
                    limit=self.max_results_per_source
                )
                
                if keyword:
                    keyword_results = [
                        {
                            **result,
                            'search_type': 'keyword',
                            'retrieval_source': 'personal'
                        }
                        for result in keyword
                    ]
            
            elapsed = time.time() - start_time
            self.logger.info(
                f"Personal retrieval for user={user_id}: "
                f"{len(semantic_results)} semantic, {len(keyword_results)} keyword results "
                f"(took {elapsed:.2f}s)"
            )
            
            return (semantic_results, keyword_results)
        
        except Exception as e:
            self.logger.error(f"Personal retrieval failed: {e}")
            return ([], [])
    
    async def retrieve_global(self, query: str) -> Tuple[List[Dict], List[Dict]]:
        """
        Search global vectordb (shared/admin-added documents).
        
        Used as fallback when personal retrieval returns insufficient results.
        
        Args:
            query: Search query text
            
        Returns:
            Tuple of (semantic_results, keyword_results)
        """
        try:
            start_time = time.time()
            
            semantic_results = []
            keyword_results = []
            
            # Search global collection by semantic similarity
            if self.rag_service and hasattr(self.rag_service, 'search'):
                semantic = await self.rag_service.search(
                    query=query,
                    collection_type='global',
                    search_type='semantic',
                    limit=self.max_results_per_source,
                    threshold=self.relevance_threshold
                )
                
                if semantic:
                    semantic_results = [
                        {
                            **result,
                            'search_type': 'semantic',
                            'retrieval_source': 'global'
                        }
                        for result in semantic
                        if result.get('score', 0) >= self.relevance_threshold
                    ]
            
            # Search global collection by keyword
            if self.rag_service and hasattr(self.rag_service, 'keyword_search'):
                keyword = await self.rag_service.keyword_search(
                    query=query,
                    collection_type='global',
                    limit=self.max_results_per_source
                )
                
                if keyword:
                    keyword_results = [
                        {
                            **result,
                            'search_type': 'keyword',
                            'retrieval_source': 'global'
                        }
                        for result in keyword
                    ]
            
            elapsed = time.time() - start_time
            self.logger.info(
                f"Global retrieval: "
                f"{len(semantic_results)} semantic, {len(keyword_results)} keyword results "
                f"(took {elapsed:.2f}s)"
            )
            
            return (semantic_results, keyword_results)
        
        except Exception as e:
            self.logger.error(f"Global retrieval failed: {e}")
            return ([], [])
    
    async def retrieve_with_fallback(self, query: str, user_id: str, 
                                      session_id: Optional[str] = None) -> Dict[str, List[Dict]]:
        """
        Retrieve from personal first, fall back to global if needed.
        
        Strategy:
        1. Search personal vectordb (semantic + keyword)
        2. If < 3 results found, also search global
        3. Return dict with all 4 result lists
        
        Args:
            query: Search query text
            user_id: User identifier
            session_id: Optional session identifier
            
        Returns:
            Dict with keys:
            {
                'personal_semantic': List[Dict],
                'personal_keyword': List[Dict],
                'global_semantic': List[Dict],
                'global_keyword': List[Dict],
                'fallback_to_global': bool,
                'total_results': int
            }
        """
        start_time = time.time()
        fallback_triggered = False
        
        # Step 1: Try personal retrieval
        self.logger.info(f"Retrieving from personal vectordb for query: {query[:100]}")
        personal_semantic, personal_keyword = await self.retrieve_personal(
            query, user_id, session_id
        )
        
        personal_total = len(personal_semantic) + len(personal_keyword)
        self.logger.info(f"Personal retrieval returned {personal_total} total results")
        
        # Step 2: Decide if we need global fallback
        global_semantic = []
        global_keyword = []
        
        if personal_total < self.personal_fallback_threshold:
            self.logger.info(
                f"Personal results ({personal_total}) below threshold "
                f"({self.personal_fallback_threshold}), triggering global fallback"
            )
            fallback_triggered = True
            global_semantic, global_keyword = await self.retrieve_global(query)
            
            global_total = len(global_semantic) + len(global_keyword)
            self.logger.info(f"Global retrieval returned {global_total} results")
        
        total_results = personal_total + len(global_semantic) + len(global_keyword)
        elapsed = time.time() - start_time
        
        result = {
            'personal_semantic': personal_semantic,
            'personal_keyword': personal_keyword,
            'global_semantic': global_semantic,
            'global_keyword': global_keyword,
            'fallback_to_global': fallback_triggered,
            'total_results': total_results,
            'elapsed_seconds': elapsed
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

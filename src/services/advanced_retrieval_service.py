"""
Advanced Retrieval Service - Handles metric-centric and structural chunk retrieval with RRF and dedup

Features:
- Dual-level retrieval: metric-centric + structural chunks
- Reciprocal Rank Fusion (RRF) for score combination
- Deduplication by structural chunk (point_id/chunk_index)
- LLM-based query classification to determine retrieval strategy
- Comprehensive logging using universal project standards
"""

import logging
import math
import json
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)


class QueryType(str, Enum):
    """Query classification for routing to appropriate retrieval strategy"""
    SUMMARY = "summary"  # Asking for summary/analysis
    METRIC_SPECIFIC = "metric_specific"  # Asking about specific metric
    COMPARATIVE = "comparative"  # Comparing metrics across periods
    ANOMALY = "anomaly"  # Finding unusual patterns
    GUIDANCE = "guidance"  # Guidance-related query
    GENERAL = "general"  # General financial query
    

class RetrievalStrategy(str, Enum):
    """Retrieval strategy selection"""
    METRIC_CENTRIC_ONLY = "metric_centric_only"
    STRUCTURAL_ONLY = "structural_only"
    DUAL_WITH_RRF = "dual_with_rrf"  # Query-based ranking with RRF
    DUAL_METADATA_ONLY = "dual_metadata_only"  # Metadata-based retrieval for generic queries
    HIERARCHICAL = "hierarchical"


class QueryClassifier:
    """LLM-based query classification for intelligent retrieval strategy selection"""
    
    METRIC_KEYWORDS = {
        QueryType.SUMMARY: ["summary", "summarize", "overview", "what happened", "analysis", "takeaways"],
        QueryType.METRIC_SPECIFIC: ["revenue", "margin", "profit", "earnings", "cash flow", "debt", "equity"],
        QueryType.COMPARATIVE: ["vs", "versus", "compared", "compared to", "difference", "change", "accelerat", "decelerat"],
        QueryType.ANOMALY: ["unusual", "strange", "anomaly", "unexpected", "concern", "risk", "red flag"],
        QueryType.GUIDANCE: ["guidance", "outlook", "forecast", "expect", "project", "guidance"],
    }
    
    def __init__(self):
        """Initialize LLM-based classifier with LLM provider"""
        from src.llm.llm_factory import LLMFactory
        self.llm = LLMFactory.get_llm()
    
    def classify_with_llm(self, query: str) -> Dict[str, any]:
        """
        Use LLM to classify query and determine if it needs:
        - GENERIC retrieval (metadata-only, all chunks from file)
        - SPECIFIC retrieval (embedding-based RRF, relevant chunks)
        
        Returns dict with:
        - is_generic: bool - whether query is generic file analysis
        - reasoning: str - LLM's explanation
        - query_type: QueryType - the query type classification
        """
        try:
            logger.debug(f"[LLM:CLASSIFY] Classifying query with LLM: {query[:100]}")
            
            classification_prompt = ChatPromptTemplate.from_template(
                """Analyze this financial query and determine the retrieval strategy needed.

Query: "{query}"

Determine if this is a:
1. GENERIC query - User wants whole-file analysis, summary, or overview. Examples:
   - "Analyze and summarize this file"
   - "Give me an overview of this document"
   - "What's in this report?"
   
2. SPECIFIC query - User wants details about specific metrics or comparisons. Examples:
   - "What is the revenue for Q1?"
   - "Compare profit margins across quarters"
   - "Show me the cash flow data"

Respond ONLY with valid JSON (no markdown, no code blocks):
{{
  "is_generic": true/false,
  "reasoning": "Brief explanation of why this is generic/specific",
  "confidence": 0.0-1.0
}}"""
            )
            
            # Create chain
            chain = classification_prompt | self.llm
            
            # Get response
            response = chain.invoke({"query": query})
            response_text = response.content.strip()
            
            # Parse JSON response
            classification = json.loads(response_text)
            is_generic = classification.get("is_generic", False)
            reasoning = classification.get("reasoning", "")
            confidence = classification.get("confidence", 0.5)
            
            logger.info(f"[LLM:CLASSIFY] Generic={is_generic}, Confidence={confidence:.2f}, Reason={reasoning}")
            
            return {
                "is_generic": is_generic,
                "reasoning": reasoning,
                "confidence": confidence,
                "query_type": self.classify_query_type(query)
            }
        
        except Exception as e:
            logger.warning(f"[LLM:CLASSIFY] LLM classification failed: {e}. Falling back to pattern matching.")
            # Fallback to pattern matching
            is_generic = self._is_generic_pattern_match(query)
            return {
                "is_generic": is_generic,
                "reasoning": "Pattern matching fallback",
                "confidence": 0.3,
                "query_type": self.classify_query_type(query)
            }
    
    def _is_generic_pattern_match(self, query: str) -> bool:
        """Fallback pattern matching if LLM fails"""
        query_lower = query.lower()
        generic_patterns = [
            "analyze", "summarize", "summary", "overview", "review", 
            "examine", "explore", "look at", "tell me about", "explain",
            "describe", "provide analysis", "whole file", "entire file", "complete file"
        ]
        
        generic_count = sum(1 for pattern in generic_patterns if pattern in query_lower)
        specific_count = sum(
            1 for keywords in self.METRIC_KEYWORDS.values() 
            for kw in keywords if kw in query_lower
        )
        
        if generic_count > 0 and specific_count == 0:
            return True
        if len(query_lower) < 60 and generic_count > 0:
            return True
        
        return False
    
    def classify_query_type(self, query: str) -> QueryType:
        """Classify query to determine best retrieval strategy"""
        query_lower = query.lower()
        scores = {}
        
        for query_type, keywords in self.METRIC_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            scores[query_type] = score
        
        # Return highest scoring type, default to GENERAL
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        return QueryType.GENERAL


class RRFScorer:
    """Reciprocal Rank Fusion scorer for combining multiple search results"""
    
    K = 60  # Standard RRF parameter
    
    @staticmethod
    def score(rank: int) -> float:
        """
        Calculate RRF score for a given rank (1-indexed)
        Formula: 1 / (k + rank)
        
        Args:
            rank: Position in ranked list (1-indexed)
            
        Returns:
            RRF score
        """
        return 1.0 / (RRFScorer.K + rank)
    
    @staticmethod
    def fuse(result_sets: List[List[Dict]]) -> List[Dict]:
        """
        Fuse multiple ranked result sets using RRF
        
        Args:
            result_sets: List of ranked result lists, each with 'id' and 'score' fields
            
        Returns:
            Fused ranked list with combined scores
        """
        logger.debug(f"[RRF] Fusing {len(result_sets)} result sets")
        
        # Track scores by document ID
        fused_scores = {}  # id -> total_rrf_score
        fused_docs = {}    # id -> document dict
        
        for result_set_idx, result_set in enumerate(result_sets):
            for rank, result in enumerate(result_set, start=1):
                doc_id = result.get('id') or result.get('point_id')
                rrf_score = RRFScorer.score(rank)
                
                if doc_id not in fused_scores:
                    fused_scores[doc_id] = 0
                    fused_docs[doc_id] = result
                
                fused_scores[doc_id] += rrf_score
                logger.debug(f"[RRF] Set {result_set_idx}, Rank {rank}: "
                            f"ID={doc_id}, RRF_score={rrf_score:.4f}, Running_total={fused_scores[doc_id]:.4f}")
        
        # Sort by fused score
        sorted_results = sorted(
            [(doc_id, fused_docs[doc_id], score) for doc_id, score in fused_scores.items()],
            key=lambda x: x[2],
            reverse=True
        )
        
        # Return with combined scores
        fused = []
        for rank, (doc_id, doc, rrf_score) in enumerate(sorted_results, start=1):
            doc['rrf_score'] = rrf_score
            doc['rrf_rank'] = rank
            fused.append(doc)
        
        logger.info(f"[RRF] ✓ Fused results: {len(fused)} unique documents")
        return fused


class DeduplicationService:
    """Deduplication logic for structural chunks"""
    
    @staticmethod
    def dedup_by_structural_chunk(results: List[Dict]) -> Tuple[List[Dict], Dict]:
        """
        Deduplicate results by chunk ID (true duplicates only)
        
        CRITICAL FIX: Keep all distinct metric-centric chunks, even if they reference
        the same structural chunk. Only remove ACTUAL duplicates (same chunk_id).
        
        Previous logic was too aggressive - it would keep only 1 metric chunk per structural
        chunk, causing loss of valuable different perspectives on the same source.
        
        Args:
            results: Mixed metric-centric and structural chunks
            
        Returns:
            (deduplicated_results, dedup_map) where dedup_map tracks which results were kept
        """
        logger.info(f"[DEDUP] Starting deduplication of {len(results)} results")
        
        seen_ids = set()
        deduplicated = []
        dedup_map = {}
        duplicates_removed = 0
        
        for result in results:
            # Use chunk_id as primary dedup key (true unique identifier for each chunk)
            chunk_id = result.get('chunk_id') or result.get('id') or result.get('point_id')
            
            if chunk_id not in seen_ids:
                # New unique chunk - keep it
                seen_ids.add(chunk_id)
                deduplicated.append(result)
                dedup_map[chunk_id] = chunk_id
                logger.debug(f"[DEDUP:KEEP] chunk_id={chunk_id}, type={result.get('chunk_type', 'unknown')}")
            else:
                # Duplicate chunk ID - skip it
                duplicates_removed += 1
                logger.debug(f"[DEDUP:SKIP] chunk_id={chunk_id} (duplicate)")
        
        logger.info(f"[DEDUP] ✓ Deduplicated from {len(results)} to {len(deduplicated)} results "
                   f"({duplicates_removed} duplicates removed)")
        
        return deduplicated, dedup_map


class AdvancedRetrievalService:
    """
    Advanced retrieval service for metric-centric and structural chunks
    
    Workflow:
    1. Classify query type
    2. Select retrieval strategy
    3. Execute dual retrieval (metric-centric + structural)
    4. Fuse results with RRF
    5. Deduplicate by structural chunk
    """
    
    def __init__(self, qdrant_manager):
        """
        Initialize retrieval service
        
        Args:
            qdrant_manager: QdrantCollectionManager instance for actual retrieval
        """
        self.qd_manager = qdrant_manager
        self.classifier = QueryClassifier()
        logger.info("[RETRIEVAL] Advanced retrieval service initialized")
    
    def retrieve(self,
                 user_id: str,
                 query: str,
                 query_embedding: List[float],
                 chat_session_id: Optional[str] = None,
                 limit: int = 10,
                 include_metrics: bool = True,
                 include_structural: bool = True) -> List[Dict]:
        """
        Advanced retrieval with metric-centric and structural chunks
        WITH FALLBACK TO GLOBAL COLLECTION
        
        Args:
            user_id: User ID
            query: User query
            query_embedding: Query embedding vector
            chat_session_id: Optional chat session ID for context
            limit: Number of results to return
            include_metrics: Include metric-centric chunks
            include_structural: Include structural chunks
            
        Returns:
            Deduplicated and ranked results with RRF scores
        """
        logger.info(f"[RETRIEVAL] Starting advanced retrieval: "
                   f"query_length={len(query)}, limit={limit}, session_id={chat_session_id}, user_id={user_id}")
        
        # Step 1: Classify query using LLM to determine retrieval strategy
        classification = self.classifier.classify_with_llm(query)
        is_generic = classification["is_generic"]
        query_type = classification["query_type"]
        logger.info(f"[RETRIEVAL:CLASSIFY] Query type: {query_type.value}, Generic: {is_generic}, "
                   f"Confidence: {classification['confidence']:.2f}")
        
        # Step 2: Select retrieval strategy based on query specificity
        strategy = self._select_strategy(query_type, is_generic, include_metrics, include_structural)
        logger.info(f"[RETRIEVAL:STRATEGY] Selected: {strategy.value}")
        
        # Step 3: Execute retrieval based on strategy
        all_results = []
        result_sets = []  # For RRF fusion
        
        # For generic queries, use metadata-only retrieval (no embedding-based search)
        if strategy == RetrievalStrategy.DUAL_METADATA_ONLY:
            metric_results = self._retrieve_metric_chunks_metadata_only(user_id, chat_session_id, limit)
            if metric_results:
                all_results.extend(metric_results)
                logger.info(f"[RETRIEVAL:METADATA] Retrieved {len(metric_results)} metric chunks (metadata-based)")
            
            structural_results = self._retrieve_structural_chunks_metadata_only(user_id, chat_session_id, limit)
            if structural_results:
                all_results.extend(structural_results)
                logger.info(f"[RETRIEVAL:METADATA] Retrieved {len(structural_results)} structural chunks (metadata-based)")
        
        # For specific queries, use embedding-based retrieval with RRF
        elif strategy in [RetrievalStrategy.DUAL_WITH_RRF, RetrievalStrategy.METRIC_CENTRIC_ONLY]:
            # Retrieve metric-centric chunks
            metric_results = self._retrieve_metric_chunks(
                user_id, query_embedding, chat_session_id, limit
            )
            if metric_results:
                all_results.extend(metric_results)
                result_sets.append(metric_results)
                logger.info(f"[RETRIEVAL:METRIC] Retrieved {len(metric_results)} metric chunks")
        
        if strategy in [RetrievalStrategy.DUAL_WITH_RRF, RetrievalStrategy.STRUCTURAL_ONLY]:
            # Retrieve structural chunks
            structural_results = self._retrieve_structural_chunks(
                user_id, query_embedding, chat_session_id, limit
            )
            if structural_results:
                all_results.extend(structural_results)
                result_sets.append(structural_results)
                logger.info(f"[RETRIEVAL:STRUCTURAL] Retrieved {len(structural_results)} structural chunks")
        
        # Step 4: Fuse with RRF if multiple result sets and not metadata-only
        if strategy != RetrievalStrategy.DUAL_METADATA_ONLY and len(result_sets) > 1:
            logger.info(f"[RETRIEVAL:RRF] Fusing {len(result_sets)} result sets with RRF")
            fused_results = RRFScorer.fuse(result_sets)
            all_results = fused_results
        
        # Step 5: Deduplicate by structural chunk
        if strategy in [RetrievalStrategy.DUAL_WITH_RRF, RetrievalStrategy.DUAL_METADATA_ONLY]:
            deduplicated, dedup_map = DeduplicationService.dedup_by_structural_chunk(all_results)
            logger.info(f"[RETRIEVAL:DEDUP] Deduplication complete: {len(all_results)} -> {len(deduplicated)}")
            all_results = deduplicated
        
        # Return top limit results
        final_results = all_results[:limit]
        
        # CRITICAL: FALLBACK TO GLOBAL COLLECTION if no results
        if not final_results or len(final_results) == 0:
            logger.warning(f"[RETRIEVAL:FALLBACK] No results from user collection, attempting fallback to global retrieval")
            try:
                # Try to retrieve from global/shared collection without user_id filter
                global_metric_results = self._retrieve_metric_chunks_global(query_embedding, limit)
                global_structural_results = self._retrieve_structural_chunks_global(query_embedding, limit)
                
                global_results = []
                if global_metric_results:
                    global_results.extend(global_metric_results)
                    logger.info(f"[RETRIEVAL:FALLBACK] Retrieved {len(global_metric_results)} metric chunks from global collection")
                if global_structural_results:
                    global_results.extend(global_structural_results)
                    logger.info(f"[RETRIEVAL:FALLBACK] Retrieved {len(global_structural_results)} structural chunks from global collection")
                
                if global_results:
                    final_results = global_results[:limit]
                    logger.info(f"[RETRIEVAL:FALLBACK] ✓ Fallback successful: {len(final_results)} results from global collection")
                else:
                    logger.warning(f"[RETRIEVAL:FALLBACK] Global collection also returned no results")
            except Exception as e:
                logger.warning(f"[RETRIEVAL:FALLBACK] Fallback attempt failed: {e}")
        
        logger.info(f"[RETRIEVAL] ✓ Final results: {len(final_results)} chunks")
        
        return final_results
    
    def _select_strategy(self,
                        query_type: QueryType,
                        is_generic: bool,
                        include_metrics: bool,
                        include_structural: bool) -> RetrievalStrategy:
        """Select retrieval strategy based on query specificity and type
        
        For generic queries (file analysis, summarization): use metadata-only retrieval
        For specific queries (detailed metric queries): use embedding-based RRF retrieval
        """
        
        # Generic queries: Use metadata-only retrieval (fast, gets all file chunks)
        if is_generic and include_metrics and include_structural:
            return RetrievalStrategy.DUAL_METADATA_ONLY
        
        # Specific queries: Use embedding-based retrieval with RRF
        if include_metrics and include_structural:
            # Certain query types benefit from RRF
            if query_type in [QueryType.SUMMARY, QueryType.COMPARATIVE, QueryType.ANOMALY]:
                return RetrievalStrategy.DUAL_WITH_RRF
            return RetrievalStrategy.DUAL_WITH_RRF
        
        # Fall back to single type if only one available
        if include_metrics:
            return RetrievalStrategy.METRIC_CENTRIC_ONLY
        if include_structural:
            return RetrievalStrategy.STRUCTURAL_ONLY
        
        return RetrievalStrategy.DUAL_WITH_RRF
    
    def _retrieve_metric_chunks(self,
                               user_id: str,
                               query_embedding: List[float],
                               chat_session_id: Optional[str],
                               limit: int) -> List[Dict]:
        """Retrieve metric-centric chunks from vector DB (post-filter by chunk_type since it's metadata)"""
        try:
            # Retrieve all results without type filter (chunk_type is metadata, can't filter in Qdrant)
            # We'll filter in Python and then take top limit results
            all_results = self.qd_manager.search(
                user_id=user_id,
                query_embedding=query_embedding,
                chat_session_id=chat_session_id,
                limit=500  # Get many results to ensure we have enough metric chunks after filtering
            )
            
            # DEBUG: Check what chunk_type values we have in the results
            chunk_types_found = {}
            for r in all_results:
                ct = r.get('chunk_type', 'NOT_SET')
                chunk_types_found[ct] = chunk_types_found.get(ct, 0) + 1
            logger.info(f"[RETRIEVAL:METRIC] Chunk types in 500 results: {chunk_types_found}")
            
            # Post-filter: Keep only metric_centric chunks, then take top limit
            metric_results = [r for r in all_results if r.get('chunk_type') == 'metric_centric'][:limit]
            
            # Add rank information for RRF
            for rank, result in enumerate(metric_results, start=1):
                result['rank_in_set'] = rank
            
            logger.info(f"[RETRIEVAL:METRIC] Retrieved {len(metric_results)} metric chunks "
                        f"(filtered from {len(all_results)} total results)")
            return metric_results
        except Exception as e:
            logger.warning(f"[RETRIEVAL:METRIC] Failed to retrieve metric chunks: {e}")
            return []
    
    def _retrieve_structural_chunks(self,
                                   user_id: str,
                                   query_embedding: List[float],
                                   chat_session_id: Optional[str],
                                   limit: int) -> List[Dict]:
        """Retrieve structural chunks from vector DB (post-filter by chunk_type since it's metadata)"""
        try:
            # Retrieve all results without type filter (chunk_type is metadata, can't filter in Qdrant)
            # We'll filter in Python and then take top limit results
            all_results = self.qd_manager.search(
                user_id=user_id,
                query_embedding=query_embedding,
                chat_session_id=chat_session_id,
                limit=500  # Get many results to ensure we have enough structural chunks after filtering
            )
            
            # DEBUG: Check what chunk_type values we have in the results
            chunk_types_found = {}
            for r in all_results:
                ct = r.get('chunk_type', 'NOT_SET')
                chunk_types_found[ct] = chunk_types_found.get(ct, 0) + 1
            logger.info(f"[RETRIEVAL:STRUCTURAL] Chunk types in 500 results: {chunk_types_found}")
            
            # Post-filter: Keep only structural chunks, then take top limit
            structural_results = [r for r in all_results if r.get('chunk_type') == 'structural'][:limit]
            
            # Add rank information for RRF
            for rank, result in enumerate(structural_results, start=1):
                result['rank_in_set'] = rank
            
            logger.info(f"[RETRIEVAL:STRUCTURAL] Retrieved {len(structural_results)} structural chunks "
                        f"(filtered from {len(all_results)} total results)")
            return structural_results
        except Exception as e:
            logger.warning(f"[RETRIEVAL:STRUCTURAL] Failed to retrieve structural chunks: {e}")
            return []
    
    def _retrieve_metric_chunks_metadata_only(self,
                                             user_id: str,
                                             chat_session_id: Optional[str],
                                             limit: int) -> List[Dict]:
        """Retrieve ALL metric-centric chunks from a chat session WITHOUT embedding search
        
        Used for generic queries where user wants entire file analysis.
        Retrieves directly from metadata without similarity scoring.
        """
        try:
            logger.info(f"[RETRIEVAL:METADATA] Retrieving metric chunks for session {chat_session_id}")
            
            # Use Qdrant search with NO query embedding to get all points from session
            # This retrieves all chunks for the chat session without semantic filtering
            all_results = self.qd_manager.search(
                user_id=user_id,
                query_embedding=None,  # No embedding - get all from session
                chat_session_id=chat_session_id,
                limit=500  # Get all available
            )
            
            # Filter to only metric chunks
            metric_results = [r for r in all_results if r.get('chunk_type') == 'metric_centric'][:limit]
            
            logger.info(f"[RETRIEVAL:METADATA] Retrieved {len(metric_results)} metric chunks "
                       f"from session (metadata-based, no scoring)")
            return metric_results
        except Exception as e:
            logger.warning(f"[RETRIEVAL:METADATA] Failed to retrieve metric chunks (metadata): {e}")
            return []
    
    def _retrieve_structural_chunks_metadata_only(self,
                                                 user_id: str,
                                                 chat_session_id: Optional[str],
                                                 limit: int) -> List[Dict]:
        """Retrieve ALL structural chunks from a chat session WITHOUT embedding search
        
        Used for generic queries where user wants entire file analysis.
        Retrieves directly from metadata without similarity scoring.
        """
        try:
            logger.info(f"[RETRIEVAL:METADATA] Retrieving structural chunks for session {chat_session_id}")
            
            # Use Qdrant search with NO query embedding to get all points from session
            all_results = self.qd_manager.search(
                user_id=user_id,
                query_embedding=None,  # No embedding - get all from session
                chat_session_id=chat_session_id,
                limit=500  # Get all available
            )
            
            # Filter to only structural chunks
            structural_results = [r for r in all_results if r.get('chunk_type') == 'structural'][:limit]
            
            logger.info(f"[RETRIEVAL:METADATA] Retrieved {len(structural_results)} structural chunks "
                       f"from session (metadata-based, no scoring)")
            return structural_results
        except Exception as e:
            logger.warning(f"[RETRIEVAL:METADATA] Failed to retrieve structural chunks (metadata): {e}")
            return []
    
    def format_for_agent(self, results: List[Dict]) -> Tuple[str, List[Dict]]:
        """
        Format retrieval results for agent processing
        
        Args:
            results: Retrieved and processed results
            
        Returns:
            (formatted_text, metadata) for agent consumption
        """
        logger.info(f"[RETRIEVAL:FORMAT] Formatting {len(results)} results for agent")
        
        # Separate by type
        metric_chunks = [r for r in results if r.get('chunk_type') == 'metric_centric']
        structural_chunks = [r for r in results if r.get('chunk_type') == 'structural']
        
        # Format text with hierarchy
        formatted_parts = []
        
        if metric_chunks:
            formatted_parts.append("=== METRIC-CENTRIC ANALYSIS ===\n")
            for chunk in metric_chunks:
                metric_name = chunk.get('metric_name', 'Unknown')
                score = chunk.get('rrf_score', chunk.get('score', 0))
                formatted_parts.append(f"\n[{metric_name} | Score: {score:.3f}]\n")
                formatted_parts.append(chunk.get('text', ''))
        
        if structural_chunks:
            formatted_parts.append("\n\n=== SUPPORTING CONTEXT ===\n")
            for chunk in structural_chunks:
                score = chunk.get('rrf_score', chunk.get('score', 0))
                formatted_parts.append(f"\n[Chunk {chunk.get('chunk_index', '?')} | Score: {score:.3f}]\n")
                formatted_parts.append(chunk.get('text', ''))
        
        formatted_text = "\n".join(formatted_parts)
        
        # Metadata for tracking
        metadata = {
            'total_results': len(results),
            'metric_chunks': len(metric_chunks),
            'structural_chunks': len(structural_chunks),
            'avg_rrf_score': sum(r.get('rrf_score', 0) for r in results) / len(results) if results else 0
        }
        
        logger.info(f"[RETRIEVAL:FORMAT] ✓ Formatted: "
                   f"metrics={len(metric_chunks)}, "
                   f"structural={len(structural_chunks)}")
        
        return formatted_text, metadata
    
    def _retrieve_metric_chunks_global(self, query_embedding: List[float], limit: int) -> List[Dict]:
        """
        Retrieve metric-centric chunks from global collection (fallback when user collection empty)
        
        Args:
            query_embedding: Query embedding vector
            limit: Number of results to return
            
        Returns:
            List of metric chunks from global collection
        """
        try:
            logger.info(f"[RETRIEVAL:GLOBAL:METRIC] Searching global collection for metric chunks")
            # Search without user_id filter to get global results
            global_results = self.qd_manager.search_global(
                query_embedding=query_embedding,
                limit=limit * 2,  # Get more to filter for metric chunks
                user_id=None  # Global search
            )
            
            # Post-filter: Keep only metric_centric chunks
            metric_results = [r for r in global_results if r.get('chunk_type') == 'metric_centric'][:limit]
            logger.info(f"[RETRIEVAL:GLOBAL:METRIC] Found {len(metric_results)} metric chunks from global collection")
            return metric_results
        except Exception as e:
            logger.warning(f"[RETRIEVAL:GLOBAL:METRIC] Failed to retrieve global metric chunks: {e}")
            return []
    
    def _retrieve_structural_chunks_global(self, query_embedding: List[float], limit: int) -> List[Dict]:
        """
        Retrieve structural chunks from global collection (fallback when user collection empty)
        
        Args:
            query_embedding: Query embedding vector
            limit: Number of results to return
            
        Returns:
            List of structural chunks from global collection
        """
        try:
            logger.info(f"[RETRIEVAL:GLOBAL:STRUCTURAL] Searching global collection for structural chunks")
            # Search without user_id filter to get global results
            global_results = self.qd_manager.search_global(
                query_embedding=query_embedding,
                limit=limit * 2,  # Get more to filter for structural chunks
                user_id=None  # Global search
            )
            
            # Post-filter: Keep only structural chunks
            structural_results = [r for r in global_results if r.get('chunk_type') != 'metric_centric'][:limit]
            logger.info(f"[RETRIEVAL:GLOBAL:STRUCTURAL] Found {len(structural_results)} structural chunks from global collection")
            return structural_results
        except Exception as e:
            logger.warning(f"[RETRIEVAL:GLOBAL:STRUCTURAL] Failed to retrieve global structural chunks: {e}")
            return []

"""
Result Filter - RRF fusion, deduplication, and ranking of search results
Implements: FILTER_AND_RANK node logic
"""

import logging
from typing import List, Dict, Any
from collections import defaultdict

logger = logging.getLogger(__name__)


class ResultFilter:
    """
    Filters and ranks search results using RRF (Reciprocal Rank Fusion) algorithm.
    
    Process:
    1. Assign rankings within each source (semantic/keyword × personal/global)
    2. Calculate RRF score combining all rankings
    3. Weight personal results higher than global
    4. Deduplicate by document ID
    5. Sort by final score
    6. Return top 10 results
    """
    
    def __init__(self, k: int = 60, personal_weight: float = 2.0, 
                 max_results: int = 10, score_threshold: float = 0.0):
        """
        Initialize result filter.
        
        Args:
            k: RRF parameter (default 60 for LLMs, higher k = less weight on low ranks)
            personal_weight: Weight multiplier for personal results vs global (default 2.0)
            max_results: Maximum number of results to return (default 10)
            score_threshold: Minimum RRF score to include (default 0.0)
        """
        self.k = k
        self.personal_weight = personal_weight
        self.max_results = max_results
        self.score_threshold = score_threshold
        self.logger = logging.getLogger(__name__)
    
    def filter_and_rank(self,
                        personal_semantic: List[Dict],
                        personal_keyword: List[Dict],
                        global_semantic: List[Dict] = None,
                        global_keyword: List[Dict] = None) -> List[Dict]:
        """
        Filter and rank results using RRF algorithm.
        
        Args:
            personal_semantic: Semantic search results from personal vectordb
            personal_keyword: Keyword search results from personal vectordb
            global_semantic: Semantic search results from global vectordb (optional)
            global_keyword: Keyword search results from global vectordb (optional)
            
        Returns:
            Ranked list of deduplicated results (top 10)
        """
        global_semantic = global_semantic or []
        global_keyword = global_keyword or []
        
        self.logger.info(
            f"Filtering: {len(personal_semantic)} personal_semantic, "
            f"{len(personal_keyword)} personal_keyword, "
            f"{len(global_semantic)} global_semantic, "
            f"{len(global_keyword)} global_keyword"
        )
        
        # Step 1: Assign rankings and calculate RRF scores
        ranked_results = self._calculate_rrf_scores(
            personal_semantic, personal_keyword,
            global_semantic, global_keyword
        )
        
        # Step 2: Deduplicate by document ID
        deduplicated = self._deduplicate_results(ranked_results)
        self.logger.info(f"After deduplication: {len(deduplicated)} unique results")
        
        # Step 3: Sort by RRF score (descending)
        sorted_results = sorted(
            deduplicated,
            key=lambda x: x['rrf_score'],
            reverse=True
        )
        
        # Step 4: Filter by threshold and limit to max results
        filtered_results = [
            r for r in sorted_results
            if r['rrf_score'] >= self.score_threshold
        ][:self.max_results]
        
        self.logger.info(
            f"Final ranked results: {len(filtered_results)} results "
            f"(threshold: {self.score_threshold}, max: {self.max_results})"
        )
        
        # Log ranking info
        for i, result in enumerate(filtered_results, 1):
            self.logger.debug(
                f"  {i}. [{result['rrf_score']:.4f}] {result.get('source', 'unknown')} "
                f"({result.get('retrieval_source', 'unknown')})"
            )
        
        return filtered_results
    
    def _calculate_rrf_scores(self,
                              personal_semantic: List[Dict],
                              personal_keyword: List[Dict],
                              global_semantic: List[Dict],
                              global_keyword: List[Dict]) -> List[Dict]:
        """
        Calculate RRF scores for all results.
        
        RRF formula: score = sum(1 / (rank + k))
        - rank: position in sorted list (0-indexed)
        - k: constant (60 default for LLMs)
        
        Args:
            Four lists of search results
            
        Returns:
            Combined list with 'rrf_score' field added to each result
        """
        # Dictionary to track scores for each document
        scores_by_doc = defaultdict(lambda: {'rrf_score': 0.0, 'result': None, 'sources': []})
        
        # Process each ranking list
        ranking_lists = [
            ('personal_semantic', personal_semantic, self.personal_weight),
            ('personal_keyword', personal_keyword, self.personal_weight),
            ('global_semantic', global_semantic, 1.0),
            ('global_keyword', global_keyword, 1.0),
        ]
        
        for list_name, results, weight in ranking_lists:
            for rank, result in enumerate(results):
                # Calculate RRF component: weight * (1 / (rank + k))
                rrf_component = weight * (1.0 / (rank + self.k))
                
                # Use result ID as primary key for deduplication
                # This preserves different chunks from same file (they have different IDs)
                doc_id = result.get('id')
                if not doc_id:
                    # Fallback to point_id if available
                    doc_id = result.get('point_id')
                if not doc_id:
                    # Last resort: create unique key from source + metadata to avoid collisions
                    # This ensures each result is treated as unique
                    source = result.get('source', '')
                    title = result.get('title', '')
                    text_hash = abs(hash(result.get('text', result.get('content', ''))[:100])) % 1000000
                    doc_id = f"{source}_{title}_{text_hash}_{rank}".replace(' ', '_')
                
                # Update total score
                scores_by_doc[doc_id]['rrf_score'] += rrf_component
                
                # Store first encountered result (will merge metadata)
                if scores_by_doc[doc_id]['result'] is None:
                    scores_by_doc[doc_id]['result'] = result.copy()
                
                # Track which sources this result came from
                scores_by_doc[doc_id]['sources'].append(list_name)
        
        # Flatten back to list and add RRF score
        combined_results = []
        for doc_id, data in scores_by_doc.items():
            result = data['result']
            result['rrf_score'] = data['rrf_score']
            result['sources_found_in'] = data['sources']  # For debugging
            combined_results.append(result)
        
        self.logger.debug(
            f"Calculated RRF scores for {len(combined_results)} unique documents"
        )
        
        return combined_results
    
    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """
        Deduplicate results by document ID, keeping highest-scoring instance.
        
        Only deduplicates truly identical documents (same ID/point_id).
        Different chunks from the same file are NOT deduplicated (they have different IDs).
        
        Args:
            results: List of results with 'rrf_score' field
            
        Returns:
            Deduplicated list
        """
        seen = {}
        
        for result in results:
            # Use document/point ID as key - this is unique per chunk
            doc_id = result.get('id') or result.get('point_id')
            if not doc_id:
                # If no ID, this is from a different source - don't deduplicate it
                # Generate a stable key from the generated ID we created earlier
                doc_id = f"{result.get('source', '')}_{result.get('title', '')}"
            
            # Keep highest scoring version
            if doc_id not in seen or result['rrf_score'] > seen[doc_id]['rrf_score']:
                seen[doc_id] = result
        
        deduplicated = list(seen.values())
        removed_count = len(results) - len(deduplicated)
        
        if removed_count > 0:
            self.logger.info(f"Deduplication removed {removed_count} duplicate results")
        
        return deduplicated
    
    def set_parameters(self, k: int = None, personal_weight: float = None,
                       max_results: int = None, score_threshold: float = None) -> None:
        """
        Update filter parameters.
        
        Args:
            k: RRF parameter
            personal_weight: Weight multiplier for personal results
            max_results: Maximum results to return
            score_threshold: Minimum score threshold
        """
        if k is not None:
            self.k = k
        if personal_weight is not None:
            self.personal_weight = personal_weight
        if max_results is not None:
            self.max_results = max_results
        if score_threshold is not None:
            self.score_threshold = score_threshold
        
        self.logger.info(
            f"Filter parameters updated: k={self.k}, personal_weight={self.personal_weight}, "
            f"max_results={self.max_results}, score_threshold={self.score_threshold}"
        )


def rrf_score(rank: int, k: int = 60) -> float:
    """
    Calculate RRF score for a single ranking.
    
    Formula: 1 / (rank + k)
    
    Args:
        rank: Position in ranking (0-indexed)
        k: RRF constant (default 60 for LLMs)
        
    Returns:
        RRF score (0.0-1.0)
    """
    return 1.0 / (rank + k)

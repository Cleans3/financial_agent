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
        
        separator = "|" * 25
        self.logger.info(f"{separator} FILTER START {separator}")
        
        # Log input details with character counts
        input_sources = [
            ('personal_semantic', personal_semantic),
            ('personal_keyword', personal_keyword),
            ('global_semantic', global_semantic),
            ('global_keyword', global_keyword),
        ]
        
        total_input_chars = 0
        for source_name, results in input_sources:
            source_char_count = sum(
                len(r.get('text', '') or r.get('content', ''))
                for r in results
            )
            total_input_chars += source_char_count
            self.logger.info(
                f"📥 {source_name}: {len(results)} results, {source_char_count:,} chars"
            )
        
        self.logger.info(
            f"📊 Total input: {len(personal_semantic) + len(personal_keyword) + len(global_semantic) + len(global_keyword)} results, "
            f"{total_input_chars:,} total chars"
        )
        
        # Step 1: Assign rankings and calculate RRF scores
        self.logger.info("📍 STEP 1: CALCULATE RRF SCORES")
        ranked_results = self._calculate_rrf_scores(
            personal_semantic, personal_keyword,
            global_semantic, global_keyword
        )
        self.logger.info(f"✅ RRF calculated: {len(ranked_results)} unique document IDs")
        
        # Step 2: Deduplicate by document ID
        self.logger.info("📍 STEP 2: DEDUPLICATE RESULTS")
        deduplicated = self._deduplicate_results(ranked_results)
        self.logger.info(f"✅ After deduplication: {len(deduplicated)} results")
        
        # Step 3: Sort by RRF score (descending)
        self.logger.info("📍 STEP 3: SORT BY RRF SCORE")
        sorted_results = sorted(
            deduplicated,
            key=lambda x: x['rrf_score'],
            reverse=True
        )
        self.logger.info(f"✅ Sorted by RRF score (descending)")
        
        # Step 4: Filter by threshold and limit to max results
        self.logger.info(f"📍 STEP 4: APPLY THRESHOLD & LIMIT")
        self.logger.info(f"   Score threshold: {self.score_threshold}")
        self.logger.info(f"   Max results: {self.max_results}")
        
        filtered_results = [
            r for r in sorted_results
            if r['rrf_score'] >= self.score_threshold
        ][:self.max_results]
        
        # Calculate total output characters
        output_char_count = sum(
            len(r.get('text', '') or r.get('content', ''))
            for r in filtered_results
        )
        output_byte_count = output_char_count * 2
        
        self.logger.info(
            f"✅ Final results: {len(filtered_results)} results, "
            f"{output_char_count:,} chars (~{output_byte_count:,} bytes)"
        )
        
        # Log detailed ranking info
        self.logger.info("\n📊 FINAL RANKED RESULTS:")
        for i, result in enumerate(filtered_results, 1):
            content = result.get('text', '') or result.get('content', '')
            content_len = len(content)
            source = result.get('source', 'unknown')
            retrieval_source = result.get('retrieval_source', 'unknown')
            sources_found = ', '.join(result.get('sources_found_in', []))
            
            self.logger.info(
                f"  {i}. RRF={result['rrf_score']:.4f} | {content_len:,} chars | "
                f"{source} | {retrieval_source} | sources: [{sources_found}]"
            )
        
        self.logger.info(f"{separator} FILTER COMPLETE {separator}\n")
        
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
        - weight: personal results weighted higher (2.0x)
        
        Args:
            Four lists of search results
            
        Returns:
            Combined list with 'rrf_score' field added to each result
        """
        self.logger.debug(f"   RRF parameters: k={self.k}, personal_weight={self.personal_weight}")
        
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
            self.logger.debug(f"   Processing {list_name} (weight={weight}x): {len(results)} results")
            
            for rank, result in enumerate(results):
                # Calculate RRF component: weight * (1 / (rank + k))
                rrf_component = weight * (1.0 / (rank + self.k))
                
                # Use result ID as primary key for deduplication
                # This preserves different chunks from same file (they have different IDs)
                doc_id = result.get('id')
                if doc_id is not None:
                    doc_id = str(doc_id)  # Ensure it's a string (point IDs are ints)
                if not doc_id:
                    # Fallback to point_id if available
                    doc_id = result.get('point_id')
                    if doc_id is not None:
                        doc_id = str(doc_id)  # Ensure it's a string
                if not doc_id:
                    # Last resort: create unique key from source + metadata to avoid collisions
                    # This ensures each result is treated as unique
                    source = result.get('source', '')
                    title = result.get('title', '')
                    text_content = result.get('text', '') or result.get('content', '')
                    text_hash = abs(hash(text_content[:100])) % 1000000
                    doc_id = f"{source}_{title}_{text_hash}_{rank}".replace(' ', '_')
                
                # Update total score
                scores_by_doc[doc_id]['rrf_score'] += rrf_component
                
                # Store first encountered result (will merge metadata)
                if scores_by_doc[doc_id]['result'] is None:
                    scores_by_doc[doc_id]['result'] = result.copy()
                    # Log first encounter of this document
                    content_len = len(result.get('text', '') or result.get('content', ''))
                    self.logger.debug(
                        f"     [new] rank={rank}, doc_id={doc_id[:50]}..., "
                        f"rrf_component={rrf_component:.4f}, content={content_len} chars"
                    )
                else:
                    # This doc was already seen from another source
                    self.logger.debug(
                        f"     [dup] rank={rank}, doc_id={doc_id[:50]}..., "
                        f"adding rrf_component={rrf_component:.4f} (new total={scores_by_doc[doc_id]['rrf_score']:.4f})"
                    )
                
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
        self.logger.info(f"   🔍 DEDUPLICATION CHECK: Input {len(results)} results")
        self.logger.info(f"   (Checking document IDs to find true duplicates)")
        
        seen = {}
        duplicates_found = {}
        detailed_decisions = []
        
        for idx, result in enumerate(results, 1):
            # Use document/point ID as key - this is unique per chunk
            doc_id = result.get('id') or result.get('point_id')
            if doc_id is not None:
                doc_id = str(doc_id)  # Ensure it's a string (point IDs are ints)
            if not doc_id:
                # If no ID, this is from a different source - don't deduplicate it
                # Generate a stable key from the generated ID we created earlier
                doc_id = f"{result.get('source', '')}_{result.get('title', '')}"
            
            content_len = len(result.get('text', '') or result.get('content', ''))
            rrf_score = result.get('rrf_score', 0)
            sources_found = ', '.join(result.get('sources_found_in', []))
            
            # Keep highest scoring version
            if doc_id not in seen:
                seen[doc_id] = result
                decision = f"     ✅ [KEEP-NEW] #{idx} doc_id={doc_id[:45]}... | RRF={rrf_score:.4f} | {content_len} chars | sources=[{sources_found}]"
                self.logger.info(decision)
                detailed_decisions.append(decision)
            else:
                # This is a duplicate
                old_score = seen[doc_id]['rrf_score']
                new_score = rrf_score
                old_content_len = len(seen[doc_id].get('text', '') or seen[doc_id].get('content', ''))
                
                if new_score > old_score:
                    decision = (
                        f"     🔄 [REPLACE] #{idx} doc_id={doc_id[:45]}...\n"
                        f"        OLD: RRF={old_score:.4f} | {old_content_len} chars | "
                        f"sources=[{', '.join(seen[doc_id].get('sources_found_in', []))}]\n"
                        f"        NEW: RRF={new_score:.4f} | {content_len} chars | "
                        f"sources=[{sources_found}]\n"
                        f"        REASON: New RRF score ({new_score:.4f}) is HIGHER than old ({old_score:.4f})"
                    )
                    self.logger.info(decision)
                    detailed_decisions.append(decision)
                    seen[doc_id] = result
                else:
                    decision = (
                        f"     ❌ [DELETE] #{idx} doc_id={doc_id[:45]}...\n"
                        f"        OLD: RRF={old_score:.4f} | {old_content_len} chars | "
                        f"sources=[{', '.join(seen[doc_id].get('sources_found_in', []))}]\n"
                        f"        NEW: RRF={new_score:.4f} | {content_len} chars | "
                        f"sources=[{sources_found}]\n"
                        f"        REASON: Old RRF score ({old_score:.4f}) is EQUAL-OR-HIGHER, keeping first occurrence"
                    )
                    self.logger.info(decision)
                    detailed_decisions.append(decision)
                
                if doc_id not in duplicates_found:
                    duplicates_found[doc_id] = 1
                else:
                    duplicates_found[doc_id] += 1
        
        deduplicated = list(seen.values())
        removed_count = len(results) - len(deduplicated)
        
        self.logger.info(f"")
        self.logger.info(f"   📊 DEDUPLICATION RESULT:")
        self.logger.info(f"     Input: {len(results)} results")
        self.logger.info(f"     Output: {len(deduplicated)} results")
        if removed_count > 0:
            self.logger.info(
                f"     Deleted: {removed_count} duplicates "
                f"(found {len(duplicates_found)} doc IDs with duplicates)"
            )
        else:
            self.logger.info(f"     Deleted: 0 (no duplicates found - all results have unique doc IDs)")
        
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

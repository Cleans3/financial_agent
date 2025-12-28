"""
Summary Tool Node - Router for applying summary techniques in agent workflow

Workflow:
1. Retrieve metric-centric + structural chunks
2. Classify query to determine if summary needed
3. If summary needed: select technique -> apply summary -> return summary
4. If no summary: pass through chunks to reformulation
5. Query reformulation gets summary (not raw chunks) -> LLM generates answer

This ensures not all queries go through summary (only relevant ones)
"""

import logging
from typing import List, Dict, Optional, Tuple

from src.services.advanced_summary_tools import (
    select_summary_technique,
    apply_summary_tool,
    SummaryTechnique
)

logger = logging.getLogger(__name__)


class SummaryToolRouter:
    """
    Routes queries to summary tools when appropriate
    
    Logic:
    - Rules-first: Check if query is summary-related
    - LLM-second: If unclear, LLM can classify
    - Always preserve original chunks as fallback
    """
    
    SUMMARY_KEYWORDS = [
        # English
        "summary", "summarize", "overview", "takeaway", "headline",
        "change", "accelerat", "deceler", "vs", "versus", "compared",
        "unusual", "anomaly", "concern", "risk", "red flag",
        "what happened", "story", "narrative", "explain", "how did",
        "sustainable", "profitable", "quality", "guidance", "beat", "miss",
        # Vietnamese
        "tóm tắt", "tóm tắt lại", "tổng hợp", "phân tích", "so sánh",
        "thay đổi", "tăng", "giảm", "so với", "bất thường", "lạ",
        "điều gì xảy ra", "kể chuyện", "câu chuyện", "giải thích", "tại sao",
        "bền vững", "lợi nhuận", "chất lượng", "hướng", "hướng dẫn", "vượt quá"
    ]
    
    @staticmethod
    def should_summarize(query: str) -> bool:
        """
        Determine if query should trigger summary tools
        
        Args:
            query: User query (supports English and Vietnamese)
            
        Returns:
            True if query is summary-related
        """
        query_lower = query.lower()
        return any(kw in query_lower for kw in SummaryToolRouter.SUMMARY_KEYWORDS)
    
    @staticmethod
    def route(query: str,
              retrieved_chunks: List[Dict],
              llm_classifier=None) -> Tuple[bool, Optional[Dict], List[Dict]]:
        """
        Route query to summary tools if appropriate
        
        Args:
            query: User query
            retrieved_chunks: Retrieved metric + structural chunks
            llm_classifier: Optional LLM for complex classification
            
        Returns:
            (should_summarize, summary_result, chunks) where:
            - should_summarize: Bool indicating if summary was applied
            - summary_result: Summary result dict if applied, None otherwise
            - chunks: Original chunks (preserved as fallback)
        """
        logger.info(f"[SUMMARY:ROUTE] Routing query: {query[:80]}...")
        
        # Step 1: Rules-first check
        needs_summary = SummaryToolRouter.should_summarize(query)
        logger.info(f"[SUMMARY:ROUTE] Rules-based check: needs_summary={needs_summary}")
        
        # Step 2: If rules are unclear, use LLM (optional)
        if not needs_summary and llm_classifier:
            logger.info("[SUMMARY:ROUTE] Rules inconclusive, consulting LLM for classification")
            try:
                # Use LLM to classify if query needs summary
                llm_prompt = f"""Given this user query, determine if the user is asking for a summary, analysis, or explanation of financial data.

Query: "{query}"

Respond with ONLY "yes" or "no" (no explanation).
- "yes" if the user wants: summary, analysis, takeaway, explanation, what changed, story, narrative, comparison, anomalies, unusual patterns
- "no" if the user wants: specific values, particular metrics, raw data, simple lookup"""
                
                llm_response = llm_classifier.invoke(llm_prompt)
                response_text = (llm_response.content if hasattr(llm_response, 'content') else str(llm_response)).strip().lower()
                needs_summary = "yes" in response_text
                logger.info(f"[SUMMARY:ROUTE] LLM classification result: needs_summary={needs_summary}")
            except Exception as e:
                logger.warning(f"[SUMMARY:ROUTE] LLM classification failed: {e}, using rules result")
        
        # Step 3: If summary needed, extract metric texts and apply technique
        if needs_summary:
            logger.info("[SUMMARY:ROUTE] ✓ Summary technique will be applied")
            
            # Extract metric-centric chunks for summary
            metric_chunks = [c for c in retrieved_chunks if c.get('chunk_type') == 'metric_centric']
            metric_texts = [c.get('text', '') for c in metric_chunks]
            
            if metric_texts:
                # Select appropriate technique
                technique = select_summary_technique(query, retrieved_chunks)
                logger.info(f"[SUMMARY:ROUTE] Selected technique: {technique.value}")
                
                # Apply summary
                summary_result = apply_summary_tool(technique, metric_texts, query)
                logger.info(f"[SUMMARY:ROUTE] ✓ Summary applied successfully")
                
                return True, summary_result, retrieved_chunks
            else:
                logger.info("[SUMMARY:ROUTE] No metric chunks available, but summary requested - will use structural chunks")
                # Fallback: Use all chunks for summary if metric chunks unavailable
                all_texts = [c.get('text', '') for c in retrieved_chunks if c.get('text')]
                if all_texts:
                    technique = select_summary_technique(query, retrieved_chunks)
                    summary_result = apply_summary_tool(technique, all_texts, query)
                    logger.info(f"[SUMMARY:ROUTE] ✓ Summary applied using structural chunks")
                    return True, summary_result, retrieved_chunks
                else:
                    logger.warning("[SUMMARY:ROUTE] No chunks available for summary")
                    return False, None, retrieved_chunks
        else:
            logger.info("[SUMMARY:ROUTE] No summary needed, passing through chunks")
            return False, None, retrieved_chunks


class AgentWorkflowIntegration:
    """
    Integration point for summary tools in agent workflow
    
    Workflow:
    1. Query comes in
    2. Retrieve chunks (structural + metric-centric with RRF/dedup)
    3. Route to summary tools if appropriate
    4. Format for agent:
       - If summary applied: use summary text as input to reformulation
       - If no summary: use formatted chunks as input
    5. Query reformulation processes result
    6. LLM generates final answer
    """
    
    def __init__(self, retrieval_service, qdrant_manager):
        """
        Initialize agent integration
        
        Args:
            retrieval_service: AdvancedRetrievalService instance
            qdrant_manager: QdrantCollectionManager instance
        """
        self.retrieval = retrieval_service
        self.qd_manager = qdrant_manager
        self.router = SummaryToolRouter()
        logger.info("[AGENT:INTEGRATION] Initialized with summary tools")
    
    def process_query_with_summary(self,
                                  user_id: str,
                                  query: str,
                                  query_embedding: List[float],
                                  chat_session_id: Optional[str] = None,
                                  llm_classifier=None) -> Dict:
        """
        Process query with advanced retrieval and optional summary
        
        Args:
            user_id: User ID
            query: User query
            query_embedding: Query embedding vector
            chat_session_id: Chat session ID
            llm_classifier: Optional LLM for classification
            
        Returns:
            Dictionary with:
            - retrieved_chunks: Original retrieved chunks
            - summary_applied: Whether summary was applied
            - summary_result: Summary (if applied)
            - formatted_input: Text to send to LLM for reformulation
            - metadata: Processing metadata
        """
        logger.info(f"[AGENT:PROCESS] Processing query with summary pipeline")
        
        # Step 1: Retrieve with advanced retrieval (RRF + dedup)
        logger.info("[AGENT:PROCESS] Step 1: Advanced retrieval")
        retrieved_chunks = self.retrieval.retrieve(
            user_id=user_id,
            query=query,
            query_embedding=query_embedding,
            chat_session_id=chat_session_id,
            limit=20  # Get more chunks since RRF will handle dedup
        )
        logger.info(f"[AGENT:PROCESS] ✓ Retrieved {len(retrieved_chunks)} chunks")
        
        # Step 2: Route to summary tools
        logger.info("[AGENT:PROCESS] Step 2: Summary tool routing")
        summary_applied, summary_result, chunks = self.router.route(
            query=query,
            retrieved_chunks=retrieved_chunks,
            llm_classifier=llm_classifier
        )
        
        # Step 3: Format input for LLM
        logger.info("[AGENT:PROCESS] Step 3: Format for LLM")
        if summary_applied and summary_result:
            # Use summary as input
            formatted_input = summary_result.get('summary', '')
            logger.info("[AGENT:PROCESS] ✓ Using summary as input to LLM")
        else:
            # Use formatted chunks
            formatted_text, format_metadata = self.retrieval.format_for_agent(chunks)
            formatted_input = formatted_text
            logger.info(f"[AGENT:PROCESS] ✓ Using {format_metadata['total_results']} chunks as input")
        
        result = {
            "query": query,
            "retrieved_chunks": chunks,
            "summary_applied": summary_applied,
            "summary_result": summary_result,
            "formatted_input": formatted_input,
            "num_chunks": len(chunks),
            "metadata": {
                "summary_technique": summary_result.get('technique') if summary_result else None,
                "chunks_by_type": {
                    "metric_centric": len([c for c in chunks if c.get('chunk_type') == 'metric_centric']),
                    "structural": len([c for c in chunks if c.get('chunk_type') == 'structural'])
                }
            }
        }
        
        logger.info(f"[AGENT:PROCESS] ✓ Query processing complete")
        return result
    
    def format_for_reformulation(self, processing_result: Dict) -> str:
        """
        Format processing result for query reformulation step
        
        Args:
            processing_result: Result from process_query_with_summary
            
        Returns:
            Formatted text for reformulation -> LLM
        """
        logger.info("[AGENT:FORMAT] Formatting for reformulation")
        
        # The formatted_input is already prepared
        # This is just a wrapper for clarity
        
        summary_info = ""
        if processing_result.get('summary_applied'):
            summary_result = processing_result.get('summary_result', {})
            summary_info = f"\n[Summary Technique: {summary_result.get('technique')}]\n"
        
        formatted = summary_info + processing_result.get('formatted_input', '')
        
        logger.info(f"[AGENT:FORMAT] ✓ Formatted {len(formatted)} chars for reformulation")
        return formatted

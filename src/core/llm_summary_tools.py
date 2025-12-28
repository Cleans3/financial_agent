"""
LLM-Driven Summary Tools - Summary techniques as callable LLM tools

These tools are similar to regular tools (get_company_info, calculate_rsi, etc.)
but are designed to summarize financial data using different techniques.

The LLM can select any of these tools when it detects the user is asking for
a summary, analysis, anomaly detection, etc.

Each tool:
1. Takes retrieval results + user query as input
2. Returns structured summary output
3. Is passed to query_reformulation which combines with structural chunks
4. Final answer generated from summary + structural chunks (not raw retrieval)
"""

import logging
from typing import List, Dict, Optional, Any
from src.services.advanced_summary_tools import (
    SummaryTechnique,
    ComparativeAnalysisSummarizer,
    AnomalyDetectionSummarizer,
    MaterialityWeightedSummarizer,
    NarrativeArcSummarizer,
    KeyQuestionsAnsweringSummarizer
)

logger = logging.getLogger(__name__)


class SummaryToolsProvider:
    """Provides callable summary tools for LLM usage"""
    
    @staticmethod
    def comparative_analysis_tool(chunks: List[Dict], query: str) -> Dict[str, Any]:
        """
        LLM-callable tool: Comparative Analysis Summarization
        
        Highlights WHAT CHANGED rather than just stating values.
        Example: "Revenue accelerated 12.1%" vs "Revenue was $500M"
        
        Args:
            chunks: Retrieved document chunks (metric-centric + structural)
            query: Original user query
            
        Returns:
            Dictionary with comparative analysis summary
        """
        logger.info("[SUMMARY:TOOL] Executing comparative_analysis tool")
        logger.info(f"[SUMMARY:TOOL] Received {len(chunks)} chunks for processing")
        
        try:
            # DEBUG: Log chunk structure and chunk_type values
            logger.info("[SUMMARY:TOOL] Analyzing chunk structure:")
            for i, chunk in enumerate(chunks[:3]):  # Log first 3 chunks
                chunk_type = chunk.get('chunk_type', 'NOT_SET')
                chunk_keys = list(chunk.keys())
                logger.info(f"  Chunk {i}: chunk_type={chunk_type}, keys={chunk_keys[:5]}...")
            
            # Extract metric texts from chunks
            metric_texts = [chunk.get('text', '') for chunk in chunks if chunk.get('chunk_type') == 'metric_centric']
            logger.info(f"[SUMMARY:TOOL] Found {len(metric_texts)} metric chunks (checking for 'metric_centric' type)")
            
            if not metric_texts:
                logger.warning("[SUMMARY:TOOL] No metric chunks available for comparative analysis")
                # Log all chunk_types for debugging
                chunk_types = set([chunk.get('chunk_type', 'NOT_SET') for chunk in chunks])
                logger.warning(f"[SUMMARY:TOOL] Chunk types found in input: {chunk_types}")
                # Fallback to structural chunks
                metric_texts = [chunk.get('text', '') for chunk in chunks]
            
            # Apply comparative analysis
            result = ComparativeAnalysisSummarizer.summarize(metric_texts, query)
            
            logger.info(f"[SUMMARY:TOOL] ✓ Comparative analysis: {len(result.get('insights', []))} insights")
            return {
                "success": True,
                "tool_name": "comparative_analysis",
                "summary": result.get('summary', ''),
                "insights": result.get('insights', []),
                "metrics_analyzed": result.get('metrics_analyzed', 0),
                "technique": "Comparative Analysis - Highlights what changed"
            }
        except Exception as e:
            logger.error(f"[SUMMARY:TOOL] Comparative analysis failed: {e}")
            return {"success": False, "error": str(e), "tool_name": "comparative_analysis"}
    
    @staticmethod
    def anomaly_detection_tool(chunks: List[Dict], query: str) -> Dict[str, Any]:
        """
        LLM-callable tool: Anomaly Detection Summarization
        
        Automatically flags unusual patterns:
        - Driver reconciliation mismatches
        - Historical anomalies (>15% deviation from 3-year avg)
        - Guidance beats/misses
        - Contra-intuitive patterns (revenue up but customers down)
        
        Args:
            chunks: Retrieved document chunks
            query: Original user query
            
        Returns:
            Dictionary with detected anomalies
        """
        logger.info("[SUMMARY:TOOL] Executing anomaly_detection tool")
        
        try:
            metric_texts = [chunk.get('text', '') for chunk in chunks if chunk.get('chunk_type') == 'metric_centric']
            if not metric_texts:
                metric_texts = [chunk.get('text', '') for chunk in chunks]
            
            result = AnomalyDetectionSummarizer.summarize(metric_texts, query)
            
            logger.info(f"[SUMMARY:TOOL] ✓ Anomaly detection: {len(result.get('anomalies', []))} anomalies found")
            return {
                "success": True,
                "tool_name": "anomaly_detection",
                "summary": result.get('summary', ''),
                "anomalies": result.get('anomalies', []),
                "confidence_scores": result.get('confidence_scores', {}),
                "technique": "Anomaly Detection - Flags unusual patterns and mismatches"
            }
        except Exception as e:
            logger.error(f"[SUMMARY:TOOL] Anomaly detection failed: {e}")
            return {"success": False, "error": str(e), "tool_name": "anomaly_detection"}
    
    @staticmethod
    def materiality_weighted_tool(chunks: List[Dict], query: str) -> Dict[str, Any]:
        """
        LLM-callable tool: Materiality-Weighted Summarization
        
        Scores each metric by importance and allocates summary space:
        - 50% to top metrics
        - 35% to medium-importance metrics
        - 15% to low-importance metrics
        
        Args:
            chunks: Retrieved document chunks
            query: Original user query
            
        Returns:
            Dictionary with materiality-weighted summary
        """
        logger.info("[SUMMARY:TOOL] Executing materiality_weighted tool")
        
        try:
            metric_texts = [chunk.get('text', '') for chunk in chunks if chunk.get('chunk_type') == 'metric_centric']
            if not metric_texts:
                metric_texts = [chunk.get('text', '') for chunk in chunks]
            
            result = MaterialityWeightedSummarizer.summarize(metric_texts, query)
            
            logger.info(f"[SUMMARY:TOOL] ✓ Materiality-weighted: {len(result.get('sections', {}).keys())} metric groups")
            return {
                "success": True,
                "tool_name": "materiality_weighted",
                "summary": result.get('summary', ''),
                "sections": result.get('sections', {}),
                "weighting_ratios": result.get('weighting_ratios', {}),
                "technique": "Materiality-Weighted - Allocates space by metric importance (50/35/15)"
            }
        except Exception as e:
            logger.error(f"[SUMMARY:TOOL] Materiality-weighted failed: {e}")
            return {"success": False, "error": str(e), "tool_name": "materiality_weighted"}
    
    @staticmethod
    def narrative_arc_tool(chunks: List[Dict], query: str) -> Dict[str, Any]:
        """
        LLM-callable tool: Narrative Arc Summarization
        
        Structures summary as a story:
        - Setup: Context (what was the starting position?)
        - Conflict: What changed (what challenges/opportunities emerged?)
        - Resolution: What it means (what are the implications?)
        
        Args:
            chunks: Retrieved document chunks
            query: Original user query
            
        Returns:
            Dictionary with narrative arc summary
        """
        logger.info("[SUMMARY:TOOL] Executing narrative_arc tool")
        
        try:
            metric_texts = [chunk.get('text', '') for chunk in chunks if chunk.get('chunk_type') == 'metric_centric']
            if not metric_texts:
                metric_texts = [chunk.get('text', '') for chunk in chunks]
            
            result = NarrativeArcSummarizer.summarize(metric_texts, query)
            
            logger.info(f"[SUMMARY:TOOL] ✓ Narrative arc: {len(result.get('sections', {}))} story sections")
            return {
                "success": True,
                "tool_name": "narrative_arc",
                "summary": result.get('summary', ''),
                "sections": result.get('sections', {}),
                "narrative_flow": result.get('narrative_flow', ''),
                "technique": "Narrative Arc - Structures as Setup -> Conflict -> Resolution story"
            }
        except Exception as e:
            logger.error(f"[SUMMARY:TOOL] Narrative arc failed: {e}")
            return {"success": False, "error": str(e), "tool_name": "narrative_arc"}
    
    @staticmethod
    def key_questions_tool(chunks: List[Dict], query: str) -> Dict[str, Any]:
        """
        LLM-callable tool: Key Questions Answerer
        
        Directly answers what analysts ask:
        - Is growth fast? Is it accelerating?
        - Are margins expanding?
        - Is growth profitable?
        - Is this sustainable?
        - Did they beat guidance?
        
        Args:
            chunks: Retrieved document chunks
            query: Original user query
            
        Returns:
            Dictionary with answers to key questions
        """
        logger.info("[SUMMARY:TOOL] Executing key_questions tool")
        
        try:
            metric_texts = [chunk.get('text', '') for chunk in chunks if chunk.get('chunk_type') == 'metric_centric']
            if not metric_texts:
                metric_texts = [chunk.get('text', '') for chunk in chunks]
            
            result = KeyQuestionsAnsweringSummarizer.summarize(metric_texts, query)
            
            logger.info(f"[SUMMARY:TOOL] ✓ Key questions: {len(result.get('answers', {}))} answers provided")
            return {
                "success": True,
                "tool_name": "key_questions",
                "summary": result.get('summary', ''),
                "answers": result.get('answers', {}),
                "confidence_scores": result.get('confidence_scores', {}),
                "technique": "Key Questions - Answers: Growth? Margins? Profitability? Sustainability? Guidance?"
            }
        except Exception as e:
            logger.error(f"[SUMMARY:TOOL] Key questions failed: {e}")
            return {"success": False, "error": str(e), "tool_name": "key_questions"}
    
    @staticmethod
    def get_all_tools() -> Dict[str, callable]:
        """Get all summary tools as callable functions"""
        return {
            "comparative_analysis": SummaryToolsProvider.comparative_analysis_tool,
            "anomaly_detection": SummaryToolsProvider.anomaly_detection_tool,
            "materiality_weighted": SummaryToolsProvider.materiality_weighted_tool,
            "narrative_arc": SummaryToolsProvider.narrative_arc_tool,
            "key_questions": SummaryToolsProvider.key_questions_tool,
        }
    
    @staticmethod
    def get_tool_descriptions() -> Dict[str, str]:
        """Get descriptions for LLM tool selection"""
        return {
            "comparative_analysis": (
                "Highlights what CHANGED ('Revenue accelerated 12.1%') rather than just stating values "
                "('Revenue was $500M'). Use when user asks for changes, trends, growth, acceleration."
            ),
            "anomaly_detection": (
                "Flags unusual patterns: driver mismatches, historical anomalies (>15% deviation), "
                "guidance beats/misses, contra-intuitive patterns. Use when user asks about anomalies, "
                "concerns, risks, unusual patterns."
            ),
            "materiality_weighted": (
                "Allocates summary space by importance: 50% top metrics, 35% medium, 15% low. "
                "Use when user needs a balanced, importance-weighted analysis of multiple metrics."
            ),
            "narrative_arc": (
                "Structures as a story: Setup (context) -> Conflict (what changed) -> Resolution (implications). "
                "Use when user wants a narrative, story-like explanation of financial performance."
            ),
            "key_questions": (
                "Directly answers what analysts ask: Growth? Acceleration? Margins expanding? "
                "Profitable growth? Sustainable? Guidance beat? Use when user asks fundamental investment questions."
            ),
        }

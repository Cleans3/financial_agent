"""
LLM-Driven Summary Tools - 4 Summary Strategies for Financial Analysis

These tools summarize financial data using 4 different factual approaches:

1. STRUCTURED DATA SUMMARY: Organize facts by category (headlines, production, workforce, etc.)
2. METRIC CONDENSING: Reduce to absolute essentials (one-line per metric, no explanation)
3. CONSTRAINT LISTING: List which constraints are binding vs slack (status only, no interpretation)
4. FEASIBILITY CHECK: Simple YES/NO feasibility check with any violations if found

Key principle: FACTS ONLY
- No interpretation ("this means...", "it's important...")
- No recommendations
- No explanations beyond what the data shows
- Just organized facts in different formats

Each tool:
1. Takes retrieval results + user query as input
2. Returns structured, factual summary output
3. Is passed to query_reformulation which combines with structural chunks
4. Final answer generated from summary + structural chunks
"""

import logging
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)


class SummaryToolsProvider:
    """Provides 4 factual summary tools for LLM usage"""
    
    @staticmethod
    def structured_data_summary_tool(chunks: List[Dict], query: str) -> Dict[str, Any]:
        """
        Tool 1: STRUCTURED DATA SUMMARY
        Organize metric chunks into clear categories.
        Returns: Facts organized by topic/category
        
        Output fields for reformulation:
        - summary: Text summary of structured facts
        - categories: Dict of organized metrics
        - insights: List of key facts/metrics
        """
        logger.info("[SUMMARY:TOOL] Executing structured_data_summary tool")
        
        try:
            # Extract metric chunks
            metric_chunks = [c for c in chunks if c.get('chunk_type') == 'metric_centric']
            if not metric_chunks:
                metric_chunks = chunks
            
            logger.info(f"[SUMMARY:TOOL] Organizing {len(metric_chunks)} metrics into structured categories")
            
            # Organize by metric type
            categories = {}
            insights = []
            summary_parts = []
            
            for chunk in metric_chunks:
                metric_type = chunk.get('metric_type', 'other')
                metric_name = chunk.get('metric_name') or 'unknown'
                text = chunk.get('text', '')
                
                if metric_type not in categories:
                    categories[metric_type] = {
                        "count": 0,
                        "metrics": [],
                        "facts": []
                    }
                
                categories[metric_type]["count"] += 1
                categories[metric_type]["metrics"].append(metric_name)
                if len(categories[metric_type]["facts"]) < 2:
                    categories[metric_type]["facts"].append(text[:100])
                insights.append(f"{metric_name}: {text[:80]}")
            
            # Build summary text
            for category, data in categories.items():
                summary_parts.append(f"{category}: {data['count']} metrics ({', '.join(data['metrics'][:3])})")
            
            summary_text = " | ".join(summary_parts)
            
            logger.info(f"[SUMMARY:TOOL] ✓ Organized {len(metric_chunks)} metrics into {len(categories)} categories")
            
            return {
                "success": True,
                "tool_name": "structured_data_summary",
                "summary": summary_text,
                "categories": categories,
                "insights": insights[:10],  # Top 10 insights for reformulation
                "total_metrics": len(metric_chunks),
                "summary_tool_used": "structured_data_summary",
                "confidence_score": min(0.95, len(metric_chunks) / 100),
                "technique": "Structured Data Summary - Facts organized by category"
            }
        except Exception as e:
            logger.error(f"[SUMMARY:TOOL] Structured summary failed: {e}")
            return {"success": False, "error": str(e), "tool_name": "structured_data_summary"}
    
    @staticmethod
    def metric_condensing_tool(chunks: List[Dict], query: str) -> Dict[str, Any]:
        """
        Tool 2: METRIC CONDENSING
        Reduce each metric to ONLY essential information.
        
        Output fields for reformulation:
        - summary: Condensed summary of all essentials
        - metrics: Dict of one-line essentials per metric
        - insights: List of top condensed facts
        """
        logger.info("[SUMMARY:TOOL] Executing metric_condensing tool")
        
        try:
            # Extract metric chunks
            metric_chunks = [c for c in chunks if c.get('chunk_type') == 'metric_centric']
            if not metric_chunks:
                metric_chunks = chunks
            
            logger.info(f"[SUMMARY:TOOL] Condensing {len(metric_chunks)} metrics to essentials")
            
            # Extract just essential information per metric
            condensed = {}
            insights = []
            
            for chunk in metric_chunks:
                metric_name = chunk.get('metric_name') or 'unknown'
                text = chunk.get('text', '')
                
                # Keep only first sentence/key facts (no explanation)
                essential = text.split('.')[0].strip()
                if len(essential) > 150:
                    essential = essential[:150] + "..."
                
                condensed[metric_name] = essential
                insights.append(f"{metric_name}: {essential}")
            
            # Build condensed summary
            summary_parts = [f"{k}: {v}" for k, v in list(condensed.items())[:5]]
            summary_text = " | ".join(summary_parts)
            
            logger.info(f"[SUMMARY:TOOL] ✓ Condensed {len(condensed)} metrics")
            
            return {
                "success": True,
                "tool_name": "metric_condensing",
                "summary": summary_text,
                "metrics": condensed,
                "insights": insights[:10],  # Top 10 condensed metrics
                "total_metrics": len(condensed),
                "summary_tool_used": "metric_condensing",
                "confidence_score": 0.90,
                "technique": "Metric Condensing - Essentials only, no explanation"
            }
        except Exception as e:
            logger.error(f"[SUMMARY:TOOL] Metric condensing failed: {e}")
            return {"success": False, "error": str(e), "tool_name": "metric_condensing"}
    
    @staticmethod
    def constraint_listing_tool(chunks: List[Dict], query: str) -> Dict[str, Any]:
        """
        Tool 3: CONSTRAINT LISTING (Factual Only)
        List which constraints are binding vs slack.
        
        Output fields for reformulation:
        - summary: Summary of constraint status
        - binding_constraints: List of binding metrics
        - tight_constraints: List of tight constraints
        - slack_constraints: List of slack constraints
        - insights: Constraint status facts
        """
        logger.info("[SUMMARY:TOOL] Executing constraint_listing tool")
        
        try:
            metric_chunks = [c for c in chunks if c.get('chunk_type') == 'metric_centric']
            if not metric_chunks:
                metric_chunks = chunks
            
            logger.info(f"[SUMMARY:TOOL] Analyzing {len(metric_chunks)} metrics for constraints")
            
            # Categorize by binding status
            constraints = {
                "binding": [],
                "tight": [],
                "slack": []
            }
            
            for chunk in metric_chunks:
                metric_name = chunk.get('metric_name') or 'unknown'
                confidence = chunk.get('confidence', 0.5)
                
                if confidence >= 0.95:
                    constraints["binding"].append(metric_name)
                elif confidence >= 0.80:
                    constraints["tight"].append(metric_name)
                else:
                    constraints["slack"].append(metric_name)
            
            # Build summary
            summary_parts = [
                f"Binding: {len(constraints['binding'])} metrics",
                f"Tight: {len(constraints['tight'])} metrics",
                f"Slack: {len(constraints['slack'])} metrics"
            ]
            summary_text = " | ".join(summary_parts)
            
            # Build insights
            insights = []
            for metric in constraints['binding'][:5]:
                insights.append(f"[BINDING] {metric}")
            for metric in constraints['tight'][:5]:
                insights.append(f"[TIGHT] {metric}")
            
            logger.info(f"[SUMMARY:TOOL] ✓ Constraints: {len(constraints['binding'])} binding, "
                       f"{len(constraints['tight'])} tight, {len(constraints['slack'])} slack")
            
            return {
                "success": True,
                "tool_name": "constraint_listing",
                "summary": summary_text,
                "binding_constraints": constraints["binding"],
                "tight_constraints": constraints["tight"],
                "slack_constraints": constraints["slack"],
                "insights": insights,
                "summary_tool_used": "constraint_listing",
                "confidence_score": 0.88,
                "technique": "Constraint Listing - Factual status only"
            }
        except Exception as e:
            logger.error(f"[SUMMARY:TOOL] Constraint listing failed: {e}")
            return {"success": False, "error": str(e), "tool_name": "constraint_listing"}
    
    @staticmethod
    def feasibility_check_tool(chunks: List[Dict], query: str) -> Dict[str, Any]:
        """
        Tool 4: FEASIBILITY CHECK
        Simple YES/NO feasibility check with violations if any.
        
        Output fields for reformulation:
        - summary: YES/NO feasibility status
        - feasible: Feasibility answer
        - violations: List of violations found
        - insights: Violation details
        """
        logger.info("[SUMMARY:TOOL] Executing feasibility_check tool")
        
        try:
            metric_chunks = [c for c in chunks if c.get('chunk_type') == 'metric_centric']
            if not metric_chunks:
                metric_chunks = chunks
            
            logger.info(f"[SUMMARY:TOOL] Checking feasibility of {len(metric_chunks)} metrics")
            
            # Check for violations (factual only)
            violations = []
            is_feasible = True
            
            for chunk in metric_chunks:
                metric_name = chunk.get('metric_name') or 'unknown'
                confidence = chunk.get('confidence', 0.5)
                relevance = chunk.get('relevance', 1.0)
                
                # Violations are FACTUAL metric quality issues
                if confidence < 0.5:
                    violations.append(f"{metric_name}: low confidence ({confidence:.0%})")
                    is_feasible = False
                
                if relevance < 0.3:
                    violations.append(f"{metric_name}: low relevance ({relevance:.0%})")
                    is_feasible = False
            
            # Determine feasibility status
            if is_feasible and len(metric_chunks) > 0:
                feasibility_status = "YES"
            elif not is_feasible:
                feasibility_status = "NO"
            else:
                feasibility_status = "PARTIAL"
            
            summary_text = f"Feasibility: {feasibility_status}"
            if violations:
                summary_text += f" ({len(violations)} violations)"
            
            logger.info(f"[SUMMARY:TOOL] ✓ Feasibility: {feasibility_status}, violations: {len(violations)}")
            
            return {
                "success": True,
                "tool_name": "feasibility_check",
                "summary": summary_text,
                "feasible": feasibility_status,
                "total_constraints_checked": len(metric_chunks),
                "violations": violations,
                "insights": violations,  # For reformulation
                "summary_tool_used": "feasibility_check",
                "confidence_score": 0.92,
                "technique": "Feasibility Check - Status with violation list"
            }
        except Exception as e:
            logger.error(f"[SUMMARY:TOOL] Feasibility check failed: {e}")
            return {"success": False, "error": str(e), "tool_name": "feasibility_check"}
    
    @staticmethod
    def get_all_tools() -> Dict[str, callable]:
        """Get all 4 summary tools as callable functions"""
        return {
            "structured_data_summary": SummaryToolsProvider.structured_data_summary_tool,
            "metric_condensing": SummaryToolsProvider.metric_condensing_tool,
            "constraint_listing": SummaryToolsProvider.constraint_listing_tool,
            "feasibility_check": SummaryToolsProvider.feasibility_check_tool,
        }
    
    @staticmethod
    def get_tool_descriptions() -> Dict[str, str]:
        """Get descriptions for LLM tool selection"""
        return {
            "structured_data_summary": (
                "Organize facts by metric type/category. Shows what metrics measure without interpretation. "
                "Use when user wants organized, categorized facts about different metrics."
            ),
            "metric_condensing": (
                "Reduce each metric to ONE LINE of essentials - just numbers and key facts, no explanation. "
                "Use when user wants quick, condensed reference of metrics."
            ),
            "constraint_listing": (
                "List metrics by constraint status: binding (must hold), tight (limited slack), slack (flexible). "
                "Factual status only. Use when user wants to understand constraint tightness."
            ),
            "feasibility_check": (
                "Simple YES/NO feasibility with list of violations. Factual check only, no interpretation. "
                "Use when user wants quick feasibility assessment of metrics/constraints."
            ),
        }

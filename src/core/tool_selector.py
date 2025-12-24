"""
Tool Selector - Intelligently selects tools based on query and data types
Implements: SELECT_TOOLS node logic
"""

import logging
import re
from typing import List, Dict, Any, Set
from .workflow_state import DataType

logger = logging.getLogger(__name__)


class ToolSelector:
    """
    Intelligently selects which tools to use based on:
    1. Query intent (calculate, compare, trend analysis, etc.)
    2. Detected data types (tables, numeric data, text)
    3. Available tools in the system
    
    Returns prioritized list of tool names to execute.
    """
    
    def __init__(self, available_tools: Dict[str, Any] = None):
        """
        Initialize tool selector.
        
        Args:
            available_tools: Dict mapping tool names to tool definitions
                {"calculator": {...}, "trend_analyzer": {...}, ...}
        """
        self.available_tools = available_tools or self._get_default_tools()
        self.logger = logging.getLogger(__name__)
        
        # Define query intent patterns
        self.intent_patterns = {
            "calculate": [
                r'\b(calculate|compute|sum|total|add|multiply|divide)\b',
                r'\b(what is|what\'s|how much|how many)\b.*\b(total|sum|amount)\b'
            ],
            "compare": [
                r'\b(compare|difference|versus|vs|between)\b',
                r'\b(which is (more|larger|smaller)|greater than|less than)\b'
            ],
            "trend": [
                r'\b(trend|growth|decline|increase|decrease|historical)\b',
                r'\b(over time|year over year|quarter over quarter)\b'
            ],
            "analyze": [
                r'\b(analyze|analysis|insights|pattern)\b',
                r'\b(breakdown|segment|categorize|classify)\b'
            ],
            "search": [
                r'\b(find|search|look for|get|retrieve|show)\b',
                r'\b(list all|show all|what are)\b'
            ],
            "forecast": [
                r'\b(forecast|predict|projection|estimate|future)\b',
                r'\b(will|expected|anticipated)\b'
            ]
        }
        
        # Map intents to tool names
        self.intent_to_tools = {
            "calculate": ["calculator", "financial_calculator"],
            "compare": ["data_comparator", "calculator"],
            "trend": ["trend_analyzer", "time_series_analyzer"],
            "analyze": ["data_analyzer", "pattern_detector"],
            "search": ["data_retriever", "search_engine"],
            "forecast": ["forecaster", "predictor"]
        }
    
    async def select_tools(
        self,
        query: str,
        detected_data_types: List[DataType],
        available_tool_names: List[str],
        conversation_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Select tools based on query intent and data types.
        
        Args:
            query: User's query/question
            detected_data_types: Data types found in search results (TABLE, NUMERIC, TEXT, MIXED)
            available_tool_names: List of tools available in the system
            conversation_history: Previous messages for context
            
        Returns:
            Dict with:
            {
                "selected_tools": ["tool1", "tool2"],
                "primary_tool": "tool1",
                "rationale": str,
                "confidence": float (0-1)
            }
        """
        if not query:
            return {
                "selected_tools": [],
                "primary_tool": None,
                "rationale": "No query provided",
                "confidence": 0.0
            }
        
        # Step 1: Detect query intent
        detected_intents = self._detect_intent(query)
        
        # Step 2: Match intents to available tools
        candidate_tools = self._get_tools_for_intents(
            detected_intents,
            available_tool_names
        )
        
        # Step 3: Score tools based on data types
        scored_tools = self._score_tools_by_data_type(
            candidate_tools,
            detected_data_types
        )
        
        # Step 4: Reorder by score and availability
        selected_tools = self._rank_and_filter(
            scored_tools,
            available_tool_names,
            max_tools=3  # Select max 3 tools to avoid redundancy
        )
        
        if not selected_tools:
            return {
                "selected_tools": [],
                "primary_tool": None,
                "rationale": "No suitable tools found for query",
                "confidence": 0.3
            }
        
        primary_tool = selected_tools[0]["name"]
        confidence = selected_tools[0]["score"]
        
        rationale = self._build_rationale(
            detected_intents,
            detected_data_types,
            selected_tools
        )
        
        self.logger.info(f"Selected tools: {[t['name'] for t in selected_tools]}, confidence: {confidence:.2f}")
        
        return {
            "selected_tools": [t["name"] for t in selected_tools],
            "primary_tool": primary_tool,
            "rationale": rationale,
            "confidence": confidence
        }
    
    def _detect_intent(self, query: str) -> Dict[str, float]:
        """
        Detect query intent using pattern matching.
        
        Args:
            query: User's query
            
        Returns:
            Dict mapping intent names to confidence scores (0-1)
        """
        query_lower = query.lower()
        intent_scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            matches = 0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    matches += 1
            
            # Score: each pattern match adds 0.5, cap at 1.0
            intent_scores[intent] = min(0.5 * matches, 1.0)
        
        # Boost confidence if multiple indicators for same intent
        for intent, score in intent_scores.items():
            if score > 0:
                intent_scores[intent] = min(score + 0.1, 1.0)
        
        # Filter out low-confidence intents
        return {k: v for k, v in intent_scores.items() if v > 0.3}
    
    def _get_tools_for_intents(
        self,
        intents: Dict[str, float],
        available_tools: List[str]
    ) -> Dict[str, float]:
        """
        Get tools recommended for detected intents.
        
        Args:
            intents: Dict of {intent: confidence}
            available_tools: List of available tool names
            
        Returns:
            Dict mapping tool names to confidence scores
        """
        tool_scores = {}
        
        for intent, confidence in intents.items():
            recommended_tools = self.intent_to_tools.get(intent, [])
            for tool_name in recommended_tools:
                # Only include available tools
                if tool_name in available_tools:
                    # Score combines intent confidence with availability
                    current_score = tool_scores.get(tool_name, 0)
                    tool_scores[tool_name] = max(current_score, confidence)
        
        return tool_scores
    
    def _score_tools_by_data_type(
        self,
        candidate_tools: Dict[str, float],
        detected_data_types: List[DataType]
    ) -> Dict[str, float]:
        """
        Adjust tool scores based on data types found in results.
        
        Args:
            candidate_tools: Dict of {tool: confidence}
            detected_data_types: List of DataType enums found
            
        Returns:
            Adjusted tool scores
        """
        # Define tool preferences for each data type
        type_preferences = {
            DataType.TABLE: {
                "data_comparator": 0.2,
                "trend_analyzer": 0.1,
                "pattern_detector": 0.1,
            },
            DataType.NUMERIC: {
                "calculator": 0.3,
                "financial_calculator": 0.25,
                "trend_analyzer": 0.15,
                "forecaster": 0.1,
            },
            DataType.TEXT: {
                "data_analyzer": 0.2,
                "pattern_detector": 0.1,
                "search_engine": 0.05,
            },
            DataType.MIXED: {
                "data_analyzer": 0.15,
                "calculator": 0.1,
                "trend_analyzer": 0.1,
            }
        }
        
        adjusted_scores = dict(candidate_tools)
        
        for data_type in detected_data_types:
            preferences = type_preferences.get(data_type, {})
            for tool_name, bonus in preferences.items():
                if tool_name in adjusted_scores:
                    adjusted_scores[tool_name] = min(
                        adjusted_scores[tool_name] + bonus,
                        1.0
                    )
        
        return adjusted_scores
    
    def _rank_and_filter(
        self,
        scored_tools: Dict[str, float],
        available_tools: List[str],
        max_tools: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Rank tools by score and filter to top N.
        
        Args:
            scored_tools: Dict of {tool: score}
            available_tools: List of available tools
            max_tools: Maximum number of tools to return
            
        Returns:
            List of dicts: [{"name": str, "score": float}, ...]
        """
        # Filter to available tools only
        valid_tools = [
            {"name": name, "score": score}
            for name, score in scored_tools.items()
            if name in available_tools
        ]
        
        # Sort by score (descending)
        valid_tools.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top N
        return valid_tools[:max_tools]
    
    def _build_rationale(
        self,
        intents: Dict[str, float],
        data_types: List[DataType],
        selected_tools: List[Dict[str, Any]]
    ) -> str:
        """
        Build human-readable rationale for tool selection.
        
        Args:
            intents: Detected query intents
            data_types: Detected data types
            selected_tools: Selected tool info
            
        Returns:
            Explanation string
        """
        parts = []
        
        if intents:
            top_intent = max(intents.items(), key=lambda x: x[1])
            parts.append(f"Query intent: {top_intent[0].title()}")
        
        if data_types:
            type_names = ", ".join([dt.value for dt in data_types])
            parts.append(f"Data types: {type_names}")
        
        tool_names = ", ".join([t["name"] for t in selected_tools])
        parts.append(f"Selected tools: {tool_names}")
        
        return " | ".join(parts)
    
    def _get_default_tools(self) -> Dict[str, Any]:
        """
        Get default set of available tools.
        
        Returns:
            Dict of tool definitions
        """
        return {
            "calculator": {"type": "calculator", "category": "compute"},
            "financial_calculator": {"type": "calculator", "category": "finance"},
            "data_comparator": {"type": "comparison", "category": "analyze"},
            "trend_analyzer": {"type": "time_series", "category": "analyze"},
            "time_series_analyzer": {"type": "time_series", "category": "analyze"},
            "data_analyzer": {"type": "analysis", "category": "analyze"},
            "pattern_detector": {"type": "pattern", "category": "analyze"},
            "data_retriever": {"type": "retrieval", "category": "search"},
            "search_engine": {"type": "search", "category": "search"},
            "forecaster": {"type": "forecasting", "category": "predict"},
            "predictor": {"type": "prediction", "category": "predict"},
        }

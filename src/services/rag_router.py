"""
Agentic RAG Router - LLM-based routing to decide when to use RAG
Analyzes queries to determine if document context is needed
"""

import json
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class RAGRouter:
    """Routes queries to determine if RAG should be used"""
    
    def __init__(self, llm):
        self.llm = llm
    
    async def should_use_rag(
        self,
        query: str,
        conversation_history: List[Dict] = None
    ) -> Tuple[bool, Dict]:
        """
        Determine if query needs RAG documents
        
        Args:
            query: User query
            conversation_history: Recent conversation for context
            
        Returns:
            Tuple of (should_use_rag: bool, reasoning: dict)
        """
        try:
            # Quick heuristic check for common financial patterns
            financial_keywords = ["giá", "price", "lịch sử", "history", "sma", "rsi", "cổ đông", "shareholders", 
                                "thông tin công ty", "company info", "cổ phiếu", "stock", "chứng khoán", "securities"]
            is_financial_query = any(keyword.lower() in query.lower() for keyword in financial_keywords)
            
            # If it's clearly a financial query, don't use RAG (use agent tools instead)
            if is_financial_query:
                return False, {
                    "use_rag": False,
                    "query_type": "financial_data",
                    "confidence": 0.95,
                    "reasoning": "Financial market data query - use agent tools"
                }
            
            # For other queries, ask LLM
            # Build context for the router decision
            history_context = ""
            if conversation_history and len(conversation_history) > 0:
                recent_msgs = conversation_history[-4:]  # Last 4 messages
                history_context = "\n".join([
                    f"- {msg.get('role', 'user')}: {msg.get('content', '')[:100]}"
                    for msg in recent_msgs
                ])
            
            # Prompt for router decision
            router_prompt = f"""Analyze this query to determine if it needs document context from uploaded files.

User Query: {query}

Recent Conversation Context:
{history_context if history_context else "No recent history"}

Respond with ONLY valid JSON (no markdown, no extra text):
{{
  "use_rag": true/false,
  "query_type": "document_lookup|financial_data|conversational|technical_analysis|other",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation"
}}

Guidelines:
- use_rag=true if query asks about uploaded documents, specific reports, or document-based information
- use_rag=false if query is about market data, general financial knowledge, or conversational
- Consider conversation context when deciding"""
            
            # Get decision from LLM (brief, structured response)
            try:
                # Use sync query method wrapped in async
                import asyncio
                
                # Create a simplified decision prompt
                response_text = await asyncio.to_thread(
                    self._get_router_decision,
                    router_prompt
                )
                
                # Parse response
                decision = self._parse_router_response(response_text)
                return decision["use_rag"], decision
                
            except Exception as e:
                logger.warning(f"Router LLM call failed: {e}, defaulting to RAG enabled")
                # Default to RAG enabled for safety
                return True, {
                    "use_rag": True,
                    "query_type": "unknown",
                    "confidence": 0.5,
                    "reasoning": "Router unavailable, defaulting to RAG enabled"
                }
        
        except Exception as e:
            logger.error(f"Error in RAG routing decision: {e}")
            return True, {
                "use_rag": True,
                "query_type": "error",
                "confidence": 0.0,
                "reasoning": f"Router error: {str(e)}"
            }
    
    def _get_router_decision(self, prompt: str) -> str:
        """Get router decision directly from LLM (sync method)"""
        try:
            # Ensure we have an LLM with invoke method
            if not hasattr(self.llm, 'invoke'):
                logger.error(f"Router received non-LLM object: {type(self.llm).__name__}")
                raise AttributeError(f"Expected LLM object with 'invoke' method, got {type(self.llm).__name__}")
            
            # Call LLM directly without agent system prompt
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"Router LLM call failed: {e}")
            raise
    
    def _parse_router_response(self, response: str) -> Dict:
        """Parse JSON response from router"""
        try:
            response = response.strip()
            
            # Remove markdown code blocks if present
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
                response = response.strip()
            
            # Try to extract JSON from text that might contain explanatory text
            # Look for JSON starting with { and ending with }
            import re
            json_match = re.search(r'\{[^{}]*"use_rag"[^{}]*\}', response, re.DOTALL)
            if json_match:
                response = json_match.group(0)
                logger.info(f"Extracted JSON from wrapped response")
            
            # Parse JSON
            decision = json.loads(response)
            
            # Validate required fields
            if "use_rag" not in decision:
                logger.warning(f"Missing 'use_rag' in router response: {decision}")
                decision["use_rag"] = True
            
            if "query_type" not in decision:
                decision["query_type"] = "unknown"
            
            if "confidence" not in decision:
                decision["confidence"] = 0.5
            
            if "reasoning" not in decision:
                decision["reasoning"] = ""
            
            return decision
        
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse router JSON response: {response}")
            return {
                "use_rag": True,
                "query_type": "parse_error",
                "confidence": 0.0,
                "reasoning": "Failed to parse router response"
            }
        except Exception as e:
            logger.error(f"Error parsing router response: {e}")
            return {
                "use_rag": True,
                "query_type": "error",
                "confidence": 0.0,
                "reasoning": f"Parsing error: {str(e)}"
            }


# Global router instance
_router: Optional[RAGRouter] = None


def get_rag_router(llm=None) -> RAGRouter:
    """Get or create the RAG router instance"""
    global _router
    
    # If llm is provided, always use it and update the cached router
    if llm is not None:
        _router = RAGRouter(llm)
        return _router
    
    # Otherwise use or create cached router
    if _router is None:
        from ..llm import LLMFactory
        llm = LLMFactory.get_llm()
        _router = RAGRouter(llm)
    
    return _router


def reset_rag_router():
    """Reset the router (useful for testing)"""
    global _router
    _router = None

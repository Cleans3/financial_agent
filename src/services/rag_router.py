"""
Agentic RAG Router - LLM-based routing to decide when to use RAG
Analyzes queries to determine if document context is needed
"""

import json
import logging
from typing import Dict, List, Tuple, Optional
from ..agent import FinancialAgent

logger = logging.getLogger(__name__)


class RAGRouter:
    """Routes queries to determine if RAG should be used"""
    
    def __init__(self, agent: FinancialAgent):
        self.agent = agent
    
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
        """Get router decision from agent (sync method)"""
        try:
            response = self.agent.query(prompt)
            return response if isinstance(response, str) else str(response)
        except Exception as e:
            logger.error(f"Router query failed: {e}")
            raise
    
    def _parse_router_response(self, response: str) -> Dict:
        """Parse JSON response from router"""
        try:
            # Extract JSON from response (handle markdown code blocks)
            response = response.strip()
            
            # Remove markdown code blocks if present
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            
            response = response.strip()
            
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


def get_rag_router(agent: FinancialAgent = None) -> RAGRouter:
    """Get or create the RAG router instance"""
    global _router
    if _router is None and agent is not None:
        _router = RAGRouter(agent)
    return _router


def reset_rag_router():
    """Reset the router (useful for testing)"""
    global _router
    _router = None

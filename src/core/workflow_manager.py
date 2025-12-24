"""
WorkflowManager - Orchestrates LangGraph workflow for financial agent
Initializes graph at pipeline entry point and manages step progression
"""

import logging
from typing import Any, Dict, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from ..agent.state import AgentState
from .step_emitter import get_step_emitter

logger = logging.getLogger(__name__)


class WorkflowManager:
    """
    Manages LangGraph workflow for financial agent.
    
    10-Node Pipeline:
    1. CLASSIFY - Analyze query intent
    2. REWRITE - Rewrite ambiguous queries
    3. EXTRACT - Extract metadata (tickers, dates, keywords)
    4. INGEST - Check if document needs ingestion
    5. RETRIEVE_PERSONAL - Search personal RAG collection
    6. RETRIEVE_GLOBAL - Search global readonly collection
    7. FILTER - Filter results by relevance threshold
    8. SELECT - Decide: use RAG, use tools, or combine
    9. GENERATE - Call LLM or tool execution
    10. COMPLETE - Format final response
    """
    
    def __init__(self, agent_executor):
        """
        Initialize WorkflowManager.
        
        Args:
            agent_executor: FinancialAgent instance with tools and LLM
        """
        self.agent_executor = agent_executor
        self.step_emitter = get_step_emitter()
        self.graph = self._build_graph()
        logger.info("WorkflowManager initialized with 10-node pipeline")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow graph"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("classify", self._node_classify)
        workflow.add_node("rewrite", self._node_rewrite)
        workflow.add_node("extract", self._node_extract)
        workflow.add_node("ingest", self._node_ingest)
        workflow.add_node("retrieve_personal", self._node_retrieve_personal)
        workflow.add_node("retrieve_global", self._node_retrieve_global)
        workflow.add_node("filter", self._node_filter)
        workflow.add_node("select", self._node_select)
        workflow.add_node("generate", self._node_generate)
        workflow.add_node("complete", self._node_complete)
        
        # Add edges
        workflow.set_entry_point("classify")
        workflow.add_edge("classify", "rewrite")
        workflow.add_edge("rewrite", "extract")
        workflow.add_edge("extract", "ingest")
        workflow.add_edge("ingest", "retrieve_personal")
        workflow.add_edge("retrieve_personal", "retrieve_global")
        workflow.add_edge("retrieve_global", "filter")
        workflow.add_edge("filter", "select")
        workflow.add_edge("select", "generate")
        workflow.add_edge("generate", "complete")
        workflow.add_edge("complete", END)
        
        return workflow.compile()
    
    async def invoke(self, query: str, user_id: str = None, session_id: str = None) -> Dict[str, Any]:
        """
        Invoke the workflow pipeline.
        
        Args:
            query: User query
            user_id: Optional user identifier
            session_id: Optional session identifier
            
        Returns:
            Final agent state with response
        """
        initial_state = AgentState(
            messages=[HumanMessage(content=query)],
            agent_steps=[],
            current_step=0,
            step_callback=None
        )
        
        result = await self.graph.ainvoke(initial_state)
        return result
    
    # ========== NODE IMPLEMENTATIONS ==========
    
    async def _node_classify(self, state: AgentState) -> AgentState:
        """Step 1: Classify query intent (financial/general/ambiguous)"""
        await self.step_emitter.step_started(1, "Analyzing query intent...")
        
        try:
            query = state["messages"][-1].content if state["messages"] else ""
            
            # Classify logic
            is_financial = any(word in query.lower() for word in 
                             ["giá", "cổ phiếu", "lãi suất", "lợi nhuận", "tài chính", 
                              "vn-index", "hose", "hnx", "bitcoin", "etf", "định giá"])
            
            classification = "financial" if is_financial else "general"
            logger.info(f"[NODE] CLASSIFY: '{classification}' for query")
            
            await self.step_emitter.step_completed(1, f"Classification: {classification}")
            return state
            
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            await self.step_emitter.step_error(1, str(e))
            raise
    
    async def _node_rewrite(self, state: AgentState) -> AgentState:
        """Step 2: Rewrite ambiguous queries"""
        await self.step_emitter.step_started(2, "Checking if query needs rewriting...")
        
        try:
            query = state["messages"][-1].content if state["messages"] else ""
            
            # Check if rewrite needed
            from ..agent.financial_agent import FinancialAgent
            agent = FinancialAgent()
            
            if agent._is_query_clear(query):
                logger.info(f"[NODE] REWRITE: Query is clear, skipping rewrite")
                await self.step_emitter.step_completed(2, "Query is clear, no rewrite needed")
                return state
            
            # Rewrite if needed
            rewritten = await agent._rewrite_query_if_needed(state)
            if rewritten:
                logger.info(f"[NODE] REWRITE: Query rewritten")
                await self.step_emitter.step_completed(2, "Query rewritten for clarity")
                return {"messages": state["messages"] + [rewritten]}
            
            await self.step_emitter.step_completed(2, "Query is clear")
            return state
            
        except Exception as e:
            logger.error(f"Rewrite failed: {e}")
            await self.step_emitter.step_error(2, str(e))
            return state  # Continue on error
    
    async def _node_extract(self, state: AgentState) -> AgentState:
        """Step 3: Extract metadata (tickers, dates, keywords)"""
        await self.step_emitter.step_started(3, "Extracting query metadata...")
        
        try:
            query = state["messages"][-1].content if state["messages"] else ""
            
            # Extract tickers, dates, etc
            import re
            tickers = re.findall(r'\b[A-Z]{2,4}\b', query.upper())
            
            logger.info(f"[NODE] EXTRACT: Found {len(set(tickers))} tickers: {set(tickers)}")
            
            state["_extracted_metadata"] = {
                "tickers": list(set(tickers)),
                "original_query": query
            }
            
            await self.step_emitter.step_completed(3, f"Extracted {len(set(tickers))} tickers")
            return state
            
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            await self.step_emitter.step_error(3, str(e))
            return state
    
    async def _node_ingest(self, state: AgentState) -> AgentState:
        """Step 4: Check if document needs ingestion"""
        await self.step_emitter.step_started(4, "Checking for document upload...")
        
        try:
            # Check for uploaded files in messages
            uploaded_files = state.get("_uploaded_files", [])
            
            if uploaded_files:
                logger.info(f"[NODE] INGEST: {len(uploaded_files)} files to ingest")
                # File ingestion would happen here
                await self.step_emitter.step_completed(4, f"Found {len(uploaded_files)} file(s) to ingest")
            else:
                logger.info(f"[NODE] INGEST: No files to ingest")
                await self.step_emitter.step_completed(4, "No files to ingest")
            
            return state
            
        except Exception as e:
            logger.error(f"Ingestion check failed: {e}")
            await self.step_emitter.step_error(4, str(e))
            return state
    
    async def _node_retrieve_personal(self, state: AgentState) -> AgentState:
        """Step 5: Search personal RAG collection"""
        await self.step_emitter.step_started(5, "Searching personal documents...")
        
        try:
            query = state["messages"][-1].content if state["messages"] else ""
            user_id = state.get("user_id", "default")
            
            # RAG search would happen here
            from ..services.multi_collection_rag_service import MultiCollectionRAGService
            rag = MultiCollectionRAGService()
            
            results = await rag.search_async(query, user_id)
            
            logger.info(f"[NODE] RETRIEVE_PERSONAL: Found {len(results)} personal results")
            state["_personal_rag_results"] = results
            
            await self.step_emitter.step_completed(5, f"Found {len(results)} personal documents")
            return state
            
        except Exception as e:
            logger.error(f"Personal RAG search failed: {e}")
            await self.step_emitter.step_error(5, str(e))
            state["_personal_rag_results"] = []
            return state
    
    async def _node_retrieve_global(self, state: AgentState) -> AgentState:
        """Step 6: Search global readonly collection"""
        await self.step_emitter.step_started(6, "Searching global documents...")
        
        try:
            query = state["messages"][-1].content if state["messages"] else ""
            
            # Global readonly search
            from ..services.multi_collection_rag_service import MultiCollectionRAGService
            rag = MultiCollectionRAGService()
            
            results = await rag.search_global_readonly_async(query)
            
            logger.info(f"[NODE] RETRIEVE_GLOBAL: Found {len(results)} global results")
            state["_global_rag_results"] = results
            
            await self.step_emitter.step_completed(6, f"Found {len(results)} global documents")
            return state
            
        except Exception as e:
            logger.error(f"Global RAG search failed: {e}")
            await self.step_emitter.step_error(6, str(e))
            state["_global_rag_results"] = []
            return state
    
    async def _node_filter(self, state: AgentState) -> AgentState:
        """Step 7: Filter results by relevance threshold"""
        await self.step_emitter.step_started(7, "Filtering by relevance...")
        
        try:
            personal = state.get("_personal_rag_results", [])
            global_results = state.get("_global_rag_results", [])
            
            # Already filtered in RAG service, but log here
            total = len(personal) + len(global_results)
            logger.info(f"[NODE] FILTER: {total} results after filtering (personal={len(personal)}, global={len(global_results)})")
            
            state["_filtered_results"] = personal + global_results
            
            await self.step_emitter.step_completed(7, f"Filtered to {total} relevant results")
            return state
            
        except Exception as e:
            logger.error(f"Filtering failed: {e}")
            await self.step_emitter.step_error(7, str(e))
            return state
    
    async def _node_select(self, state: AgentState) -> AgentState:
        """Step 8: Decide strategy (RAG, tools, or combine)"""
        await self.step_emitter.step_started(8, "Selecting retrieval strategy...")
        
        try:
            rag_results = state.get("_filtered_results", [])
            query = state["messages"][-1].content if state["messages"] else ""
            
            # Decision logic
            has_high_relevance = len(rag_results) > 0 and rag_results[0].get("score", 0) > 0.5
            is_financial = any(word in query.lower() for word in 
                             ["giá", "cổ phiếu", "lãi suất", "lợi nhuận", "vn-index"])
            
            if has_high_relevance:
                strategy = "rag"
                reason = f"RAG found {len(rag_results)} relevant documents"
            elif is_financial:
                strategy = "tools"
                reason = "Financial query - using VnStock tools"
            else:
                strategy = "llm"
                reason = "General query - using LLM only"
            
            logger.info(f"[SELECT] Strategy: {strategy} ({reason})")
            state["_strategy"] = strategy
            
            await self.step_emitter.step_completed(8, f"Strategy: {strategy}")
            return state
            
        except Exception as e:
            logger.error(f"Strategy selection failed: {e}")
            await self.step_emitter.step_error(8, str(e))
            state["_strategy"] = "llm"
            return state
    
    async def _node_generate(self, state: AgentState) -> AgentState:
        """Step 9: Execute selected strategy (RAG/tools/LLM)"""
        await self.step_emitter.step_started(9, "Generating response...")
        
        try:
            # This delegates to the existing agent logic
            # In real implementation, call agent_executor._agent_node()
            logger.info(f"[NODE] GENERATE: Using strategy '{state.get('_strategy', 'llm')}'")
            await self.step_emitter.step_completed(9, "Response generated")
            return state
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            await self.step_emitter.step_error(9, str(e))
            raise
    
    async def _node_complete(self, state: AgentState) -> AgentState:
        """Step 10: Format and return final response"""
        await self.step_emitter.step_started(10, "Finalizing response...")
        
        try:
            logger.info(f"[NODE] COMPLETE: Pipeline finished")
            await self.step_emitter.step_completed(10, "Pipeline complete")
            return state
            
        except Exception as e:
            logger.error(f"Completion failed: {e}")
            await self.step_emitter.step_error(10, str(e))
            raise


def get_workflow_manager(agent_executor) -> WorkflowManager:
    """Factory function to get WorkflowManager instance"""
    return WorkflowManager(agent_executor)

"""
Agent executor with LangGraph workflow control.
Handles all node invocations and routing decisions.
"""

from typing import Any, Dict, List, Optional
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage
import logging

logger = logging.getLogger(__name__)


class AgentExecutor:
    """Orchestrates LangGraph workflow for financial agent."""

    def __init__(self, llm, tools_config, summarization_config):
        self.llm = llm
        self.tools_config = tools_config
        self.summarization_config = summarization_config
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build LangGraph workflow with 10 nodes."""
        graph = StateGraph(WorkflowState)
        
        # Add all 10 nodes
        graph.add_node("classify", self._node_classify)
        graph.add_node("chitchat_handler", self._node_chitchat_handler)
        graph.add_node("rewrite_prompt", self._node_rewrite_prompt)
        graph.add_node("extract_data", self._node_extract_data)
        graph.add_node("ingest_file", self._node_ingest_file)
        graph.add_node("retrieve_personal", self._node_retrieve_personal)
        graph.add_node("retrieve_global", self._node_retrieve_global)
        graph.add_node("filter_search", self._node_filter_search)
        graph.add_node("select_tools", self._node_select_tools)
        graph.add_node("generate_answer", self._node_generate_answer)
        
        # Edges
        graph.add_edge(START, "classify")
        
        # From classify
        graph.add_conditional_edges(
            "classify",
            self._route_from_classify,
            {
                "chitchat": "chitchat_handler",
                "extract": "extract_data",
                "rewrite": "rewrite_prompt",
            }
        )
        graph.add_edge("chitchat_handler", END)
        
        # From rewrite
        graph.add_conditional_edges(
            "rewrite_prompt",
            self._route_from_rewrite,
            {
                "extract": "extract_data",
                "retrieve": "retrieve_personal",
                "generate": "generate_answer",
            }
        )
        
        # From extract
        graph.add_conditional_edges(
            "extract_data",
            self._route_from_extract,
            {
                "ingest": "ingest_file",
                "retrieve": "retrieve_personal",
                "generate": "generate_answer",
            }
        )
        graph.add_edge("ingest_file", "retrieve_personal")
        
        # Retrieval chain
        graph.add_conditional_edges(
            "retrieve_personal",
            self._route_from_retrieve_personal,
            {
                "filter": "filter_search",
                "global": "retrieve_global",
            }
        )
        
        graph.add_conditional_edges(
            "retrieve_global",
            self._route_from_retrieve_global,
            {
                "filter": "filter_search",
                "select": "select_tools",
            }
        )
        
        graph.add_edge("filter_search", "select_tools")
        graph.add_edge("select_tools", "generate_answer")
        graph.add_edge("generate_answer", END)
        
        return graph.compile()

    # Node implementations
    def _node_classify(self, state: WorkflowState) -> WorkflowState:
        """CLASSIFY node - determine prompt type and file handling."""
        logger.info("==== NODE: CLASSIFY ====")
        
        has_prompt = bool(state.get("user_prompt"))
        has_files = bool(state.get("uploaded_files"))
        
        if not has_prompt and has_files:
            prompt_type = "file_only"
        elif has_prompt and has_files:
            prompt_type = "prompt_and_file"
        elif has_prompt:
            # Detect chitchat vs request vs instruction
            prompt_type = self._detect_prompt_type(state["user_prompt"])
        else:
            prompt_type = "invalid"
        
        state["prompt_type"] = prompt_type
        state["needs_file_processing"] = has_files
        
        logger.info(f"  prompt_type={prompt_type}, needs_file_processing={has_files}")
        return state

    def _node_rewrite_prompt(self, state: WorkflowState) -> WorkflowState:
        """REWRITE_PROMPT node - disambiguate query with context."""
        logger.info("==== NODE: REWRITE_PROMPT ====")
        
        # Guard: max 1 rewrite per query
        if state.get("rewrite_count", 0) >= 1:
            logger.info("  ⚠️  Rewrite limit reached, skipping")
            return state
        
        original = state["user_prompt"]
        logger.info(f"  Original: {original[:80]}...")
        
        # Skip rewrite if filename already mentioned
        if state.get("uploaded_files"):
            first_file = state["uploaded_files"][0].get("filename", "")
            if first_file in original:
                logger.info(f"  Filename found in query, skipping rewrite")
                state["rewritten_prompt"] = original
                state["rewrite_count"] = 1
                return state
        
        # Rewrite with limited history (only last 2 exchanges)
        rewritten = self._rewrite_with_context(
            original, 
            state.get("conversation_history", [])[-4:]  # Last 2 exchanges
        )
        
        state["rewritten_prompt"] = rewritten
        state["rewrite_count"] = state.get("rewrite_count", 0) + 1
        
        logger.info(f"  Rewritten: {rewritten[:80]}...")
        return state

    def _node_extract_data(self, state: WorkflowState) -> WorkflowState:
        """EXTRACT_DATA node - process uploaded files."""
        logger.info("==== NODE: EXTRACT_DATA ====")
        
        files = state.get("uploaded_files", [])
        if not files:
            logger.info("  No files to extract")
            return state
        
        logger.info(f"  Processing {len(files)} file(s)")
        extracted_data = self._process_files(files)
        
        state["extracted_file_data"] = extracted_data
        logger.info(f"  ✓ Extracted: {len(extracted_data)} items")
        
        return state

    def _node_ingest_file(self, state: WorkflowState) -> WorkflowState:
        """INGEST_FILE node - embed and store to RAG."""
        logger.info("==== NODE: INGEST_FILE ====")
        
        extracted = state.get("extracted_file_data", {})
        if not extracted:
            return state
        
        # Determine embedding method
        size_kb = extracted.get("size_bytes", 0) / 1024
        if size_kb < 5:
            method = "SINGLE_DENSE"
        elif size_kb < 50:
            method = "MULTIDIMENSIONAL"
        else:
            method = "HIERARCHICAL"
        
        logger.info(f"  File size: {size_kb:.1f}KB → {method}")
        logger.info(f"  Model: general (sentence-transformers/all-MiniLM-L6-v2, dim=384)")
        
        file_ids = self._ingest_to_rag(extracted, method)
        
        state["embedding_method"] = method
        state["ingested_file_ids"] = file_ids
        
        logger.info(f"  ✓ Ingested {len(file_ids)} document IDs")
        return state

    def _node_retrieve_personal(self, state: WorkflowState) -> WorkflowState:
        """RETRIEVE_PERSONAL node - search personal collection."""
        logger.info("==== NODE: RETRIEVE_PERSONAL ====")
        
        query = state.get("rewritten_prompt") or state.get("user_prompt")
        filenames = [f.get("filename", "") for f in state.get("uploaded_files", [])]
        
        logger.info(f"  Query: {query[:60]}...")
        logger.info(f"  Files: {filenames}")
        
        semantic_results = self._semantic_search(query)
        keyword_results = self._keyword_search(query)
        
        logger.info(f"  Semantic: {len(semantic_results)} results (scores: {[r.get('score') for r in semantic_results[:3]]})")
        logger.info(f"  Keyword: {len(keyword_results)} results")
        
        state["personal_semantic_results"] = semantic_results
        state["personal_keyword_results"] = keyword_results
        
        if not semantic_results and not keyword_results:
            logger.info("  → No personal results, will try global")
        
        return state

    def _node_retrieve_global(self, state: WorkflowState) -> WorkflowState:
        """RETRIEVE_GLOBAL node - fallback search."""
        logger.info("==== NODE: RETRIEVE_GLOBAL ====")
        
        query = state.get("rewritten_prompt") or state.get("user_prompt")
        
        semantic_results = self._global_semantic_search(query)
        keyword_results = self._global_keyword_search(query)
        
        logger.info(f"  Semantic: {len(semantic_results)} results")
        logger.info(f"  Keyword: {len(keyword_results)} results")
        
        state["global_semantic_results"] = semantic_results
        state["global_keyword_results"] = keyword_results
        
        return state

    def _node_filter_search(self, state: WorkflowState) -> WorkflowState:
        """FILTER_SEARCH node - rank and deduplicate results."""
        logger.info("==== NODE: FILTER_SEARCH ====")
        
        personal_sem = state.get("personal_semantic_results", [])
        personal_kw = state.get("personal_keyword_results", [])
        global_sem = state.get("global_semantic_results", [])
        global_kw = state.get("global_keyword_results", [])
        
        # RRF ranking
        best_results = self._apply_rrf_ranking(
            personal_sem, personal_kw, global_sem, global_kw
        )
        
        logger.info(f"  RRF ranking: {len(best_results)} unique results")
        for i, r in enumerate(best_results[:3]):
            logger.info(f"    [{i+1}] Score: {r.get('score'):.2f}, Source: {r.get('source')}")
        
        state["best_search_results"] = best_results
        return state

    def _node_select_tools(self, state: WorkflowState) -> WorkflowState:
        """SELECT_TOOLS node - decide which tools to invoke."""
        logger.info("==== NODE: SELECT_TOOLS ====")
        
        query = state.get("rewritten_prompt") or state.get("user_prompt")
        results = state.get("best_search_results", [])
        
        tools = self._select_tools(query, results)
        
        logger.info(f"  Selected: {tools}")
        state["selected_tools"] = tools
        
        return state

    def _node_generate_answer(self, state: WorkflowState) -> WorkflowState:
        """GENERATE_ANSWER node - final LLM response."""
        logger.info("==== NODE: GENERATE_ANSWER ====")
        
        query = state.get("rewritten_prompt") or state.get("user_prompt")
        results = state.get("best_search_results", [])
        
        logger.info(f"  RAG results: {len(results)}")
        logger.info(f"  Tools: {state.get('selected_tools', [])}")
        
        answer = self._generate_with_llm(
            query, results, state.get("selected_tools", [])
        )
        
        state["generated_answer"] = answer
        logger.info(f"  ✓ Answer generated ({len(answer)} chars)")
        
        return state

    # Routing logic
    def _route_from_classify(self, state: WorkflowState) -> str:
        """Route after CLASSIFY node."""
        if state["prompt_type"] == "chitchat":
            return "chitchat"
        elif state["prompt_type"] == "file_only":
            return "extract"
        else:
            return "rewrite"

    def _route_from_rewrite(self, state: WorkflowState) -> str:
        """Route after REWRITE node."""
        if state.get("uploaded_files"):
            return "extract"
        elif state["prompt_type"] in ["request", "instruction"]:
            return "retrieve"
        else:
            return "generate"

    def _route_from_extract(self, state: WorkflowState) -> str:
        """Route after EXTRACT node."""
        if state.get("extracted_file_data"):
            return "ingest"
        elif state.get("user_prompt"):
            return "retrieve"
        else:
            return "generate"

    def _route_from_retrieve_personal(self, state: WorkflowState) -> str:
        """Route after RETRIEVE_PERSONAL node."""
        results = (
            state.get("personal_semantic_results", []) +
            state.get("personal_keyword_results", [])
        )
        if results:
            return "filter"
        else:
            return "global"

    def _route_from_retrieve_global(self, state: WorkflowState) -> str:
        """Route after RETRIEVE_GLOBAL node."""
        results = (
            state.get("global_semantic_results", []) +
            state.get("global_keyword_results", [])
        )
        if results:
            return "filter"
        else:
            return "select"

    # Helper methods (implement with actual logic)
    def _detect_prompt_type(self, prompt: str) -> str:
        # ...existing implementation...
        pass

    def _rewrite_with_context(self, query: str, history: List) -> str:
        # ...existing implementation...
        pass

    def _process_files(self, files: List) -> Dict:
        # ...existing implementation...
        pass

    def _ingest_to_rag(self, data: Dict, method: str) -> List[str]:
        # ...existing implementation...
        pass

    def _semantic_search(self, query: str) -> List[Dict]:
        # ...existing implementation...
        pass

    def _keyword_search(self, query: str) -> List[Dict]:
        # ...existing implementation...
        pass

    def _global_semantic_search(self, query: str) -> List[Dict]:
        # ...existing implementation...
        pass

    def _global_keyword_search(self, query: str) -> List[Dict]:
        # ...existing implementation...
        pass

    def _apply_rrf_ranking(self, *results) -> List[Dict]:
        # ...existing implementation...
        pass

    def _select_tools(self, query: str, results: List) -> List[str]:
        # ...existing implementation...
        pass

    def _generate_with_llm(self, query: str, results: List, tools: List) -> str:
        # ...existing implementation...
        pass

    def invoke(self, state: WorkflowState) -> WorkflowState:
        """Execute workflow."""
        return self.graph.invoke(state)

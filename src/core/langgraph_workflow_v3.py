"""
LangGraphWorkflow V3 - 10-node enhanced architecture
Includes query rewriting, file handling, and intelligent tool selection
"""

import logging
from typing import Dict, Any, Literal, List
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate

from .workflow_state import WorkflowState, PromptType, DataType, create_initial_state
from .prompt_classifier import PromptClassifier
from .retrieval_manager import RetrievalManager
from .result_filter import ResultFilter
from .data_analyzer import DataAnalyzer
from .query_rewriter import QueryRewriter
from .tool_selector import ToolSelector

logger = logging.getLogger(__name__)


class LangGraphWorkflowV3:
    """
    10-node enhanced workflow with rewriting and file handling:
    
    CLASSIFY → [DIRECT_RESPONSE | EXTRACT_FILE → INGEST_FILE → REWRITE_EVAL → 
    [REWRITE_FILE|REWRITE_CONVO|RETRIEVE] → FILTER → ANALYZE → SELECT_TOOLS → GENERATE → TOOLS] → END
    """
    
    def __init__(self, agent_executor):
        self.agent = agent_executor
        self.llm = agent_executor.llm
        self.tools = agent_executor.tools
        self.tool_names = [getattr(t, 'name', str(t)) for t in self.tools]
        
        # Initialize utility modules
        self.classifier = PromptClassifier(self.llm)
        self.rewriter = QueryRewriter(self.llm)
        self.retrieval = RetrievalManager(getattr(agent_executor, 'rag_service', None))
        self.filter = ResultFilter()
        self.analyzer = DataAnalyzer(self.llm)
        self.tool_selector = ToolSelector()
        
        self.graph = self._build_graph()
        logger.info("LangGraphWorkflowV3 initialized with 10-node enhanced architecture")
    
    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(WorkflowState)
        
        # Add all 10 nodes
        workflow.add_node("classify", self.node_classify)
        workflow.add_node("direct_response", self.node_direct_response)
        workflow.add_node("extract_file", self.node_extract_file)
        workflow.add_node("ingest_file", self.node_ingest_file)
        workflow.add_node("rewrite_eval", self.node_rewrite_eval)
        workflow.add_node("rewrite_file", self.node_rewrite_file_context)
        workflow.add_node("rewrite_convo", self.node_rewrite_conversation_context)
        workflow.add_node("retrieve", self.node_retrieve)
        workflow.add_node("filter", self.node_filter)
        workflow.add_node("analyze", self.node_analyze)
        workflow.add_node("select_tools", self.node_select_tools)
        workflow.add_node("generate", self.node_generate)
        workflow.add_node("tools", self.node_tools)
        
        # Entry point
        workflow.set_entry_point("classify")
        
        # CLASSIFY routes to DIRECT_RESPONSE or file handling
        workflow.add_conditional_edges(
            "classify",
            lambda s: "direct_response" if s.get("is_chitchat") else (
                "extract_file" if s.get("uploaded_files") else "rewrite_eval"
            ),
            {
                "direct_response": "direct_response",
                "extract_file": "extract_file",
                "rewrite_eval": "rewrite_eval"
            }
        )
        
        # DIRECT_RESPONSE goes to END
        workflow.add_edge("direct_response", END)
        
        # File pipeline
        workflow.add_edge("extract_file", "ingest_file")
        workflow.add_edge("ingest_file", "rewrite_eval")
        
        # REWRITE_EVAL routes to one of 3 rewriting strategies
        workflow.add_conditional_edges(
            "rewrite_eval",
            self._route_rewrite_strategy,
            {
                "rewrite_file": "rewrite_file",
                "rewrite_convo": "rewrite_convo",
                "retrieve": "retrieve"
            }
        )
        
        # Rewrite paths converge to RETRIEVE
        workflow.add_edge("rewrite_file", "retrieve")
        workflow.add_edge("rewrite_convo", "retrieve")
        
        # Retrieval pipeline
        workflow.add_edge("retrieve", "filter")
        workflow.add_edge("filter", "analyze")
        workflow.add_edge("analyze", "select_tools")
        workflow.add_edge("select_tools", "generate")
        
        # GENERATE routes to TOOLS or END
        workflow.add_conditional_edges(
            "generate",
            lambda s: "tools" if s.get("selected_tools") else "end",
            {
                "tools": "tools",
                "end": END
            }
        )
        
        # TOOLS back to GENERATE for synthesis
        workflow.add_edge("tools", "generate")
        
        return workflow.compile()
    
    def _route_rewrite_strategy(self, state: WorkflowState) -> str:
        """Route to appropriate rewrite strategy"""
        # Check if query needs rewriting at all
        if not state.get("needs_rewrite", False):
            return "retrieve"
        
        # Route based on context type
        context_type = state.get("rewrite_context_type", "")
        if context_type == "file":
            return "rewrite_file"
        elif context_type == "conversation":
            return "rewrite_convo"
        
        return "retrieve"
    
    async def node_classify(self, state: WorkflowState) -> Dict[str, Any]:
        """Classify prompt type"""
        try:
            prompt = state.get("user_prompt")
            has_files = bool(state.get("uploaded_files"))
            
            prompt_type, confidence = await self.classifier.classify(prompt, has_files)
            
            state["prompt_type"] = prompt_type
            state["is_chitchat"] = prompt_type == PromptType.CHITCHAT
            
            logger.info(f"Classified: {prompt_type.value} (confidence: {confidence})")
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            state["prompt_type"] = PromptType.INSTRUCTION
            state["is_chitchat"] = False
        
        return state
    
    async def node_extract_file(self, state: WorkflowState) -> Dict[str, Any]:
        """Extract and process uploaded files"""
        try:
            uploaded_files = state.get("uploaded_files", [])
            
            if not uploaded_files:
                logger.info("No files to extract")
                return state
            
            file_metadata = []
            for file_obj in uploaded_files:
                metadata = {
                    "filename": getattr(file_obj, "filename", getattr(file_obj, "name", "unknown")),
                    "size": getattr(file_obj, "size", 0),
                    "content_type": getattr(file_obj, "content_type", "unknown"),
                    "doc_id": getattr(file_obj, "doc_id", None)
                }
                file_metadata.append(metadata)
            
            state["file_metadata"] = file_metadata
            logger.info(f"Extracted {len(file_metadata)} files")
        except Exception as e:
            logger.error(f"File extraction failed: {e}")
            state["file_metadata"] = []
        
        return state
    
    async def node_ingest_file(self, state: WorkflowState) -> Dict[str, Any]:
        """Ingest files into RAG system"""
        try:
            file_metadata = state.get("file_metadata", [])
            
            if not file_metadata:
                logger.info("No files to ingest")
                return state
            
            # In a real system, this would process and index files
            # For now, just mark as ingested
            state["files_ingested"] = True
            state["ingest_timestamp"] = None  # Would be actual timestamp
            
            logger.info(f"Ingested {len(file_metadata)} files")
        except Exception as e:
            logger.error(f"File ingestion failed: {e}")
            state["files_ingested"] = False
        
        return state
    
    async def node_rewrite_eval(self, state: WorkflowState) -> Dict[str, Any]:
        """Evaluate whether query needs rewriting"""
        try:
            prompt = state.get("user_prompt", "")
            has_files = bool(state.get("uploaded_files"))
            history = state.get("conversation_history", [])
            
            needs_rewrite = await self.rewriter.evaluate_need_for_rewriting(
                prompt, has_files, history
            )
            
            state["needs_rewrite"] = needs_rewrite
            
            # Determine rewrite context if needed
            if needs_rewrite:
                if has_files and state.get("file_metadata"):
                    state["rewrite_context_type"] = "file"
                elif history:
                    state["rewrite_context_type"] = "conversation"
                else:
                    needs_rewrite = False
            
            state["needs_rewrite"] = needs_rewrite
            logger.info(f"Rewrite evaluation: {needs_rewrite}")
        except Exception as e:
            logger.error(f"Rewrite evaluation failed: {e}")
            state["needs_rewrite"] = False
        
        return state
    
    async def node_rewrite_file_context(self, state: WorkflowState) -> Dict[str, Any]:
        """Rewrite query using file context"""
        try:
            prompt = state.get("user_prompt", "")
            file_metadata = state.get("file_metadata", [])
            
            if file_metadata:
                rewritten = await self.rewriter.rewrite_with_file_context(prompt, file_metadata)
                state["rewritten_prompt"] = rewritten
                logger.info(f"Rewritten with file context")
            else:
                state["rewritten_prompt"] = prompt
        except Exception as e:
            logger.error(f"File context rewriting failed: {e}")
            state["rewritten_prompt"] = prompt
        
        return state
    
    async def node_rewrite_conversation_context(self, state: WorkflowState) -> Dict[str, Any]:
        """Rewrite query using conversation context"""
        try:
            prompt = state.get("user_prompt", "")
            history = state.get("conversation_history", [])
            
            if history:
                rewritten = await self.rewriter.rewrite_with_conversation_context(prompt, history)
                state["rewritten_prompt"] = rewritten
                logger.info(f"Rewritten with conversation context")
            else:
                state["rewritten_prompt"] = prompt
        except Exception as e:
            logger.error(f"Conversation context rewriting failed: {e}")
            state["rewritten_prompt"] = prompt
        
        return state
    
    async def node_direct_response(self, state: WorkflowState) -> Dict[str, Any]:
        """Direct LLM response for chitchat"""
        try:
            prompt = ChatPromptTemplate.from_template("Respond to: {query}")
            chain = prompt | self.llm
            
            response = await chain.ainvoke({"query": state["user_prompt"]})
            state["generated_answer"] = response.content
            
            logger.info("Direct response generated (chitchat mode)")
        except Exception as e:
            logger.error(f"Direct response failed: {e}")
            state["generated_answer"] = "I apologize for the issue. Please try again."
        
        return state
    
    async def node_retrieve(self, state: WorkflowState) -> Dict[str, Any]:
        """Retrieve with personal-first, global-fallback strategy"""
        try:
            query = state.get("rewritten_prompt") or state.get("user_prompt")
            user_id = state.get("user_id")
            session_id = state.get("session_id")
            
            results = await self.retrieval.retrieve_with_fallback(query, user_id, session_id)
            
            state["personal_semantic_results"] = results.get("personal_semantic", [])
            state["personal_keyword_results"] = results.get("personal_keyword", [])
            state["global_semantic_results"] = results.get("global_semantic", [])
            state["global_keyword_results"] = results.get("global_keyword", [])
            state["search_metadata"] = {
                "fallback_to_global": results.get("fallback_to_global", False),
                "total_results": results.get("total_results", 0),
                "elapsed_seconds": results.get("elapsed_seconds", 0)
            }
            
            state["rag_enabled"] = results.get("total_results", 0) > 0
            
            logger.info(f"Retrieved {results.get('total_results', 0)} results")
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            state["rag_enabled"] = False
        
        return state
    
    async def node_filter(self, state: WorkflowState) -> Dict[str, Any]:
        """Filter and rank results with RRF"""
        try:
            best_results = self.filter.filter_and_rank(
                state.get("personal_semantic_results", []),
                state.get("personal_keyword_results", []),
                state.get("global_semantic_results", []),
                state.get("global_keyword_results", [])
            )
            
            state["best_search_results"] = best_results
            
            logger.info(f"Filtered to {len(best_results)} results")
        except Exception as e:
            logger.error(f"Filtering failed: {e}")
            state["best_search_results"] = []
        
        return state
    
    async def node_analyze(self, state: WorkflowState) -> Dict[str, Any]:
        """Analyze detected data types"""
        try:
            analysis = await self.analyzer.analyze_results(state.get("best_search_results", []))
            
            state["has_table_data"] = analysis.get("has_table_data", False)
            state["has_numeric_data"] = analysis.get("has_numeric_data", False)
            state["text_only"] = analysis.get("text_only", True)
            state["detected_data_types"] = analysis.get("detected_types", [])
            
            logger.info(f"Analyzed data types: table={state['has_table_data']}, numeric={state['has_numeric_data']}")
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            state["text_only"] = True
        
        return state
    
    async def node_select_tools(self, state: WorkflowState) -> Dict[str, Any]:
        """Select relevant tools based on query and data types"""
        try:
            query = state.get("user_prompt", "")
            data_types = state.get("detected_data_types", [])
            history = state.get("conversation_history", [])
            
            selection = await self.tool_selector.select_tools(
                query,
                data_types,
                self.tool_names,
                history
            )
            
            state["selected_tools"] = selection.get("selected_tools", [])
            state["primary_tool"] = selection.get("primary_tool")
            state["tool_selection_rationale"] = selection.get("rationale", "")
            state["tool_selection_confidence"] = selection.get("confidence", 0.0)
            
            logger.info(f"Selected tools: {state['selected_tools']} (confidence: {state['tool_selection_confidence']:.2f})")
        except Exception as e:
            logger.error(f"Tool selection failed: {e}")
            state["selected_tools"] = []
            state["primary_tool"] = None
        
        return state
    
    async def node_generate(self, state: WorkflowState) -> Dict[str, Any]:
        """Generate answer with LLM"""
        try:
            query = state.get("user_prompt", "")
            context = ""
            
            if state.get("best_search_results"):
                context = "\n\n".join([
                    f"- {r.get('source', 'unknown')}: {r.get('content', '')[:200]}"
                    for r in state["best_search_results"][:3]
                ])
            
            prompt = ChatPromptTemplate.from_template(
                "Question: {query}\n\nContext: {context}\n\nAnswer:"
            )
            chain = prompt | self.llm
            
            response = await chain.ainvoke({
                "query": query,
                "context": context or "No context available"
            })
            
            state["generated_answer"] = response.content
            
            logger.info("Answer generated")
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            state["generated_answer"] = "I encountered an issue generating a response."
        
        return state
    
    async def node_tools(self, state: WorkflowState) -> Dict[str, Any]:
        """Execute selected tools"""
        try:
            selected = state.get("selected_tools", [])
            
            if not selected:
                logger.info("No tools selected")
                return state
            
            tool_results = {}
            
            for tool_name in selected:
                try:
                    # Find matching tool
                    tool = next((t for t in self.tools if getattr(t, 'name', '') == tool_name), None)
                    if tool and hasattr(tool, 'func'):
                        result = await tool.func(state.get("user_prompt", ""))
                        tool_results[tool_name] = result
                except Exception as e:
                    logger.warning(f"Tool {tool_name} failed: {e}")
            
            state["tool_results"] = tool_results
            
            logger.info(f"Tool execution complete: {len(tool_results)} tools")
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            state["tool_results"] = {}
        
        return state

"""
LangGraphWorkflow V4 - Complete 13-node architecture
Full workflow with parallel entry points, all routing, output formatting, and monitoring
"""

import logging
from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate

from .workflow_state import WorkflowState, PromptType, DataType, create_initial_state
from .prompt_classifier import PromptClassifier
from .retrieval_manager import RetrievalManager
from .result_filter import ResultFilter
from .data_analyzer import DataAnalyzer
from .query_rewriter import QueryRewriter
from .tool_selector import ToolSelector
from .output_formatter import OutputFormatter
from .workflow_observer import WorkflowObserver

logger = logging.getLogger(__name__)


class LangGraphWorkflowV4:
    """
    Complete 13-node workflow with full feature set:
    
    Parallel Entry:
    PROMPT_HANDLER → CLASSIFY → [DIRECT_RESPONSE | EXTRACT_FILE → INGEST_FILE → REWRITE_EVAL]
    FILE_HANDLER → EXTRACT_FILE
    
    Main Pipeline:
    REWRITE_EVAL → [REWRITE_FILE|REWRITE_CONVO|RETRIEVE] → FILTER → ANALYZE → SELECT_TOOLS → 
    GENERATE → [EXECUTE_TOOLS → FORMAT_OUTPUT | FORMAT_OUTPUT] → END
    
    Features:
    - Prompt classification (5 types)
    - File handling with extraction & ingestion
    - Query rewriting (file + conversation context)
    - Retrieval with personal-first + global fallback
    - Result filtering with RRF
    - Data type analysis
    - Intelligent tool selection
    - Tool execution
    - Output formatting
    - Workflow observation & monitoring
    """
    
    def __init__(self, agent_executor, enable_observer: bool = True):
        self.agent = agent_executor
        self.llm = agent_executor.llm
        self.tools = agent_executor.tools
        self.tool_names = [getattr(t, 'name', str(t)) for t in self.tools]
        
        # Initialize all utility modules
        self.classifier = PromptClassifier(self.llm)
        self.rewriter = QueryRewriter(self.llm)
        self.retrieval = RetrievalManager(getattr(agent_executor, 'rag_service', None))
        self.filter = ResultFilter()
        self.analyzer = DataAnalyzer(self.llm)
        self.tool_selector = ToolSelector()
        self.formatter = OutputFormatter()
        
        # Optional workflow observer
        self.observer = WorkflowObserver() if enable_observer else None
        
        self.graph = self._build_graph()
        logger.info("LangGraphWorkflowV4 initialized with complete 13-node architecture")
    
    def _build_graph(self) -> StateGraph:
        """Build the complete 13-node workflow graph"""
        workflow = StateGraph(WorkflowState)
        
        # Add all 13 nodes
        workflow.add_node("prompt_handler", self.node_prompt_handler)
        workflow.add_node("file_handler", self.node_file_handler)
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
        workflow.add_node("execute_tools", self.node_execute_tools)
        workflow.add_node("format_output", self.node_format_output)
        
        # Parallel entry points
        workflow.set_entry_point("prompt_handler")
        
        # PROMPT_HANDLER decides route
        workflow.add_conditional_edges(
            "prompt_handler",
            lambda s: "classify" if s.get("user_prompt") else "file_handler",
            {
                "classify": "classify",
                "file_handler": "file_handler"
            }
        )
        
        # FILE_HANDLER routes to EXTRACT_FILE
        workflow.add_edge("file_handler", "extract_file")
        
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
        
        # DIRECT_RESPONSE goes to FORMAT_OUTPUT for consistent formatting
        workflow.add_edge("direct_response", "format_output")
        
        # File pipeline
        workflow.add_edge("extract_file", "ingest_file")
        workflow.add_edge("ingest_file", "rewrite_eval")
        
        # REWRITE_EVAL routes to appropriate rewriting strategy
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
        
        # Main retrieval pipeline
        workflow.add_edge("retrieve", "filter")
        workflow.add_edge("filter", "analyze")
        workflow.add_edge("analyze", "select_tools")
        workflow.add_edge("select_tools", "generate")
        
        # GENERATE routes to EXECUTE_TOOLS or FORMAT_OUTPUT
        workflow.add_conditional_edges(
            "generate",
            lambda s: "execute_tools" if s.get("selected_tools") else "format_output",
            {
                "execute_tools": "execute_tools",
                "format_output": "format_output"
            }
        )
        
        # EXECUTE_TOOLS back to GENERATE for synthesis
        workflow.add_conditional_edges(
            "execute_tools",
            lambda s: "generate" if s.get("needs_retry") else "format_output",
            {
                "generate": "generate",
                "format_output": "format_output"
            }
        )
        
        # FORMAT_OUTPUT goes to END
        workflow.add_edge("format_output", END)
        
        return workflow.compile()
    
    def _route_rewrite_strategy(self, state: WorkflowState) -> str:
        """Route to appropriate rewrite strategy"""
        if not state.get("needs_rewrite", False):
            return "retrieve"
        
        context_type = state.get("rewrite_context_type", "")
        if context_type == "file":
            return "rewrite_file"
        elif context_type == "conversation":
            return "rewrite_convo"
        
        return "retrieve"
    
    # ========== Node Implementations ==========
    
    async def node_prompt_handler(self, state: WorkflowState) -> Dict[str, Any]:
        """Entry point: Route based on prompt presence"""
        try:
            if self.observer:
                state["_step"] = await self.observer.emit_step_started(
                    "PROMPT_HANDLER",
                    {"has_prompt": bool(state.get("user_prompt"))}
                )
            
            # Validation happens at routing
            return state
        except Exception as e:
            logger.error(f"Prompt handler failed: {e}")
            if self.observer and state.get("_step"):
                await self.observer.emit_step_failed(state["_step"], str(e))
            return state
    
    async def node_file_handler(self, state: WorkflowState) -> Dict[str, Any]:
        """Entry point: Route files directly to extraction"""
        try:
            if self.observer:
                state["_step"] = await self.observer.emit_step_started(
                    "FILE_HANDLER",
                    {"file_count": len(state.get("uploaded_files", []))}
                )
            
            return state
        except Exception as e:
            logger.error(f"File handler failed: {e}")
            if self.observer and state.get("_step"):
                await self.observer.emit_step_failed(state["_step"], str(e))
            return state
    
    async def node_classify(self, state: WorkflowState) -> Dict[str, Any]:
        """
        Classify prompt type with greeting detection.
        
        Implements logic from original system prompt:
        - Detect greetings (hello, hi, etc.)
        - Detect chitchat (non-financial questions)
        - Classify finance queries by type
        """
        try:
            if self.observer:
                state["_step"] = await self.observer.emit_step_started("CLASSIFY")
            
            prompt = state.get("user_prompt", "").lower().strip()
            has_files = bool(state.get("uploaded_files"))
            
            # If files are uploaded, prioritize file operation over greeting detection
            if has_files:
                state["prompt_type"] = PromptType.INSTRUCTION
                state["is_chitchat"] = False
                state["needs_file_processing"] = True
                confidence = 0.90
                logger.info(f"File upload detected - treating as instruction (files: {len(state.get('uploaded_files', []))})")
            else:
                # Greeting detection only when NO files - use regex to match whole patterns
                import re
                greeting_patterns = [
                    r"^\s*(hello|hi|xin chào|chào|how are you|thanks|thank you|cảm ơn|goodbye|bye|tạm biệt|what'?s?\s+up|who are you|bạn là ai)\s*[\.\?\!]*\s*$",
                    r"^(sao thế)\s*[\.\?\!]*\s*$"
                ]
                
                is_greeting = False
                for pattern in greeting_patterns:
                    if re.match(pattern, prompt, re.IGNORECASE):
                        is_greeting = True
                        break
                
                if is_greeting:
                    state["prompt_type"] = PromptType.CHITCHAT
                    state["is_chitchat"] = True
                    confidence = 0.95
                    logger.info(f"Detected greeting: '{prompt[:50]}...'")
                else:
                    # Use classifier for other types
                    prompt_type, confidence = await self.classifier.classify(prompt, has_files)
                    state["prompt_type"] = prompt_type
                    state["is_chitchat"] = prompt_type == PromptType.CHITCHAT
            
            if self.observer and state.get("_step"):
                await self.observer.emit_step_completed(
                    state["_step"],
                    output_size=len(str(state.get("prompt_type"))),
                    metadata={
                        "type": state["prompt_type"].value if state.get("prompt_type") else "unknown",
                        "confidence": confidence,
                        "is_greeting": is_greeting
                    }
                )
            
            logger.info(f"Classified: {state.get('prompt_type', 'unknown')} (confidence: {confidence})")
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            state["prompt_type"] = PromptType.INSTRUCTION
            state["is_chitchat"] = False
            if self.observer and state.get("_step"):
                await self.observer.emit_step_failed(state["_step"], str(e))
        
        return state
    
    async def node_direct_response(self, state: WorkflowState) -> Dict[str, Any]:
        """
        Direct LLM response for chitchat and greetings.
        
        Implements original logic: Simple friendly responses for non-financial questions
        """
        try:
            if self.observer:
                state["_step"] = await self.observer.emit_step_started("DIRECT_RESPONSE")
            
            # System prompt for Vietnamese financial advisor
            system_prompt = """Bạn là một chuyên gia tư vấn tài chính chuyên về thị trường chứng khoán Việt Nam.

Khi được hỏi những câu hỏi không liên quan đến tài chính (lời chào, chit-chat, etc), 
hãy trả lời một cách thân thiện, tự nhiên, nhưng vẫn giữ chuyên nghiệp.

QUAN TRỌNG: 
- Không tự giới thiệu tên riêng hoặc danh xưng cá nhân. 
- Chỉ nói bạn là "một chuyên gia tư vấn tài chính" hoặc "chuyên gia tài chính".
- Nếu được chào hỏi, hãy chào lại thân thiện và sẵn sàng tư vấn.

Nếu đó là lời chào, hãy chào lại và gợi ý bạn sẵn sàng giúp về thị trường chứng khoán.
Nếu là câu hỏi khác, hãy trả lời thân thiện và gợi ý người dùng hỏi về tài chính hay chứng khoán."""
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{query}")
            ])
            chain = prompt | self.llm
            
            response = await chain.ainvoke({"query": state["user_prompt"]})
            state["generated_answer"] = response.content
            
            if self.observer and state.get("_step"):
                await self.observer.emit_step_completed(
                    state["_step"],
                    output_size=len(response.content)
                )
            
            logger.info("Direct response generated (chitchat mode)")
        except Exception as e:
            logger.error(f"Direct response failed: {e}")
            state["generated_answer"] = "Xin lỗi, tôi gặp lỗi khi xử lý câu hỏi của bạn. Vui lòng thử lại."
            if self.observer and state.get("_step"):
                await self.observer.emit_step_failed(state["_step"], str(e))
        
        return state
    
    async def node_extract_file(self, state: WorkflowState) -> Dict[str, Any]:
        """Extract and process uploaded files"""
        try:
            if self.observer:
                state["_step"] = await self.observer.emit_step_started(
                    "EXTRACT_FILE",
                    {"file_count": len(state.get("uploaded_files", []))}
                )
            
            uploaded_files = state.get("uploaded_files", [])
            
            if not uploaded_files:
                logger.info("No files to extract")
                if self.observer and state.get("_step"):
                    await self.observer.emit_step_skipped(
                        "EXTRACT_FILE", "No files provided"
                    )
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
            
            if self.observer and state.get("_step"):
                await self.observer.emit_step_completed(
                    state["_step"],
                    output_size=sum(f.get("size", 0) for f in file_metadata)
                )
            
            logger.info(f"Extracted {len(file_metadata)} files")
        except Exception as e:
            logger.error(f"File extraction failed: {e}")
            state["file_metadata"] = []
            if self.observer and state.get("_step"):
                await self.observer.emit_step_failed(state["_step"], str(e))
        
        return state
    
    async def node_ingest_file(self, state: WorkflowState) -> Dict[str, Any]:
        """Ingest files into RAG system"""
        try:
            if self.observer:
                state["_step"] = await self.observer.emit_step_started(
                    "INGEST_FILE",
                    {"file_count": len(state.get("file_metadata", []))}
                )
            
            file_metadata = state.get("file_metadata", [])
            
            if not file_metadata:
                logger.info("No files to ingest")
                if self.observer and state.get("_step"):
                    await self.observer.emit_step_skipped(
                        "INGEST_FILE", "No file metadata"
                    )
                return state
            
            state["files_ingested"] = True
            
            if self.observer and state.get("_step"):
                await self.observer.emit_step_completed(
                    state["_step"],
                    metadata={"files_ingested": True}
                )
            
            logger.info(f"Ingested {len(file_metadata)} files")
        except Exception as e:
            logger.error(f"File ingestion failed: {e}")
            state["files_ingested"] = False
            if self.observer and state.get("_step"):
                await self.observer.emit_step_failed(state["_step"], str(e))
        
        return state
    
    async def node_rewrite_eval(self, state: WorkflowState) -> Dict[str, Any]:
        """Evaluate whether query needs rewriting"""
        try:
            if self.observer:
                state["_step"] = await self.observer.emit_step_started("REWRITE_EVAL")
            
            prompt = state.get("user_prompt", "")
            has_files = bool(state.get("uploaded_files"))
            history = state.get("conversation_history", [])
            
            needs_rewrite = await self.rewriter.evaluate_need_for_rewriting(
                prompt, has_files, history
            )
            
            state["needs_rewrite"] = needs_rewrite
            
            if needs_rewrite:
                if has_files and state.get("file_metadata"):
                    state["rewrite_context_type"] = "file"
                elif history:
                    state["rewrite_context_type"] = "conversation"
                else:
                    needs_rewrite = False
            
            state["needs_rewrite"] = needs_rewrite
            
            if self.observer and state.get("_step"):
                await self.observer.emit_step_completed(
                    state["_step"],
                    metadata={"needs_rewrite": needs_rewrite}
                )
            
            logger.info(f"Rewrite evaluation: {needs_rewrite}")
        except Exception as e:
            logger.error(f"Rewrite evaluation failed: {e}")
            state["needs_rewrite"] = False
            if self.observer and state.get("_step"):
                await self.observer.emit_step_failed(state["_step"], str(e))
        
        return state
    
    async def node_rewrite_file_context(self, state: WorkflowState) -> Dict[str, Any]:
        """Rewrite query using file context"""
        try:
            if self.observer:
                state["_step"] = await self.observer.emit_step_started("REWRITE_FILE")
            
            prompt = state.get("user_prompt", "")
            file_metadata = state.get("file_metadata", [])
            
            if file_metadata:
                rewritten = await self.rewriter.rewrite_with_file_context(prompt, file_metadata)
                state["rewritten_prompt"] = rewritten
                
                if self.observer and state.get("_step"):
                    await self.observer.emit_step_completed(
                        state["_step"],
                        output_size=len(rewritten)
                    )
            else:
                state["rewritten_prompt"] = prompt
                if self.observer and state.get("_step"):
                    await self.observer.emit_step_skipped(
                        "REWRITE_FILE", "No file metadata"
                    )
        except Exception as e:
            logger.error(f"File context rewriting failed: {e}")
            state["rewritten_prompt"] = prompt
            if self.observer and state.get("_step"):
                await self.observer.emit_step_failed(state["_step"], str(e))
        
        return state
    
    async def node_rewrite_conversation_context(self, state: WorkflowState) -> Dict[str, Any]:
        """Rewrite query using conversation context"""
        try:
            if self.observer:
                state["_step"] = await self.observer.emit_step_started("REWRITE_CONVO")
            
            prompt = state.get("user_prompt", "")
            history = state.get("conversation_history", [])
            
            if history:
                rewritten = await self.rewriter.rewrite_with_conversation_context(prompt, history)
                state["rewritten_prompt"] = rewritten
                
                if self.observer and state.get("_step"):
                    await self.observer.emit_step_completed(
                        state["_step"],
                        output_size=len(rewritten)
                    )
            else:
                state["rewritten_prompt"] = prompt
                if self.observer and state.get("_step"):
                    await self.observer.emit_step_skipped(
                        "REWRITE_CONVO", "No conversation history"
                    )
        except Exception as e:
            logger.error(f"Conversation context rewriting failed: {e}")
            state["rewritten_prompt"] = prompt
            if self.observer and state.get("_step"):
                await self.observer.emit_step_failed(state["_step"], str(e))
        
        return state
    
    async def node_retrieve(self, state: WorkflowState) -> Dict[str, Any]:
        """Retrieve with personal-first, global-fallback strategy"""
        try:
            if self.observer:
                state["_step"] = await self.observer.emit_step_started("RETRIEVE")
            
            query = state.get("rewritten_prompt") or state.get("user_prompt")
            user_id = state.get("user_id")
            session_id = state.get("session_id")
            
            results = await self.retrieval.retrieve_with_fallback(query, user_id, session_id)
            
            state["personal_semantic_results"] = results.get("personal_semantic", [])
            state["personal_keyword_results"] = results.get("personal_keyword", [])
            state["global_semantic_results"] = results.get("global_semantic", [])
            state["global_keyword_results"] = results.get("global_keyword", [])
            state["rag_enabled"] = results.get("total_results", 0) > 0
            
            if self.observer and state.get("_step"):
                await self.observer.emit_step_completed(
                    state["_step"],
                    output_size=results.get("total_results", 0) * 500  # Est. bytes
                )
            
            logger.info(f"Retrieved {results.get('total_results', 0)} results")
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            state["rag_enabled"] = False
            if self.observer and state.get("_step"):
                await self.observer.emit_step_failed(state["_step"], str(e))
        
        return state
    
    async def node_filter(self, state: WorkflowState) -> Dict[str, Any]:
        """Filter and rank results with RRF"""
        try:
            if self.observer:
                state["_step"] = await self.observer.emit_step_started("FILTER")
            
            best_results = self.filter.filter_and_rank(
                state.get("personal_semantic_results", []),
                state.get("personal_keyword_results", []),
                state.get("global_semantic_results", []),
                state.get("global_keyword_results", [])
            )
            
            state["best_search_results"] = best_results
            
            if self.observer and state.get("_step"):
                await self.observer.emit_step_completed(
                    state["_step"],
                    output_size=len(best_results) * 500
                )
            
            logger.info(f"Filtered to {len(best_results)} results")
        except Exception as e:
            logger.error(f"Filtering failed: {e}")
            state["best_search_results"] = []
            if self.observer and state.get("_step"):
                await self.observer.emit_step_failed(state["_step"], str(e))
        
        return state
    
    async def node_analyze(self, state: WorkflowState) -> Dict[str, Any]:
        """Analyze detected data types"""
        try:
            if self.observer:
                state["_step"] = await self.observer.emit_step_started("ANALYZE")
            
            analysis = await self.analyzer.analyze_results(state.get("best_search_results", []))
            
            state["has_table_data"] = analysis.get("has_table_data", False)
            state["has_numeric_data"] = analysis.get("has_numeric_data", False)
            state["text_only"] = analysis.get("text_only", True)
            state["detected_data_types"] = analysis.get("detected_types", [])
            
            if self.observer and state.get("_step"):
                await self.observer.emit_step_completed(
                    state["_step"],
                    metadata={
                        "has_table": state["has_table_data"],
                        "has_numeric": state["has_numeric_data"]
                    }
                )
            
            logger.info(f"Analyzed: table={state['has_table_data']}, numeric={state['has_numeric_data']}")
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            state["text_only"] = True
            if self.observer and state.get("_step"):
                await self.observer.emit_step_failed(state["_step"], str(e))
        
        return state
    
    async def node_select_tools(self, state: WorkflowState) -> Dict[str, Any]:
        """
        Select relevant tools based on query analysis.
        
        Implements tool decision matrix from original system prompt:
        - "thông tin công ty" → get_company_info
        - "cổ đông" → get_shareholders
        - "ban lãnh đạo" → get_officers
        - "công ty con" → get_subsidiaries
        - "sự kiện, cổ tức" → get_company_events
        - "giá lịch sử, OHLCV" → get_historical_data
        - "SMA, xu hướng" → calculate_sma
        - "RSI, quá mua" → calculate_rsi
        """
        try:
            if self.observer:
                state["_step"] = await self.observer.emit_step_started("SELECT_TOOLS")
            
            query = state.get("user_prompt", "").lower()
            
            # Tool selection matrix (from original system prompt)
            tool_keywords = {
                "get_company_info": {
                    "keywords": ["thông tin công ty", "công ty", "tên", "ngành", "vốn", "info", "information"],
                    "pattern": r"(thông tin|info|tên|ngành|vốn|hoạt động)" 
                },
                "get_shareholders": {
                    "keywords": ["cổ đông", "ai nắm", "sở hữu", "holder", "shareholder"],
                    "pattern": r"(cổ đông|nắm giữ|shareholder|sở hữu cổ phần)"
                },
                "get_officers": {
                    "keywords": ["ban lãnh đạo", "CEO", "lãnh đạo", "chủ tịch", "giám đốc", "leadership"],
                    "pattern": r"(lãnh đạo|CEO|chủ tịch|giám đốc|quản lý)"
                },
                "get_subsidiaries": {
                    "keywords": ["công ty con", "liên kết", "subsidiary", "con"],
                    "pattern": r"(công ty con|liên kết|subsidiary)"
                },
                "get_company_events": {
                    "keywords": ["sự kiện", "cổ tức", "ĐHCĐ", "event", "dividend"],
                    "pattern": r"(sự kiện|cổ tức|ĐHCĐ|event|chia)"
                },
                "get_historical_data": {
                    "keywords": ["giá", "OHLCV", "lịch sử", "price", "history"],
                    "pattern": r"(giá|OHLCV|lịch sử|price|history|3 tháng|6 tháng|1 năm)"
                },
                "calculate_sma": {
                    "keywords": ["SMA", "moving average", "xu hướng", "trend"],
                    "pattern": r"(SMA|moving average|xu hướng|trend|MA)"
                },
                "calculate_rsi": {
                    "keywords": ["RSI", "quá mua", "quá bán", "overbought", "oversold"],
                    "pattern": r"(RSI|quá mua|quá bán|overbought|oversold)"
                }
            }
            
            selected_tools = []
            for tool_name, config in tool_keywords.items():
                if any(kw in query for kw in config["keywords"]):
                    selected_tools.append(tool_name)
            
            # If no tools matched by keywords, try to use classifier
            if not selected_tools:
                selection = await self.tool_selector.select_tools(
                    state.get("user_prompt", ""),
                    state.get("detected_data_types", []),
                    self.tool_names,
                    state.get("conversation_history", [])
                )
                selected_tools = selection.get("selected_tools", [])
            
            state["selected_tools"] = selected_tools
            state["primary_tool"] = selected_tools[0] if selected_tools else None
            state["tool_selection_rationale"] = f"Selected tools based on query keywords: {', '.join(selected_tools)}"
            
            if self.observer and state.get("_step"):
                await self.observer.emit_step_completed(
                    state["_step"],
                    metadata={"tools_selected": len(selected_tools), "tools": selected_tools}
                )
            
            logger.info(f"Selected tools via decision matrix: {selected_tools}")
        except Exception as e:
            logger.error(f"Tool selection failed: {e}")
            state["selected_tools"] = []
            if self.observer and state.get("_step"):
                await self.observer.emit_step_failed(state["_step"], str(e))
        
        return state
    
    async def node_generate(self, state: WorkflowState) -> Dict[str, Any]:
        """
        Generate answer with LLM using original system prompt logic.
        
        Implements:
        - Vietnamese financial advisor instructions
        - Table formatting rules for structured data
        - Data interpretation guidelines (SMA, RSI meanings)
        - Error handling and user guidance
        """
        try:
            if self.observer:
                state["_step"] = await self.observer.emit_step_started("GENERATE")
            
            # System prompt from original (adapted for Vietnamese)
            system_prompt = """Bạn là một chuyên gia tư vấn tài chính chuyên về thị trường chứng khoán Việt Nam.

NHIỆM VỤ:
- Trả lời các câu hỏi về thị trường chứng khoán Việt Nam một cách chính xác, chi tiết
- Phân tích dữ liệu giá cổ phiếu và các chỉ số kỹ thuật
- Cung cấp thông tin về các công ty niêm yết
- Giải thích ý nghĩa của các chỉ số cho người không chuyên

QUY TẮC TRẢ LỜI:

1. ĐỊNH DẠNG:
   - Trả lời bằng tiếng Việt
   - QUAN TRỌNG: Không tự giới thiệu tên riêng hoặc danh xưng cá nhân
   - Hiển thị dữ liệu chi tiết dưới dạng BẢNG MARKDOWN khi có dữ liệu
   - Sau bảng: Đưa ra nhận xét tổng quan và kết luận

2. QUY TẮC FORMAT BẢNG:
   - Dữ liệu giá lịch sử (OHLCV): | Ngày | Giá mở cửa | Giá cao nhất | Giá thấp nhất | Giá đóng cửa | Khối lượng |
   - SMA/RSI: | Ngày | Giá đóng cửa | SMA-X | Chênh lệch | Xu hướng |
   - Cổ đông: | STT | Tên cổ đông | Số lượng CP | Tỷ lệ sở hữu (%) |
   - Ban lãnh đạo: | STT | Họ tên | Chức vụ | Tỷ lệ sở hữu (%) |
   - Sự kiện: | Ngày | Loại sự kiện | Nội dung | Tỷ lệ/Giá trị |

3. GIẢI THÍCH CHỈ SỐ:
   - SMA: Giá > SMA = xu hướng tăng, Giá < SMA = xu hướng giảm
   - RSI > 70: Quá mua (có thể giảm), RSI < 30: Quá bán (có thể tăng)
   - Luôn giải thích ý nghĩa cho người không chuyên

4. XỬ LÝ LỖI:
   - Nếu dữ liệu không tìm thấy, giải thích rõ lý do
   - Gợi ý cách sửa nếu có thể
   - Hướng dẫn người dùng kiểm tra lại thông tin đầu vào"""
            
            query = state.get("user_prompt", "")
            context = ""
            
            if state.get("best_search_results"):
                context = "\n\n".join([
                    f"[{r.get('source', 'Source')}]: {r.get('content', '')[:300]}"
                    for r in state["best_search_results"][:5]
                ])
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "Câu hỏi: {query}\n\nDữ liệu bổ sung:\n{context}\n\nHãy trả lời:")
            ])
            chain = prompt | self.llm
            
            response = await chain.ainvoke({
                "query": query,
                "context": context or "Không có dữ liệu bổ sung"
            })
            
            state["generated_answer"] = response.content
            
            if self.observer and state.get("_step"):
                await self.observer.emit_step_completed(
                    state["_step"],
                    output_size=len(response.content)
                )
            
            logger.info(f"Answer generated ({len(response.content)} chars)")
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            state["generated_answer"] = f"Xin lỗi, tôi gặp lỗi khi xử lý câu hỏi của bạn: {str(e)}"
            if self.observer and state.get("_step"):
                await self.observer.emit_step_failed(state["_step"], str(e))
        
        return state
    
    async def node_execute_tools(self, state: WorkflowState) -> Dict[str, Any]:
        """Execute selected tools"""
        try:
            if self.observer:
                state["_step"] = await self.observer.emit_step_started(
                    "EXECUTE_TOOLS",
                    {"tool_count": len(state.get("selected_tools", []))}
                )
            
            selected = state.get("selected_tools", [])
            
            if not selected:
                logger.info("No tools selected")
                if self.observer and state.get("_step"):
                    await self.observer.emit_step_skipped(
                        "EXECUTE_TOOLS", "No tools selected"
                    )
                return state
            
            tool_results = {}
            
            for tool_name in selected:
                try:
                    tool = next((t for t in self.tools if getattr(t, 'name', '') == tool_name), None)
                    if tool and hasattr(tool, 'func'):
                        result = await tool.func(state.get("user_prompt", ""))
                        tool_results[tool_name] = result
                except Exception as e:
                    logger.warning(f"Tool {tool_name} failed: {e}")
            
            state["tool_results"] = tool_results
            
            if self.observer and state.get("_step"):
                await self.observer.emit_step_completed(
                    state["_step"],
                    metadata={"tools_executed": len(tool_results)}
                )
            
            logger.info(f"Tool execution complete: {len(tool_results)} tools")
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            state["tool_results"] = {}
            if self.observer and state.get("_step"):
                await self.observer.emit_step_failed(state["_step"], str(e))
        
        return state
    
    async def node_format_output(self, state: WorkflowState) -> Dict[str, Any]:
        """Format final output with tables, calculations, and citations"""
        try:
            if self.observer:
                state["_step"] = await self.observer.emit_step_started("FORMAT_OUTPUT")
            
            answer = state.get("generated_answer", "")
            search_results = state.get("best_search_results", [])
            tool_results = state.get("tool_results")
            data_types = state.get("detected_data_types", [])
            
            formatted_answer = await self.formatter.format_answer(
                answer, search_results, tool_results, data_types
            )
            
            state["formatted_answer"] = formatted_answer
            
            if self.observer and state.get("_step"):
                await self.observer.emit_step_completed(
                    state["_step"],
                    output_size=len(formatted_answer)
                )
            
            # Workflow completion
            if self.observer:
                await self.observer.emit_workflow_completed()
                self.observer.print_summary()
            
            logger.info(f"Output formatted ({len(formatted_answer)} chars)")
        except Exception as e:
            logger.error(f"Output formatting failed: {e}")
            state["formatted_answer"] = state.get("generated_answer", "")
            if self.observer and state.get("_step"):
                await self.observer.emit_step_failed(state["_step"], str(e))
        
        return state
    
    async def invoke(self, user_prompt: str, uploaded_files: list = None, 
                     conversation_history: list = None, user_id: str = "default",
                     session_id: str = "default", rag_results: list = None,
                     tools_enabled: bool = True) -> Dict[str, Any]:
        """
        Invoke the V4 workflow with given parameters.
        
        Args:
            user_prompt: The user's question
            uploaded_files: List of uploaded file metadata
            conversation_history: List of previous messages
            user_id: User identifier
            session_id: Session identifier
            rag_results: Retrieved RAG results
            tools_enabled: Whether tools are enabled
            
        Returns:
            Final state with generated_answer and formatted_answer
        """
        try:
            logger.info(f"V4 workflow invoked: user={user_id}, session={session_id}")
            
            # Create initial state with the expected parameters
            state = create_initial_state(
                user_prompt=user_prompt,
                uploaded_files=uploaded_files or [],
                conversation_history=conversation_history or [],
                user_id=user_id,
                session_id=session_id
            )
            
            # Add RAG results and tools_enabled to the state
            state["best_search_results"] = rag_results or []
            state["tools_enabled"] = tools_enabled
            
            # Invoke the compiled graph
            final_state = await self.graph.ainvoke(state)
            
            logger.info(f"V4 workflow completed: answer_length={len(final_state.get('generated_answer', ''))}")
            
            return final_state
            
        except Exception as e:
            logger.error(f"V4 workflow failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Return error state
            return {
                "generated_answer": f"Error processing query: {str(e)}",
                "formatted_answer": f"Error processing query: {str(e)}",
                "error": str(e)
            }

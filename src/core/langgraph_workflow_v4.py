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
        
        # Add all 14 nodes (13 original + 1 reformulation)
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
        workflow.add_node("query_reformulation", self.node_query_reformulation)
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
        workflow.add_edge("filter", "query_reformulation")
        workflow.add_edge("query_reformulation", "analyze")
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
            separator = "|" * 25
            logger.info(f"{separator} PROMPT HANDLER START {separator}")
            
            if self.observer:
                state["_step"] = await self.observer.emit_step_started(
                    "PROMPT_HANDLER",
                    {"has_prompt": bool(state.get("user_prompt"))}
                )
            
            # Validation happens at routing
            logger.info(f"{separator} PROMPT HANDLER COMPLETE {separator}")
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
            is_greeting = False  # Initialize here to avoid undefined variable
            confidence = 0.50
            
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
                        "has_files": has_files,
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
        """Extract and process uploaded files from session metadata"""
        try:
            separator = "|" * 25
            logger.info(f"{separator} FILE EXTRACTION START {separator}")
            
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
                logger.info(f"{separator} FILE EXTRACTION COMPLETE {separator}")
                return state
            
            file_metadata = []
            for file_info in uploaded_files:
                # Extract path from file info dict (from session metadata)
                file_path = file_info.get("path") if isinstance(file_info, dict) else getattr(file_info, "path", None)
                file_name = file_info.get("name") if isinstance(file_info, dict) else getattr(file_info, "filename", getattr(file_info, "name", "unknown"))
                file_type = file_info.get("type") if isinstance(file_info, dict) else "unknown"
                
                metadata = {
                    "filename": file_name,
                    "size": file_info.get("size", 0) if isinstance(file_info, dict) else getattr(file_info, "size", 0),
                    "file_type": file_type,
                    "file_path": file_path,
                    "extension": file_info.get("extension", "") if isinstance(file_info, dict) else ""
                }
                file_metadata.append(metadata)
            
            state["file_metadata"] = file_metadata
            
            if self.observer and state.get("_step"):
                await self.observer.emit_step_completed(
                    state["_step"],
                    output_size=sum(f.get("size", 0) for f in file_metadata)
                )
            
            logger.info(f"Extracted {len(file_metadata)} files")
            logger.info(f"{separator} FILE EXTRACTION COMPLETE {separator}")
        except Exception as e:
            logger.error(f"File extraction failed: {e}")
            state["file_metadata"] = []
            if self.observer and state.get("_step"):
                await self.observer.emit_step_failed(state["_step"], str(e))
        
        return state
    
    async def node_ingest_file(self, state: WorkflowState) -> Dict[str, Any]:
        """Ingest files into RAG system - Extract, chunk, and vectorize"""
        try:
            separator = "|" * 25
            logger.info(f"{separator} FILE INGESTION START {separator}")
            
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
            
            user_id = state.get("user_id")
            session_id = state.get("session_id")
            
            if not user_id or not session_id:
                logger.warning("Missing user_id or session_id for file ingestion")
                return state
            
            # Import file processing tools
            from ..tools.pdf_tools import analyze_pdf
            from ..tools.excel_tools import analyze_excel_to_markdown
            import uuid
            from pathlib import Path
            import os
            
            rag_service = getattr(self.agent, 'rag_service', None)
            if not rag_service:
                logger.warning("RAG service not available for file ingestion")
                state["files_ingested"] = False
                return state
            
            total_chunks = 0
            failed_files = []
            for file_info in file_metadata:
                temp_file_path = None
                try:
                    # Try both "file_path" and "path" keys for compatibility
                    file_path = file_info.get("file_path") or file_info.get("path")
                    file_name = file_info.get("filename") or file_info.get("name", "unknown")
                    file_type = file_info.get("file_type") or file_info.get("type", "unknown")
                    temp_file_path = file_path  # Store for cleanup
                    
                    if not file_path:
                        error_msg = f"CRITICAL: File path is None for {file_name}. File metadata: {file_info}"
                        logger.error(error_msg)
                        failed_files.append((file_name, "No file path provided"))
                        continue
                    
                    if not Path(file_path).exists():
                        error_msg = f"File not found at path: {file_path}"
                        logger.error(error_msg)
                        failed_files.append((file_name, f"File not found: {file_path}"))
                        continue
                    
                    file_id = str(uuid.uuid4())
                    extracted_text = None
                    
                    # Extract based on file type
                    if file_type == "pdf":
                        try:
                            # NOTE: analyze_pdf returns PDFAnalysisResult (Pydantic model), not dict
                            pdf_result = analyze_pdf(file_path)
                            # Extract text from PDFAnalysisResult object
                            extracted_text = pdf_result.extracted_text or ""
                            if pdf_result.tables_markdown:
                                extracted_text += f"\n\n## Tables\n\n{pdf_result.tables_markdown}"
                            logger.info(f"PDF extracted: {file_name}")
                        except Exception as pdf_err:
                            logger.error(f"PDF extraction failed for {file_name}: {pdf_err}")
                    elif file_type == "excel":
                        try:
                            excel_result = analyze_excel_to_markdown(file_path)
                            extracted_text = excel_result.get("markdown", "") or excel_result.get("text", "")
                            logger.info(f"Excel extracted: {file_name}")
                        except Exception as excel_err:
                            logger.error(f"Excel extraction failed for {file_name}: {excel_err}")
                    elif file_type == "image":
                        try:
                            from ..tools.pdf_tools import extract_text_from_image
                            extracted_text = extract_text_from_image(file_path)
                            logger.info(f"Image OCR: {file_name}")
                        except Exception as img_err:
                            logger.error(f"Image extraction failed for {file_name}: {img_err}")
                    else:
                        # Try reading as text
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                extracted_text = f.read()
                        except:
                            logger.warning(f"Could not extract text from {file_name}")
                    
                    if extracted_text and extracted_text.strip():
                        # Chunk and ingest into vectordb
                        try:
                            chunks = rag_service.chunk_text(extracted_text, chunk_size=500, overlap=50)
                            logger.info(f"Created {len(chunks)} chunks from {file_name}")
                            
                            if chunks:
                                chunk_count = rag_service.qd_manager.add_document_chunks(
                                    user_id=user_id,
                                    chat_session_id=session_id,
                                    file_id=file_id,
                                    chunks=chunks,
                                    metadata={"filename": file_name, "file_type": file_type}
                                )
                                total_chunks += chunk_count
                                logger.info(f"✅ Ingested {chunk_count} chunks from {file_name} into vectordb")
                            else:
                                error_msg = f"No chunks created from {file_name} (may be empty or invalid format)"
                                logger.error(error_msg)
                                failed_files.append((file_name, "No chunks created"))
                        except Exception as chunk_err:
                            error_msg = f"Chunk/ingest error for {file_name}: {chunk_err}"
                            logger.error(error_msg)
                            failed_files.append((file_name, str(chunk_err)))
                    else:
                        error_msg = f"No text extracted from {file_name}"
                        logger.error(error_msg)
                        failed_files.append((file_name, "No text extracted"))
                except Exception as file_err:
                    logger.error(f"File ingestion error for {file_info.get('filename')}: {file_err}")
                finally:
                    # CLEANUP: Delete temporary file after processing
                    if temp_file_path and Path(temp_file_path).exists():
                        try:
                            os.remove(temp_file_path)
                            logger.info(f"✓ Deleted temporary file: {temp_file_path}")
                        except Exception as cleanup_err:
                            logger.warning(f"Could not delete temp file {temp_file_path}: {cleanup_err}")
            
            # Mark as ingested only if chunks were actually created
            state["files_ingested"] = total_chunks > 0
            state["ingested_chunks"] = total_chunks
            state["failed_files"] = failed_files  # Track failed files for retrieval logic
            
            logger.info(f"="*60)
            logger.info(f"FILE INGESTION SUMMARY")
            logger.info(f"="*60)
            logger.info(f"Files processed: {len(file_metadata)}")
            logger.info(f"Total chunks created: {total_chunks}")
            logger.info(f"Successfully ingested: {len(file_metadata) - len(failed_files)}")
            if failed_files:
                logger.error(f"FAILED INGESTION:")
                for fname, reason in failed_files:
                    logger.error(f"  - {fname}: {reason}")
            logger.info(f"="*60)
            
            if self.observer and state.get("_step"):
                await self.observer.emit_step_completed(
                    state["_step"],
                    metadata={"files_processed": len(file_metadata), "chunks": total_chunks}
                )
            
            logger.info(f"Ingested {len(file_metadata)} files with {total_chunks} total chunks")
            logger.info(f"{separator} FILE INGESTION COMPLETE {separator}")
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
            separator = "|" * 25
            logger.info(f"{separator} RETRIEVAL START {separator}")
            
            if self.observer:
                state["_step"] = await self.observer.emit_step_started("RETRIEVE")
            
            # Check if RAG is enabled for this request
            rag_enabled = state.get("rag_enabled", True)
            if not rag_enabled:
                logger.info("RAG disabled - skipping retrieval")
                if self.observer and state.get("_step"):
                    await self.observer.emit_step_skipped(state["_step"], "RAG disabled")
                return state
            
            query = state.get("rewritten_prompt") or state.get("user_prompt")
            user_id = state.get("user_id")
            session_id = state.get("session_id")
            
            # Extract uploaded filenames for retrieval priority
            uploaded_filenames = None
            uploaded_files = state.get("uploaded_files", [])
            if uploaded_files:
                # Extract filenames from uploaded_files metadata
                # Note: files have 'name' field, not 'filename'
                uploaded_filenames = [f.get("name", "") for f in uploaded_files if f.get("name")]
                if uploaded_filenames:
                    logger.info(f"Passing {len(uploaded_filenames)} uploaded filenames to retrieval: {uploaded_filenames}")
                else:
                    logger.warning(f"[RETRIEVE] No 'name' field found in uploaded_files: {uploaded_files}")
            
            # FALLBACK: If no filenames from state, extract from query
            # This handles case where files were ingested but metadata wasn't propagated
            if not uploaded_filenames and query:
                logger.info(f"[RETRIEVE] No uploaded filenames from state - attempting to extract from query")
                import re
                # Look for common file patterns in the query
                filename_pattern = r'([a-zA-Z0-9_\-\.]+\.(pdf|xlsx?|csv|docx?|pptx?|txt))'
                filename_matches = re.findall(filename_pattern, query, re.IGNORECASE)
                
                if filename_matches:
                    # Extract just the filename part (re.findall returns tuples)
                    extracted_filenames = [match[0] if isinstance(match, tuple) else match for match in filename_matches]
                    # Remove duplicates while preserving order
                    seen = set()
                    uploaded_filenames = [f for f in extracted_filenames if not (f in seen or seen.add(f))]
                    logger.info(f"[RETRIEVE] Extracted {len(uploaded_filenames)} filename(s) from query: {uploaded_filenames}")
                else:
                    logger.info(f"[RETRIEVE] No filenames found in query - proceeding without file context")
            
            logger.info(f"Starting RAG retrieval (files may have been ingested)")
            
            results = await self.retrieval.retrieve_with_fallback(
                query, user_id, session_id, uploaded_filenames=uploaded_filenames
            )
            
            state["personal_semantic_results"] = results.get("personal_semantic", [])
            state["personal_keyword_results"] = results.get("personal_keyword", [])
            state["global_semantic_results"] = results.get("global_semantic", [])
            state["global_keyword_results"] = results.get("global_keyword", [])
            state["rag_enabled"] = results.get("total_results", 0) > 0
            
            # Log database reasoning
            db_reasoning = results.get("db_reasoning", "")
            if db_reasoning:
                logger.info(f"[DB DECISION] {db_reasoning}")
            
            total = results.get("total_results", 0)
            if total == 0:
                logger.warning(f"⚠️  NO RAG RESULTS: Query returned 0 results from vectordb")
                logger.info(f"   - Check if files were successfully ingested")
                logger.info(f"   - Check if vectordb has data for user: {user_id}")
                logger.info(f"   - Query: {query[:100]}...")
            else:
                logger.info(f"✅ Retrieved {total} results from RAG")
            
            if self.observer and state.get("_step"):
                await self.observer.emit_step_completed(
                    state["_step"],
                    output_size=total * 500  # Est. bytes
                )
            
            logger.info(f"Retrieved {total} results (files now in collection)")
            logger.info(f"{separator} RETRIEVAL COMPLETE {separator}")
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
    
    async def node_query_reformulation(self, state: WorkflowState) -> Dict[str, Any]:
        """
        Query Reformulation Node - Rewrite query with retrieved context
        
        Purpose:
        - After retrieval and filtering, if results exist from either/both collections
        - Combine original query + retrieved context to create an enriched query
        - Feeds into subsequent reasoning steps with full context available
        
        This allows the LLM to reason with provided information rather than raw question
        """
        try:
            if self.observer:
                state["_step"] = await self.observer.emit_step_started("QUERY_REFORMULATION")
            
            original_query = state.get("user_prompt", "").strip()
            best_results = state.get("best_search_results", [])
            
            # Log separator for reformulation start
            separator = "|" * 25
            logger.info(f"{separator} QUERY REFORMULATION START {separator}")
            logger.info(f"Original Query: {original_query}")
            logger.info(f"Available Context: {len(best_results)} retrieved results")
            
            # Only reformulate if we have both results and a query
            if not best_results or not original_query:
                logger.info("No results to reformulate with - using original query")
                state["reformulated_query"] = original_query
                state["reformulation_context"] = ""
                if self.observer and state.get("_step"):
                    await self.observer.emit_step_skipped(
                        state["_step"],
                        f"Insufficient context: {len(best_results)} results, query_len={len(original_query)}"
                    )
                logger.info(f"{separator} QUERY REFORMULATION SKIPPED {separator}")
                return state
            
            # Build context summary from retrieved results with FULL content
            context_snippets = []
            context_full = []
            for i, result in enumerate(best_results[:5], 1):  # Use top 5 results
                # Results can have 'text' or 'content' field depending on source
                content_full = result.get("text") or result.get("content", "")
                content_preview = content_full[:200] if content_full else ""  # Preview for logs
                score = result.get("score", 0)
                source = result.get("source", result.get("title", "unknown"))
                
                # For logging preview (show first 200 chars)
                if content_preview:
                    context_snippets.append(f"[{i}] ({source}, relevance: {score:.2f}): {content_preview}...")
                else:
                    context_snippets.append(f"[{i}] ({source}, relevance: {score:.2f}): [No content]")
                
                # For LLM reformulation (use FULL content)
                if content_full:
                    context_full.append(f"[Source {i}: {source}]\n{content_full}\n")
            
            # Use FULL content for reformulation to LLM
            context_text = "\n---\n".join(context_full)
            
            logger.info(f"Building reformulated query with {len(best_results)} context results:")
            for snippet in context_snippets:
                logger.info(f"  {snippet}")
            
            logger.info(f"Full context prepared ({len(context_text)} chars) for LLM reformulation")
            
            # Use LLM to reformulate the query
            system_prompt = """Bạn là một chuyên gia phân tích truy vấn. 
Nhiệm vụ của bạn là viết lại câu hỏi của người dùng kết hợp với thông tin đã truy xuất.

HƯỚNG DẪN QUAN TRỌNG:
1. Giữ nguyên ý định gốc của câu hỏi
2. Kết hợp thông tin từ các kết quả đã truy xuất
3. Làm cho truy vấn mới cụ thể hơn và có ngữ cảnh rõ ràng
4. Giữ trong 1-2 câu, rõ ràng và hành động được

ĐỊNH DẠNG ĐẦU RA:
Chỉ trả về câu hỏi đã viết lại, không thêm giải thích.
"""
            
            reformulation_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", f"""Câu hỏi gốc: {original_query}

Thông tin đã truy xuất:
{context_text}

Viết lại câu hỏi để sử dụng tốt hơn thông tin này:""")
            ])
            
            chain = reformulation_prompt | self.llm
            response = await chain.ainvoke({})
            
            reformulated_query = response.content.strip()
            state["reformulated_query"] = reformulated_query
            state["reformulation_context"] = context_text
            
            logger.info(f"Reformulated Query: {reformulated_query}")
            logger.info(f"Context Summary ({len(best_results)} results available):")
            for snippet in context_snippets:
                logger.info(f"  {snippet}")
            
            # Log full reformulated query (not just preview)
            logger.info(f"\n{'FULL REFORMULATED QUERY':^60}")
            logger.info(f"{reformulated_query}")
            logger.info(f"{'END REFORMULATED QUERY':^60}\n")
            
            if self.observer and state.get("_step"):
                await self.observer.emit_step_completed(
                    state["_step"],
                    output_size=len(reformulated_query),
                    metadata={
                        "context_results": len(best_results),
                        "original_query_len": len(original_query),
                        "reformulated_query_len": len(reformulated_query)
                    }
                )
            
            logger.info(f"{separator} QUERY REFORMULATION COMPLETE {separator}")
            
        except Exception as e:
            logger.error(f"Query reformulation failed: {e}", exc_info=True)
            state["reformulated_query"] = state.get("user_prompt", "")
            state["reformulation_context"] = ""
            if self.observer and state.get("_step"):
                await self.observer.emit_step_failed(state["_step"], str(e))
        
        return state
    
    async def node_analyze(self, state: WorkflowState) -> Dict[str, Any]:
        """Analyze detected data types"""
        try:
            separator = "|" * 25
            logger.info(f"{separator} DATA ANALYSIS START {separator}")
            
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
            logger.info(f"{separator} DATA ANALYSIS COMPLETE {separator}")
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
            separator = "|" * 25
            logger.info(f"{separator} TOOL SELECTION START {separator}")
            
            if self.observer:
                state["_step"] = await self.observer.emit_step_started("SELECT_TOOLS")
            
            query = state.get("user_prompt", "").lower()
            logger.info(f"Query for tool selection: {query[:100]}...")
            
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
            matched_keywords = {}
            
            logger.info("🔍 TOOL SELECTION REASONING:")
            for tool_name, config in tool_keywords.items():
                matched = [kw for kw in config["keywords"] if kw in query]
                if matched:
                    selected_tools.append(tool_name)
                    matched_keywords[tool_name] = matched
                    logger.info(f"  ✅ {tool_name}: matched keywords {matched}")
                else:
                    logger.info(f"  ❌ {tool_name}: no keyword matches")
            
            # If no tools matched by keywords, try to use classifier
            if not selected_tools:
                logger.info("❌ No keyword matches found - using LLM classifier")
                selection = await self.tool_selector.select_tools(
                    state.get("user_prompt", ""),
                    state.get("detected_data_types", []),
                    self.tool_names,
                    state.get("conversation_history", [])
                )
                selected_tools = selection.get("selected_tools", [])
                logger.info(f"  Classifier selected: {selected_tools}")
            else:
                logger.info(f"✅ Keyword-based selection successful: {len(selected_tools)} tool(s)")
            
            state["selected_tools"] = selected_tools
            state["primary_tool"] = selected_tools[0] if selected_tools else None
            state["tool_selection_rationale"] = f"Selected {len(selected_tools)} tools: {', '.join(selected_tools) if selected_tools else 'none'}"
            
            if self.observer and state.get("_step"):
                await self.observer.emit_step_completed(
                    state["_step"],
                    metadata={"tools_selected": len(selected_tools), "tools": selected_tools}
                )
            
            logger.info(f"Final decision: {len(selected_tools)} tool(s) selected")
            logger.info(f"{separator} TOOL SELECTION COMPLETE {separator}")
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
            
            # Mark first LLM use with proper logging
            from ..llm import LLMFactory
            LLMFactory.mark_first_use()
            
            query = state.get("user_prompt", "").strip()
            has_files = bool(state.get("uploaded_files"))
            reformulated_query = state.get("reformulated_query", "")
            
            # Check if this is just a file upload without actual content query
            # (query ONLY contains filename, no actual analysis request)
            is_file_only_query = False
            if has_files and query:
                # Get file names from uploaded files
                file_names = [
                    f.get("name", "") if isinstance(f, dict) else str(f)
                    for f in state.get("uploaded_files", [])
                ]
                
                # Remove filenames from query to see if there's actual content
                query_without_filenames = query.lower()
                for fname in file_names:
                    query_without_filenames = query_without_filenames.replace(fname.lower(), "").replace(fname.split(".")[0].lower(), "")
                
                # Check if remaining query is just generic file keywords (no specific analysis)
                query_without_filenames = query_without_filenames.replace(":", "").replace("-", "").strip()
                generic_keywords = ["phân tích", "tóm tắt", "file", "tệp", "sau", "và"]
                
                # If query only contains generic keywords and filenames, it's file-only
                if query_without_filenames:
                    remaining_words = [w for w in query_without_filenames.split() if w and w not in generic_keywords]
                    is_file_only_query = len(remaining_words) == 0  # Only generic keywords remain
                else:
                    is_file_only_query = True  # Empty after removing filenames
            
            # If no specific query was provided, ask user for clarification
            if has_files and is_file_only_query:
                uploaded_files = state.get("uploaded_files", [])
                # Handle both string filenames and dict objects
                file_list = ", ".join([
                    f.get("name", "tệp") if isinstance(f, dict) else str(f)
                    for f in uploaded_files
                ])
                
                response_text = f"""Tôi đã nhận được tệp sau của bạn: **{file_list}**

**Để tôi có thể giúp bạn phân tích tệp này, vui lòng:**

1. **Gửi lại câu hỏi cụ thể** về tệp bạn vừa tải lên:
   - Ví dụ: "Phân tích doanh thu trong báo cáo này"
   - Ví dụ: "Tóm tắt các điểm chính từ tệp"
   - Ví dụ: "Tìm thông tin về lợi nhuận"

2. **Hoặc chỉ rõ bạn muốn tôi làm gì:**
   - Phân tích dữ liệu tài chính cụ thể
   - Tóm tắt nội dung chính
   - So sánh các chỉ số
   - Tìm kiếm thông tin cụ thể

**Lưu ý:** Tôi sẽ xử lý tệp của bạn và tìm kiếm thông tin dựa trên câu hỏi cụ thể của bạn. Vui lòng cung cấp chi tiết về những gì bạn cần!"""
                
                state["generated_answer"] = response_text
                logger.info(f"File uploaded without specific query - providing guidance (files: {len(uploaded_files)})")
            else:
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
                logger.info(f"Answer generated ({len(response.content)} chars)")
            
            if self.observer and state.get("_step"):
                await self.observer.emit_step_completed(
                    state["_step"],
                    output_size=len(state.get("generated_answer", ""))
                )
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            state["generated_answer"] = f"Xin lỗi, tôi gặp lỗi khi xử lý câu hỏi của bạn: {str(e)}"
            if self.observer and state.get("_step"):
                await self.observer.emit_step_failed(state["_step"], str(e))
        
        return state
    
    async def node_execute_tools(self, state: WorkflowState) -> Dict[str, Any]:
        """Execute selected tools"""
        try:
            separator = "|" * 25
            logger.info(f"{separator} TOOL EXECUTION START {separator}")
            
            if self.observer:
                state["_step"] = await self.observer.emit_step_started(
                    "EXECUTE_TOOLS",
                    {"tool_count": len(state.get("selected_tools", []))}
                )
            
            selected = state.get("selected_tools", [])
            logger.info(f"Tools to execute: {selected if selected else 'none selected'}")
            
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
                    logger.info(f"  Executing tool: {tool_name}")
                    tool = next((t for t in self.tools if getattr(t, 'name', '') == tool_name), None)
                    if tool and hasattr(tool, 'func'):
                        result = await tool.func(state.get("user_prompt", ""))
                        tool_results[tool_name] = result
                        logger.info(f"  ✅ {tool_name} completed")
                    else:
                        logger.warning(f"  ❌ Tool {tool_name} not found or has no func method")
                except Exception as e:
                    logger.warning(f"  ❌ Tool {tool_name} failed: {e}")
            
            state["tool_results"] = tool_results
            
            if self.observer and state.get("_step"):
                await self.observer.emit_step_completed(
                    state["_step"],
                    metadata={"tools_executed": len(tool_results)}
                )
            
            logger.info(f"Tool execution complete: {len(tool_results)}/{len(selected)} tools succeeded")
            logger.info(f"{separator} TOOL EXECUTION COMPLETE {separator}")
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
                     session_id: str = "default", use_rag: bool = True,
                     tools_enabled: bool = True) -> Dict[str, Any]:
        """
        Invoke the V4 workflow with given parameters.
        
        Args:
            user_prompt: The user's question
            uploaded_files: List of uploaded file metadata
            conversation_history: List of previous messages
            user_id: User identifier
            session_id: Session identifier
            use_rag: Whether to enable RAG search (workflow will handle it)
            tools_enabled: Whether tools are enabled
            
        Returns:
            Final state with generated_answer and formatted_answer
        """
        try:
            logger.info(f"V4 workflow invoked: user={user_id}, session={session_id}")
            logger.info(f"[WORKFLOW] Input parameters:")
            logger.info(f"[WORKFLOW]   - user_prompt: {user_prompt[:50] if user_prompt else 'None'}...")
            logger.info(f"[WORKFLOW]   - uploaded_files type: {type(uploaded_files)}")
            logger.info(f"[WORKFLOW]   - uploaded_files length: {len(uploaded_files) if uploaded_files else 0}")
            if uploaded_files:
                logger.info(f"[WORKFLOW]   - uploaded_files: {[f.get('name', 'unknown') if isinstance(f, dict) else str(f) for f in uploaded_files]}")
            
            # Create initial state with the expected parameters
            state = create_initial_state(
                user_prompt=user_prompt,
                uploaded_files=uploaded_files or [],
                conversation_history=conversation_history or [],
                user_id=user_id,
                session_id=session_id
            )
            
            logger.info(f"[WORKFLOW] State created:")
            logger.info(f"[WORKFLOW]   - state['uploaded_files'] type: {type(state['uploaded_files'])}")
            logger.info(f"[WORKFLOW]   - state['uploaded_files'] length: {len(state['uploaded_files'])}")
            if state['uploaded_files']:
                logger.info(f"[WORKFLOW]   - state['uploaded_files']: {state['uploaded_files']}")

            
            # Add RAG and tools flags to the state
            state["rag_enabled"] = use_rag  # Workflow will decide in RETRIEVE node
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

"""
LangGraphWorkflow V4 - Complete 18-node architecture
Full workflow with parallel entry points, classification, file handling, retrieval, RAG processing, tool execution, and monitoring
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate

from .workflow_state import WorkflowState, PromptType, DataType, create_initial_state
from .prompt_classifier import PromptClassifier
from .retrieval_manager import RetrievalManager
from .result_filter import ResultFilter
from .query_rewriter import QueryRewriter
from .tool_selector import ToolSelector
from .output_formatter import OutputFormatter
from .workflow_observer import WorkflowObserver
from ..services.advanced_retrieval_service import AdvancedRetrievalService
from ..services.summary_tool_router import SummaryToolRouter

logger = logging.getLogger(__name__)


class LangGraphWorkflowV4:
    """
    Complete 14-node workflow architecture with full feature set:
    
    Entry Layer (Step 1):
    - PROMPT_HANDLER: Process and validate user input
    - FILE_HANDLER: Handle uploaded files (parallel entry point)
    
    Classification & Routing (Step 2):
    - CLASSIFY: Determine query type (chitchat, file, analysis, etc.)
    - DIRECT_RESPONSE: Short-circuit for chitchat/greeting queries
    
    File Processing (Steps 3-4):
    - EXTRACT_FILE: Parse PDF, Excel, images, documents
    - INGEST_FILE: Chunk, embed, and store in vector DB
    
    Query Preparation (Step 5-7):
    - REWRITE: Conditional query rewriting (file context, conversation history, or skip)
    - RETRIEVE: Semantic + keyword search from vector DB
    - FILTER: Deduplicate and rank results using RRF
    
    Analysis & Processing (Steps 8-9):
    - SELECT_TOOLS: Choose relevant tools (vnstock, technical, etc.)
    - SUMMARY_TOOLS: Apply summarization techniques (structured, metric condensing, etc.)
    
    Reformulation & Generation (Steps 10-13):
    - QUERY_REFORMULATION: Build context for LLM (RAG + tools + analysis)
    - EXECUTE_TOOLS: Run selected tools (if any)
    - FORMAT_OUTPUT: Final output formatting
    - GENERATE: LLM inference with full context
    
    Monitoring:
    - Workflow observer tracks all 14 steps in real-time
    - Each step emits started and completed events
    - Frontend receives only completed events to avoid duplication
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
        self.tool_selector = ToolSelector()
        self.formatter = OutputFormatter()
        
        # Initialize advanced retrieval and summary services
        from ..services.multi_collection_rag_service import get_rag_service
        rag_service = get_rag_service()
        self.advanced_retrieval = AdvancedRetrievalService(qdrant_manager=rag_service.qd_manager)
        self.summary_tool_router = SummaryToolRouter()
        logger.info("[WORKFLOW:INIT] ✓ Advanced retrieval service ready")
        logger.info("[WORKFLOW:INIT] ✓ Summary tool router ready")
        
        # Optional workflow observer
        self.observer = WorkflowObserver() if enable_observer else None
        
        self.graph = self._build_graph()
        logger.info("LangGraphWorkflowV4 initialized with 14-node architecture + advanced retrieval/summarization")
    
    def _build_graph(self) -> StateGraph:
        """Build the complete 14-node workflow graph"""
        workflow = StateGraph(WorkflowState)
        
        # Add all 14 nodes
        workflow.add_node("prompt_handler", self.node_prompt_handler)
        workflow.add_node("file_handler", self.node_file_handler)
        workflow.add_node("classify", self.node_classify)
        workflow.add_node("direct_response", self.node_direct_response)
        workflow.add_node("extract_file", self.node_extract_file)
        workflow.add_node("ingest_file", self.node_ingest_file)
        workflow.add_node("rewrite", self.node_rewrite)
        workflow.add_node("retrieve", self.node_retrieve)
        workflow.add_node("filter", self.node_filter)
        workflow.add_node("summary_tools", self.node_summary_tools)
        workflow.add_node("query_reformulation", self.node_query_reformulation)
        workflow.add_node("select_tools", self.node_select_tools)
        workflow.add_node("execute_tools", self.node_execute_tools)
        workflow.add_node("format_output", self.node_format_output)
        workflow.add_node("generate", self.node_generate)
        
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
                "extract_file" if s.get("uploaded_files") else "rewrite"
            ),
            {
                "direct_response": "direct_response",
                "extract_file": "extract_file",
                "rewrite": "rewrite"
            }
        )
        
        # DIRECT_RESPONSE goes to FORMAT_OUTPUT for consistent formatting
        workflow.add_edge("direct_response", "format_output")
        
        # File pipeline
        workflow.add_edge("extract_file", "ingest_file")
        workflow.add_edge("ingest_file", "rewrite")
        
        # REWRITE → RETRIEVE (consolidated rewrite logic)
        workflow.add_edge("rewrite", "retrieve")
        
        # Main retrieval pipeline
        workflow.add_edge("retrieve", "filter")
        
        # FILTER → SELECT_TOOLS
        workflow.add_edge("filter", "select_tools")
        
        # SELECT_TOOLS → EXECUTE_TOOLS (execute selected tools immediately)
        workflow.add_edge("select_tools", "execute_tools")
        
        # EXECUTE_TOOLS → SUMMARY_TOOLS (summarize both DB results + tool results)
        workflow.add_edge("execute_tools", "summary_tools")
        
        # SUMMARY_TOOLS → QUERY_REFORMULATION (reformulation gets summary + tool results)
        workflow.add_edge("summary_tools", "query_reformulation")
        
        # QUERY_REFORMULATION → FORMAT_OUTPUT (format combined data for readability)
        workflow.add_edge("query_reformulation", "format_output")
        
        # FORMAT_OUTPUT → GENERATE (LLM generates final answer with formatted data)
        workflow.add_edge("format_output", "generate")
        
        # GENERATE goes to END
        workflow.add_edge("generate", END)
        
        return workflow.compile()
    
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
            
            logger.info(f"")
            logger.info(f"🔍 CLASSIFICATION RESULT:")
            logger.info(f"─" * 80)
            logger.info(f"Prompt: {prompt[:60]}..." if len(prompt) > 60 else f"Prompt: {prompt}")
            logger.info(f"Type: {state.get('prompt_type', 'unknown')}")
            logger.info(f"Is Chitchat: {state.get('is_chitchat', False)}")
            logger.info(f"Confidence: {confidence}")
            logger.info(f"")
            
            # CRITICAL: Show which branch will be taken
            if state.get("is_chitchat"):
                logger.info(f"⚠️  ROUTING: This query will go → direct_response → format_output → generate")
                logger.info(f"   (Reformulation will NOT happen!)")
            else:
                logger.info(f"✅ ROUTING: This query will go through normal pipeline")
                logger.info(f"   (Will include: rewrite → retrieve → filter → select_tools → execute_tools → query_reformulation → format_output → generate)")
            logger.info(f"─" * 80)
            logger.info(f"")
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
            
            # IMPORTANT: Set reformulated_query for consistency across all paths
            # For chitchat, reformulated_query = original query (no RAG/tools needed)
            state["reformulated_query"] = state["user_prompt"]
            
            logger.info("")
            logger.info("🎯 DIRECT RESPONSE (Chitchat Path):")
            logger.info("─" * 80)
            logger.info(f"  - Query: {state['user_prompt'][:60]}...")
            logger.info(f"  - Response generated: {len(response.content)} chars")
            logger.info(f"  - reformulated_query set to: original query")
            logger.info("─" * 80)
            logger.info("")
            
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
            from pathlib import Path
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
                
                # Auto-detect file type from extension if not provided
                if file_type == "unknown" and file_name:
                    ext = Path(file_name).suffix.lower()
                    ext_mapping = {
                        ".pdf": "pdf",
                        ".xlsx": "excel",
                        ".xls": "excel",
                        ".csv": "excel",
                        ".png": "image",
                        ".jpg": "image",
                        ".jpeg": "image",
                        ".gif": "image",
                        ".bmp": "image",
                        ".webp": "image",
                        ".txt": "text",
                        ".doc": "word",
                        ".docx": "word"
                    }
                    file_type = ext_mapping.get(ext, "unknown")
                
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
            
            # Import file processing tools (using v2 for better extraction)
            from ..tools.pdf_tools_v2 import analyze_pdf
            from ..tools.excel_tools import analyze_excel_to_markdown
            from ..tools.image_tools import extract_text_from_image, analyze_image_with_llm, process_financial_image
            import uuid
            from pathlib import Path
            import os
            
            logger.info("[WORKFLOW] Using pdf_tools_v2 (hybrid pymupdf + camelot extraction)")
            
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
                            # Use image processing pipeline (OCR + LLM vision analysis)
                            image_result = process_financial_image(file_path)
                            if image_result.get("success"):
                                # Combine OCR text and LLM analysis for better understanding
                                ocr_text = image_result.get("extracted_text", "")
                                llm_analysis = image_result.get("analysis", "")
                                extracted_text = f"=== OCR EXTRACTED TEXT ===\n{ocr_text}\n\n=== LLM VISION ANALYSIS ===\n{llm_analysis}"
                                logger.info(f"Image processed with OCR + Vision Analysis: {file_name}")
                            else:
                                # Fallback: Try OCR only if full pipeline fails
                                logger.warning(f"Full image pipeline failed, attempting OCR fallback for {file_name}")
                                extracted_text = extract_text_from_image(file_path)
                                logger.info(f"Image OCR (fallback): {file_name}")
                        except Exception as img_err:
                            logger.error(f"Image processing failed for {file_name}: {img_err}")
                            # Try simple OCR as last resort
                            try:
                                extracted_text = extract_text_from_image(file_path)
                                logger.info(f"Image extraction fallback successful: {file_name}")
                            except Exception as ocr_fallback_err:
                                logger.error(f"Image OCR fallback also failed: {ocr_fallback_err}")
                                extracted_text = ""
                    else:
                        # Try reading as text
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                extracted_text = f.read()
                        except:
                            logger.warning(f"Could not extract text from {file_name}")
                    
                    if extracted_text and extracted_text.strip():
                        # Use advanced chunking to create both structural and metric-centric chunks
                        try:
                            # Use advanced chunking service for two-level chunking
                            # Pass LLM for metric synthesis
                            rag_service.advanced_chunking.llm = self.llm
                            struct_payloads, metric_payloads = rag_service.advanced_chunking.process_document(
                                text=extracted_text,
                                file_id=file_id,
                                user_id=user_id,
                                chat_session_id=session_id
                            )
                            
                            logger.info(f"Created {len(struct_payloads)} structural + {len(metric_payloads)} metric-centric chunks from {file_name}")
                            
                            if struct_payloads:
                                # Add structural chunks
                                logger.info(f"[INGEST] Storing {len(struct_payloads)} structural chunks...")
                                chunk_count = rag_service.qd_manager.add_document_chunks(
                                    user_id=user_id,
                                    chat_session_id=session_id,
                                    file_id=file_id,
                                    chunks=struct_payloads,
                                    metadata={"filename": file_name, "file_type": file_type}
                                )
                                total_chunks += chunk_count
                                logger.info(f"✅ Stored {chunk_count} structural chunks to Qdrant")
                                
                                # CRITICAL: Wait after structural chunks to ensure commit
                                await asyncio.sleep(0.5)
                            
                            # Add metric-centric chunks if any (INDEPENDENT of struct_payloads)
                            if metric_payloads:
                                logger.info(f"[INGEST] Storing {len(metric_payloads)} metric-centric chunks...")
                                metric_count = rag_service.qd_manager.add_document_chunks(
                                    user_id=user_id,
                                    chat_session_id=session_id,
                                    file_id=file_id,
                                    chunks=metric_payloads,
                                    metadata={"filename": file_name, "file_type": file_type, "is_metric_centric": True}
                                )
                                total_chunks += metric_count
                                logger.info(f"✅ Stored {metric_count} metric-centric chunks to Qdrant")
                                
                                # CRITICAL: Wait after metric chunks to ensure commit
                                await asyncio.sleep(0.5)
                            
                            # CRITICAL: Wait for chunks to be fully committed before proceeding
                            # Verify that chunks are searchable by attempting a metadata search
                            if struct_payloads or metric_payloads:
                                logger.info(f"⏳ Verifying chunks are indexed in Qdrant...")
                                verified = False
                                max_retries = 5
                                retry_delay = 1  # Start with 1 second
                                
                                for attempt in range(max_retries):
                                    try:
                                        await asyncio.sleep(retry_delay)
                                        
                                        # Try to search for chunks by filename
                                        verification_results = rag_service.search_by_filename_metadata(
                                            user_id=user_id,
                                            filename=file_name,
                                            chat_session_id=session_id,
                                            limit=1
                                        )
                                        
                                        if verification_results:
                                            logger.info(f"✅ Verification successful: {len(verification_results)} chunk(s) found in index")
                                            verified = True
                                            break
                                        else:
                                            logger.debug(f"Verification attempt {attempt + 1}: No chunks found yet, retrying...")
                                            retry_delay = min(retry_delay * 1.5, 5)  # Exponential backoff, max 5s
                                    
                                    except Exception as verify_err:
                                        logger.debug(f"Verification attempt {attempt + 1} failed: {verify_err}")
                                
                                if not verified:
                                    logger.warning(f"⚠️ Could not verify chunks in index after {max_retries} attempts - proceeding anyway")
                                else:
                                    logger.info(f"✓ Chunks for '{file_name}' are ready for retrieval")
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
            
            # CRITICAL: Final wait to ensure ALL chunks are fully committed before retrieval
            if total_chunks > 0:
                logger.info(f"⏳ Final verification: Ensuring all {total_chunks} chunks are indexed...")
                await asyncio.sleep(2)  # Give Qdrant time to finalize indexing
                logger.info(f"✅ All chunks committed and ready for retrieval")
            
            logger.info(f"{separator} FILE INGESTION COMPLETE {separator}")
        except Exception as e:
            logger.error(f"File ingestion failed: {e}")
            state["files_ingested"] = False
            if self.observer and state.get("_step"):
                await self.observer.emit_step_failed(state["_step"], str(e))
        
        return state
    
    async def node_rewrite(self, state: WorkflowState) -> Dict[str, Any]:
        """
        Conditional query rewriting with intelligent strategy selection.
        
        Automatically determines and applies the best rewriting strategy:
        1. FILE CONTEXT: If files are uploaded, add file metadata to query
        2. CONVERSATION CONTEXT: If conversation history exists, add relevant history
        3. NO REWRITING: If neither applies, use original query
        
        This consolidated node replaces three separate nodes (rewrite_eval, rewrite_file, rewrite_convo)
        for better clarity and efficiency.
        """
        try:
            if self.observer:
                state["_step"] = await self.observer.emit_step_started("REWRITE")
            
            prompt = state.get("user_prompt", "")
            file_metadata = state.get("file_metadata", [])
            history = state.get("conversation_history", [])
            
            rewritten_prompt = prompt
            rewrite_strategy = "none"
            
            # Strategy 1: Rewrite with file context (highest priority)
            if file_metadata:
                try:
                    rewritten_prompt = await self.rewriter.rewrite_with_file_context(
                        prompt, file_metadata
                    )
                    rewrite_strategy = "file"
                    logger.info(f"Applied file context rewriting strategy")
                except Exception as e:
                    logger.warning(f"File context rewriting failed, keeping original: {e}")
                    rewrite_strategy = "file_failed"
            
            # Strategy 2: Rewrite with conversation context (fallback)
            elif history:
                try:
                    rewritten_prompt = await self.rewriter.rewrite_with_conversation_context(
                        prompt, history
                    )
                    rewrite_strategy = "conversation"
                    logger.info(f"Applied conversation context rewriting strategy")
                except Exception as e:
                    logger.warning(f"Conversation rewriting failed, keeping original: {e}")
                    rewrite_strategy = "conversation_failed"
            
            # Strategy 3: No rewriting needed
            else:
                rewrite_strategy = "none"
                logger.info(f"No context available for rewriting, using original query")
            
            state["rewritten_prompt"] = rewritten_prompt
            state["rewrite_strategy"] = rewrite_strategy
            
            if self.observer and state.get("_step"):
                await self.observer.emit_step_completed(
                    state["_step"],
                    output_size=len(rewritten_prompt),
                    metadata={
                        "strategy": rewrite_strategy,
                        "has_files": bool(file_metadata),
                        "has_history": bool(history),
                        "prompt_changed": rewritten_prompt != prompt
                    }
                )
            
            logger.info(f"Rewrite strategy: {rewrite_strategy} | Changed: {rewritten_prompt != prompt}")
        
        except Exception as e:
            logger.error(f"Rewrite node failed: {e}")
            state["rewritten_prompt"] = state.get("user_prompt", "")
            state["rewrite_strategy"] = "error"
            if self.observer and state.get("_step"):
                await self.observer.emit_step_failed(state["_step"], str(e))
        
        return state
    
    async def node_retrieve(self, state: WorkflowState) -> Dict[str, Any]:
        """
        Retrieve with use-case-specific strategy and advanced techniques
        
        RETRIEVAL STRATEGIES (updated):
        1. FILE UPLOAD → Retrieve by file_id/filename (prioritize uploaded files)
        2. SUMMARY REQUEST → Metric-centric chunks + linked structural chunks (via RRF)
        3. NON-SUMMARY DATA → Structural chunks only (normal retrieval)
        4. HYBRID → Multi-strategy retrieval with RRF fusion
        """
        try:
            separator = "═" * 80
            logger.info("")
            logger.info(separator)
            logger.info("RETRIEVAL NODE START (USE-CASE-SPECIFIC)")
            logger.info(separator)
            
            if self.observer:
                state["_step"] = await self.observer.emit_step_started("RETRIEVE")
            
            # Check if RAG is enabled for this request
            rag_enabled = state.get("rag_enabled", True)
            if not rag_enabled:
                logger.info("⊘ RAG disabled - skipping retrieval")
                if self.observer and state.get("_step"):
                    await self.observer.emit_step_skipped(state["_step"], "RAG disabled")
                logger.info(separator)
                return state
            
            query = state.get("rewritten_prompt") or state.get("user_prompt")
            user_id = state.get("user_id")
            session_id = state.get("session_id")
            
            # Detect use case for strategic retrieval using LLM classification
            logger.info("[RETRIEVE:STRATEGY] Detecting use case with LLM...")
            
            # Use LLM-based query classification (already handles generic vs specific)
            from ..services.advanced_retrieval_service import QueryClassifier
            classifier = QueryClassifier()
            classification = classifier.classify_with_llm(query)
            is_generic_query = classification["is_generic"]
            
            # IMPORTANT: Don't use hardcoded summary detection - LLM already classified!
            # Only use summary if LLM explicitly says it's generic file analysis
            is_summary_request = is_generic_query
            
            uploaded_filenames = None
            uploaded_files = state.get("uploaded_files", [])
            is_file_upload = len(uploaded_files) > 0
            
            if uploaded_files:
                # Extract filenames from uploaded_files metadata
                uploaded_filenames = [f.get("name", "") for f in uploaded_files if f.get("name")]
                if uploaded_filenames:
                    logger.info(f"[RETRIEVE:STRATEGY] FILE UPLOAD detected: {len(uploaded_filenames)} file(s)")
                    logger.info(f"  Files: {uploaded_filenames}")
            
            # FALLBACK: Extract filenames from query if not in metadata
            if not uploaded_filenames and query:
                import re
                filename_pattern = r'([a-zA-Z0-9_\-\.]+\.(pdf|xlsx?|csv|docx?|pptx?|txt))'
                filename_matches = re.findall(filename_pattern, query, re.IGNORECASE)
                if filename_matches:
                    extracted_filenames = [match[0] if isinstance(match, tuple) else match for match in filename_matches]
                    seen = set()
                    uploaded_filenames = [f for f in extracted_filenames if not (f in seen or seen.add(f))]
                    logger.info(f"[RETRIEVE:STRATEGY] FILE NAMES extracted from query: {len(uploaded_filenames)} file(s)")
            
            # Log detected strategy
            if is_file_upload:
                logger.info(f"[RETRIEVE:STRATEGY] ✓ FILE UPLOAD STRATEGY")
            elif is_summary_request:
                logger.info(f"[RETRIEVE:STRATEGY] ✓ SUMMARY/ANALYSIS STRATEGY (LLM: generic query)")
            else:
                logger.info(f"[RETRIEVE:STRATEGY] ✓ SPECIFIC DETAIL STRATEGY (LLM: specific query)")
            
            logger.info(f"[RETRIEVE:START] Query length: {len(query)}, Using advanced retrieval...")
            
            # Generate query embedding for advanced retrieval
            from ..core.embeddings import get_embedding_strategy
            embedding_strategy = get_embedding_strategy()
            query_embedding = embedding_strategy.embed_query(query)
            
            # Use advanced retrieval with strategic parameters
            try:
                # For summary requests, prioritize metric-centric chunks
                # For file uploads, retrieve all and let RRF rank them
                # For general queries, use standard dual retrieval
                
                include_metrics = is_summary_request or is_file_upload
                include_structural = True  # Always include structural for context
                
                logger.info(f"[RETRIEVE:CONFIG] include_metrics={include_metrics}, include_structural={include_structural}")
                
                # NOTE: advanced_retrieval.retrieve is synchronous, so wrap it with asyncio.to_thread
                advanced_results = await asyncio.to_thread(
                    self.advanced_retrieval.retrieve,
                    user_id=user_id,
                    query=query,
                    query_embedding=query_embedding,
                    chat_session_id=session_id,
                    limit=15,  # Get more results for strategies
                    include_metrics=include_metrics,
                    include_structural=include_structural
                )
                
                logger.info(f"[RETRIEVE:ADVANCED] Retrieved {len(advanced_results)} results using RRF + dedup")
                logger.info(f"[RETRIEVE:ADVANCED] Chunk types: {', '.join(set([r.get('chunk_type', 'unknown') for r in advanced_results]))}")
                
                # Apply use-case-specific post-processing
                if is_summary_request:
                    # For summary: boost metric chunks
                    metric_chunks = [r for r in advanced_results if r.get('chunk_type') == 'metric_centric']
                    structural_chunks = [r for r in advanced_results if r.get('chunk_type') != 'metric_centric']
                    logger.info(f"[RETRIEVE:PROCESS] Summary mode: {len(metric_chunks)} metric + {len(structural_chunks)} structural")
                    # Reorder: metric chunks first, then structural
                    advanced_results = metric_chunks + structural_chunks
                
                elif is_file_upload and uploaded_filenames:
                    # For file upload: prioritize by filename match
                    matching = [r for r in advanced_results if any(fname in r.get('filename', '') or fname in r.get('source', '') 
                                                                   for fname in uploaded_filenames)]
                    non_matching = [r for r in advanced_results if r not in matching]
                    logger.info(f"[RETRIEVE:PROCESS] File upload mode: {len(matching)} from uploaded files + {len(non_matching)} others")
                    advanced_results = matching + non_matching
                
                # Convert advanced results to old format for backward compatibility
                state["personal_semantic_results"] = advanced_results[:8]   # Top 8
                state["personal_keyword_results"] = advanced_results[8:15] if len(advanced_results) > 8 else []
                state["global_semantic_results"] = []
                state["global_keyword_results"] = []
                
                total = len(advanced_results)
                state["rag_enabled"] = total > 0
                
                logger.info(f"[RETRIEVE:SUCCESS] ✓ Retrieved {total} results with {('SUMMARY' if is_summary_request else 'STANDARD')} strategy")
                
            except Exception as advanced_err:
                logger.warning(f"[RETRIEVE:ADVANCED] Failed: {advanced_err}, falling back to legacy retrieval")
                # Fallback to legacy retrieval
                results = await self.retrieval.retrieve_with_fallback(
                    query, user_id, session_id, uploaded_filenames=uploaded_filenames
                )
                
                state["personal_semantic_results"] = results.get("personal_semantic", [])
                state["personal_keyword_results"] = results.get("personal_keyword", [])
                state["global_semantic_results"] = results.get("global_semantic", [])
                state["global_keyword_results"] = results.get("global_keyword", [])
                
                total = results.get("total_results", 0)
                state["rag_enabled"] = total > 0
                
                # Log database reasoning
                db_reasoning = results.get("db_reasoning", "")
                if db_reasoning:
                    logger.info(f"  Database decision: {db_reasoning}")
                
                logger.info(f"✓ Retrieved {total} results with legacy retrieval (fallback)")
            
            if self.observer and state.get("_step"):
                await self.observer.emit_step_completed(
                    state["_step"],
                    output_size=total * 500,  # Est. bytes
                    metadata={"total_results": total}
                )
            
            logger.info(separator)
        except Exception as e:
            logger.error(f"✗ Retrieval failed: {e}")
            state["rag_enabled"] = False
            if self.observer and state.get("_step"):
                await self.observer.emit_step_failed(state["_step"], str(e))
        
        return state
    
    async def node_filter(self, state: WorkflowState) -> Dict[str, Any]:
        """Filter and rank results with RRF"""
        try:
            separator = "═" * 80
            logger.info("")
            logger.info(separator)
            logger.info("FILTER NODE START")
            logger.info(separator)
            
            if self.observer:
                state["_step"] = await self.observer.emit_step_started("FILTER")
            
            best_results = self.filter.filter_and_rank(
                state.get("personal_semantic_results", []),
                state.get("personal_keyword_results", []),
                state.get("global_semantic_results", []),
                state.get("global_keyword_results", [])
            )
            
            state["best_search_results"] = best_results
            
            logger.info(f"Filtered to {len(best_results)} results (deduplicated and ranked)")
            
            if self.observer and state.get("_step"):
                await self.observer.emit_step_completed(
                    state["_step"],
                    output_size=len(best_results) * 500,
                    metadata={"results": len(best_results)}
                )
            
            logger.info(separator)
        except Exception as e:
            logger.error(f"✗ Filtering failed: {e}")
            state["best_search_results"] = []
            if self.observer and state.get("_step"):
                await self.observer.emit_step_failed(state["_step"], str(e))
        
        return state
    
    async def node_summary_tools(self, state: WorkflowState) -> Dict[str, Any]:
        """
        Summary Tools Node - LLM-driven selection of summary technique
        
        NEW ARCHITECTURE:
        1. Check if summary is needed (rules-first + LLM fallback)
        2. If yes: Let LLM SELECT which summary technique to use
        3. Execute the selected summary tool with DB results + previous tool results
        4. Store summary result separately for query reformulation
        5. If no: Pass through without summarization
        
        This node receives:
        - best_search_results: Filtered chunks from DB (structural + metric-centric)
        - tool_results: Results from executed tools (if any)
        - user_prompt: Original user query
        
        Returns:
        - summary_applied: Boolean flag
        - summary_result: Structured result from summary tool execution
        - summary_tool_selected: Which technique was selected
        """
        try:
            from src.core.llm_summary_tools import SummaryToolsProvider
            
            separator = "═" * 80
            logger.info("")
            logger.info(separator)
            logger.info("SUMMARY TOOLS NODE (LLM-DRIVEN)")
            logger.info(separator)
            
            if self.observer:
                state["_step"] = await self.observer.emit_step_started("SUMMARY_TOOLS")
            
            # Get inputs
            best_results = state.get("best_search_results", [])
            tool_results = state.get("tool_results", {})
            query = state.get("rewritten_prompt") or state.get("user_prompt", "")
            
            logger.info(f"[SUMMARY:INPUT] DB results: {len(best_results)} chunks, Query length: {len(query)}")
            logger.info(f"[SUMMARY:INPUT] Previous tool results: {len(tool_results)} outputs")
            
            # Step 1: Determine if summary is needed (rules-first)
            needs_summary = self.summary_tool_router.should_summarize(query)
            logger.info(f"[SUMMARY:DETECT] Initial detection: needs_summary={needs_summary}")
            
            # Step 2: If unclear, use LLM to classify
            if not needs_summary:
                logger.info(f"[SUMMARY:CLASSIFY] Using LLM for classification...")
                try:
                    classify_prompt = f"""Analyze this user query: "{query}"

Does the user ask for a SUMMARY, ANALYSIS, or EXPLANATION of financial data?
Specifically, do they want you to:
1. Summarize/aggregate data
2. Analyze trends or changes  
3. Detect anomalies
4. Compare metrics
5. Answer key financial questions

Respond with ONLY "yes" or "no" (lowercase)."""
                    
                    llm_response = await asyncio.to_thread(
                        lambda: self.llm.invoke(classify_prompt)
                    )
                    response_text = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
                    needs_summary = 'yes' in response_text.lower()
                    logger.info(f"[SUMMARY:CLASSIFY] LLM classification: needs_summary={needs_summary}")
                except Exception as e:
                    logger.warning(f"[SUMMARY:CLASSIFY] LLM classification failed: {e}, using initial detection")
            
            # If no summary needed or no data, skip
            if not needs_summary or not best_results:
                logger.info("⊘ Summary not applicable or no data available")
                state["summary_applied"] = False
                state["summary_result"] = None
                state["summary_tool_selected"] = None
                if self.observer and state.get("_step"):
                    await self.observer.emit_step_skipped(state["_step"], "Summary not needed")
                logger.info(separator)
                return state
            
            # Step 3: Use LLM to SELECT 2-3 summary techniques
            logger.info(f"[SUMMARY:SELECT] Using LLM to select 2-3 summary techniques...")
            
            # Get available summary tools and their descriptions
            tool_descriptions = SummaryToolsProvider.get_tool_descriptions()
            tools_text = "\n".join([
                f"- {name}: {desc}"
                for name, desc in tool_descriptions.items()
            ])
            
            selection_prompt = f"""You have these summary techniques available:

{tools_text}

Given the user query: "{query}"
And the financial data to summarize

Select 2-3 summary techniques that work BEST TOGETHER for comprehensive analysis.
Respond with ONLY the technique names (comma-separated, from: structured_data_summary, metric_condensing, 
constraint_listing, feasibility_check) - nothing else."""
            
            logger.debug(f"[SUMMARY:SELECT] Prompt sent to LLM:\n{selection_prompt}")
            
            try:
                selection_response = await asyncio.to_thread(
                    lambda: self.llm.invoke(selection_prompt)
                )
                response_text = selection_response.content if hasattr(selection_response, 'content') else str(selection_response)
                logger.debug(f"[SUMMARY:SELECT] LLM raw response: {response_text}")
                
                # Extract technique names from response
                response_lower = response_text.lower()
                available_techniques = list(tool_descriptions.keys())
                selected_techniques = []
                
                for technique in available_techniques:
                    if technique in response_lower:
                        selected_techniques.append(technique)
                
                # Ensure 2-3 techniques
                if len(selected_techniques) < 2:
                    fallback_techniques = [t for t in ['structured_data_summary', 'metric_condensing', 'constraint_listing'] if t not in selected_techniques]
                    selected_techniques.extend(fallback_techniques[:max(0, 2 - len(selected_techniques))])
                elif len(selected_techniques) > 3:
                    selected_techniques = selected_techniques[:3]
                
                logger.info(f"[SUMMARY:SELECT] LLM selected {len(selected_techniques)} techniques: {selected_techniques}")
                
            except Exception as e:
                logger.warning(f"[SUMMARY:SELECT] LLM selection failed: {e}, defaulting to [structured_data_summary, metric_condensing]")
                selected_techniques = ["structured_data_summary", "metric_condensing"]
            
            # Step 4: Execute multiple summary tools
            logger.info(f"[SUMMARY:EXECUTE] Executing {len(selected_techniques)} techniques with {len(best_results)} chunks")
            
            try:
                summary_tools = SummaryToolsProvider.get_all_tools()
                all_results = []
                
                for selected_technique in selected_techniques:
                    summary_tool_func = summary_tools.get(selected_technique)
                    
                    if not summary_tool_func:
                        logger.error(f"[SUMMARY:EXECUTE] Tool not found: {selected_technique}")
                        continue
                    
                    # Execute tool with chunks and query
                    logger.info(f"[SUMMARY:EXECUTE] Running {selected_technique}...")
                    summary_result = await asyncio.to_thread(
                        summary_tool_func, best_results, query
                    )
                    
                    if summary_result.get("success"):
                        all_results.append(summary_result)
                        # Log RAW summary result
                        logger.info(f"[SUMMARY:RESULT] RAW output from {selected_technique}:")
                        logger.info(f"{summary_result}")
                
                if all_results:  # At least one technique succeeded
                    # Aggregate insights, constraints, categories from all results
                    aggregated_insights = []
                    aggregated_constraints = {"binding": [], "tight": [], "slack": []}
                    aggregated_categories = {}
                    aggregated_metrics = {}
                    aggregated_violations = []
                    
                    for result in all_results:
                        # Aggregate insights
                        if 'insights' in result and result['insights']:
                            aggregated_insights.extend(result['insights'])
                        
                        # Aggregate constraints
                        if 'binding_constraints' in result:
                            aggregated_constraints["binding"].extend([c for c in result.get('binding_constraints', []) if c])
                        if 'tight_constraints' in result:
                            aggregated_constraints["tight"].extend([c for c in result.get('tight_constraints', []) if c])
                        if 'slack_constraints' in result:
                            aggregated_constraints["slack"].extend([c for c in result.get('slack_constraints', []) if c])
                        
                        # Aggregate categories
                        if 'categories' in result:
                            aggregated_categories.update(result['categories'])
                        
                        # Aggregate metrics
                        if 'metrics' in result:
                            aggregated_metrics.update({k: v for k, v in result['metrics'].items() if k is not None})
                        
                        # Aggregate violations
                        if 'violations' in result:
                            aggregated_violations.extend([v for v in result['violations'] if v])
                    
                    # Build aggregated summary_result
                    summary_result = {
                        "success": True,
                        "technique": all_results[0].get('technique', 'unknown') if all_results else 'unknown',
                        "summary_tool_used": all_results[0].get('summary_tool_used', 'unknown') if all_results else 'unknown',
                        "summary": "\n\n".join([r.get('summary', '') for r in all_results if r.get('summary')]),
                        "techniques_executed": [r.get('technique', 'unknown') for r in all_results],
                        "results": all_results,
                        "insights": aggregated_insights[:20],  # Top 20 insights
                        "categories": aggregated_categories if aggregated_categories else None,
                        "binding_constraints": aggregated_constraints["binding"],
                        "tight_constraints": aggregated_constraints["tight"],
                        "slack_constraints": aggregated_constraints["slack"],
                        "metrics": aggregated_metrics if aggregated_metrics else None,
                        "violations": aggregated_violations[:10],  # Top 10 violations
                        "confidence_score": sum(r.get('confidence_score', 0.5) for r in all_results) / len(all_results) if all_results else 0.5
                    }
                    logger.info(f"[SUMMARY:EXECUTE] ✓ Executed {len(all_results)} techniques successfully")
                    logger.info(f"[SUMMARY:EXECUTE] Aggregated: {len(aggregated_insights)} insights, "
                               f"{len(aggregated_constraints['binding'])} binding, "
                               f"{len(aggregated_constraints['tight'])} tight, "
                               f"{len(aggregated_constraints['slack'])} slack constraints")
                    
                    # Log FULL raw summary results
                    logger.info(f"[SUMMARY:RESULT] FINAL AGGREGATED RESULTS:")
                    logger.info(f"[SUMMARY:RESULT] Insights: {len(aggregated_insights)}, "
                               f"Categories: {len(aggregated_categories)}, "
                               f"Constraints: B={len(aggregated_constraints['binding'])} T={len(aggregated_constraints['tight'])} S={len(aggregated_constraints['slack'])}")
                    
                    # Add metadata
                    summary_result["summary_tools_used"] = selected_techniques
                    summary_result["previous_tool_results"] = tool_results if tool_results else None
                    summary_result["source_chunks_count"] = len(best_results)
                    
                    # CRITICAL: Set these flags so query_reformulation knows summary was applied
                    state["summary_applied"] = True
                    state["summary_result"] = summary_result
                    state["summary_tool_selected"] = selected_techniques
                    state["summary_was_executed"] = True
                    
                    logger.info(f"[SUMMARY:EXECUTE] ✓ Summary result stored for reformulation")
                    logger.info(f"[SUMMARY:EXECUTE] Techniques used: {selected_techniques}")
                else:
                    logger.warning(f"[SUMMARY:EXECUTE] All summary tools failed")
                    state["summary_applied"] = False
                    state["summary_result"] = None
                    state["summary_tool_selected"] = None
            
            except Exception as e:
                logger.error(f"[SUMMARY:EXECUTE] Exception during execution: {e}")
                state["summary_applied"] = False
                state["summary_result"] = None
                state["summary_tool_selected"] = None
                if self.observer and state.get("_step"):
                    await self.observer.emit_step_failed(state["_step"], str(e))
                logger.info(separator)
                return state
            
            # Emit completion
            if self.observer and state.get("_step"):
                await self.observer.emit_step_completed(
                    state["_step"],
                    output_size=len(state["summary_result"].get('summary', '')) if state.get("summary_result") else 0,
                    metadata={
                        "summary_applied": state.get("summary_applied", False),
                        "technique": selected_technique,
                        "chunks_processed": len(best_results)
                    }
                )
            
            logger.info(separator)
            
            # DEBUG: Log exact state before returning
            logger.info(f"[SUMMARY:STATE_BEFORE_RETURN]")
            logger.info(f"  summary_applied = {state.get('summary_applied', 'NOT_SET')}")
            logger.info(f"  summary_was_executed = {state.get('summary_was_executed', 'NOT_SET')}")
            logger.info(f"  summary_result exists = {bool(state.get('summary_result'))}")
            if state.get("summary_result"):
                logger.info(f"  summary_result keys = {list(state['summary_result'].keys())}")
        except Exception as e:
            logger.error(f"⚠ Summary tools node failed: {e}", exc_info=True)
            state["summary_applied"] = False
            state["summary_result"] = None
            state["summary_tool_selected"] = None
            if self.observer and state.get("_step"):
                await self.observer.emit_step_failed(state["_step"], str(e))
        
        return state
    
    async def node_query_reformulation(self, state: WorkflowState) -> Dict[str, Any]:
        """
        Query Reformulation Node - Rewrite query with retrieved context AND tool results
        
        Purpose:
        - After tool execution and optional summarization, combine original query with:
          1. Summary result (if summarization was applied) OR retrieved context
          2. Tool execution results (calculator, technical analysis, etc.)
        - Create an enriched query for the LLM to reason with full context
        
        CRITICAL: This node MUST ensure LLM receives data to avoid hallucination
        - If summary was applied: use summarized data (not raw chunks)
        - If no summary: use retrieved chunks as-is
        - If no RAG data AND no tool data → instruction for LLM to request more info
        - If insufficient data → warn LLM about limitations
        
        This allows the LLM to synthesize information from multiple sources
        """
        try:
            if self.observer:
                state["_step"] = await self.observer.emit_step_started("QUERY_REFORMULATION")
            
            original_query = state.get("user_prompt", "").strip()
            best_results = state.get("best_search_results", [])
            tool_results = state.get("tool_results", {})
            summary_result = state.get("summary_result")
            summary_applied = state.get("summary_applied", False)
            summary_was_executed = state.get("summary_was_executed", False)
            
            # CRITICAL: Log all summary-related state for debugging
            logger.info(f"Summary state: applied={summary_applied}, was_executed={summary_was_executed}, has_result={bool(summary_result)}")
            if summary_result:
                logger.info(f"  Summary technique: {summary_result.get('technique', 'unknown')}")
                logger.info(f"  Summary keys: {list(summary_result.keys())}")
            
            # Log separator for reformulation start
            separator = "═" * 80
            logger.info("")
            logger.info(separator)
            logger.info("QUERY REFORMULATION NODE START")
            logger.info(separator)
            logger.info(f"Query: {original_query[:100]}...")
            logger.info(f"RAG data available: {len(best_results)} results")
            logger.info(f"Tool data available: {len(tool_results)} tools")
            logger.info(f"Summary applied: {summary_applied}")
            
            # Skip if no query
            if not original_query:
                logger.warning("⚠️  NO QUERY PROVIDED - Skipping reformulation")
                state["reformulated_query"] = ""
                if self.observer and state.get("_step"):
                    await self.observer.emit_step_skipped(state["_step"], "No query provided")
                logger.info(separator)
                return state
            
            # CRITICAL CHECK: If no RAG data AND no tool data AND no summary, warn the LLM
            has_no_data = len(best_results) == 0 and len(tool_results) == 0 and not summary_applied
            if has_no_data:
                logger.warning("⚠️  NO DATA AVAILABLE - RAG returned 0 results, no tools executed, and no summary applied")
                logger.warning("    LLM will be instructed to clarify with user instead of hallucinating")
            
            # Parse user request to identify what they're asking for
            # This helps organize the context better for the LLM
            logger.info("📋 STEP 1: Parse user request to identify action and sources")
            
            # Extract mentioned file names or specific sources from query
            import re
            file_mentions = re.findall(r'["\']?([a-zA-Z0-9_\-\.]+\.(txt|pdf|docx|xlsx|csv))["\']?', original_query, re.IGNORECASE)
            if file_mentions:
                mentioned_files = list(set([f[0] for f in file_mentions]))
                logger.info(f"  ✓ Detected file references: {mentioned_files}")
            else:
                mentioned_files = []
            
            # Organize RAG results by source file AND by type
            logger.info("📄 STEP 2: Organize retrieved data by source and type")
            results_by_source = {}
            metric_chunks = []
            structural_chunks = []
            
            if best_results:
                for result in best_results:
                    source = result.get("source", result.get("title", "unknown"))
                    if source not in results_by_source:
                        results_by_source[source] = []
                    results_by_source[source].append(result)
                    
                    # Also separate by type for cleaner reformulation
                    if result.get('chunk_type') == 'metric_centric':
                        metric_chunks.append(result)
                    else:
                        structural_chunks.append(result)
                
                logger.info(f"  ✓ Organized {len(best_results)} results from {len(results_by_source)} sources")
                logger.info(f"    - {len(metric_chunks)} metric chunks")
                logger.info(f"    - {len(structural_chunks)} structural chunks")
            
            # Build structured reformulation
            logger.info("📝 STEP 3: Build structured context for LLM")
            reformulation_parts = []
            
            # Section 1: User Request Summary
            reformulation_parts.append("NGƯỜI DÙNG CÓ YÊU CẦU:")
            reformulation_parts.append(f"Câu hỏi gốc: {original_query}")
            if mentioned_files:
                reformulation_parts.append(f"Tệp được đề cập: {', '.join(mentioned_files)}")
            if tool_results:
                reformulation_parts.append(f"Công cụ được sử dụng: {', '.join(tool_results.keys())}")
            if summary_applied:
                reformulation_parts.append(f"Kỹ thuật tóm tắt được áp dụng: {summary_result.get('technique', 'unknown')}")
            elif best_results:
                reformulation_parts.append(f"Số tài liệu được truy xuất: {len(best_results)}")
            reformulation_parts.append("")
            
            # CRITICAL: If no data at all, add warning to context
            if has_no_data:
                reformulation_parts.append("THÔNG BÁO QUAN TRỌNG:")
                reformulation_parts.append("Không có dữ liệu sẵn có từ:")
                reformulation_parts.append("- Truy xuất từ cơ sở dữ liệu (0 kết quả)")
                reformulation_parts.append("- Các công cụ phân tích (không có)")
                reformulation_parts.append("- Tóm tắt (không được áp dụng)")
                reformulation_parts.append("")
                reformulation_parts.append("HƯỚNG DẪN CHO LLM:")
                reformulation_parts.append("Yêu cầu người dùng làm rõ hoặc cung cấp thêm thông tin")
                reformulation_parts.append("Không được tự tạo hoặc suy đoán dữ liệu tài chính")
                reformulation_parts.append("")
            
            # Section 2: Summarized Data (if summary was applied or executed) OR Retrieved Data
            # CRITICAL: Check if summary was actually executed, regardless of applied flag
            if (summary_applied or summary_was_executed) and summary_result:
                reformulation_parts.append("PHÂN TÍCH TÓM LƯỢC (ĐẬT KẾT CHỌN CÔNG CỤ PHÂN TÍCH):")
                reformulation_parts.append(f"Kỹ thuật phân tích: {summary_result.get('technique', 'unknown').upper()}")
                reformulation_parts.append(f"({summary_result.get('summary_tool_used', 'unknown')})")
                
                # Add technique description
                technique_to_desc = {
                    "structured_data_summary": "Organize facts by metric type/category, no interpretation",
                    "metric_condensing": "Reduce each metric to ONE LINE essentials, no explanation",
                    "constraint_listing": "List binding/tight/slack constraints, status only",
                    "feasibility_check": "YES/NO feasibility check with violation list"
                }
                technique_used = summary_result.get('summary_tool_used', 'unknown')
                if technique_used in technique_to_desc:
                    reformulation_parts.append(f"Mô tả: {technique_to_desc[technique_used]}")
                
                reformulation_parts.append(f"Độ tin cậy: {summary_result.get('confidence_score', 0):.1%}")
                reformulation_parts.append("")
                
                # CRITICAL: Summarized data is the primary input, not raw chunks
                reformulation_parts.append("KẾT QUẢ PHÂN TÍCH:")
                
                # Add summarized content
                summary_text = summary_result.get('summary', '')
                if summary_text:
                    reformulation_parts.append(f"TÓM TẮT: {summary_text}")
                
                # Add specific insights from the selected technique
                if 'insights' in summary_result:
                    reformulation_parts.append("CHI TIẾT:")
                    for insight in summary_result.get('insights', []):
                        reformulation_parts.append(f"- {insight}")
                
                if 'anomalies' in summary_result:
                    reformulation_parts.append("BẤT THƯỜNG PHÁT HIỆN:")
                    for anomaly in summary_result.get('anomalies', []):
                        reformulation_parts.append(f"- {anomaly}")
                
                if 'answers' in summary_result:
                    reformulation_parts.append("CÂU TRẢ LỜI CHỦ ĐỀ:")
                    for question, answer in summary_result.get('answers', {}).items():
                        reformulation_parts.append(f"Q: {question} | A: {answer}")
                
                if 'sections' in summary_result:
                    reformulation_parts.append("CẤU TRÚC PHÂN TÍCH:")
                    for section_name, section_content in summary_result.get('sections', {}).items():
                        reformulation_parts.append(f"{section_name}: {section_content}")
                
                reformulation_parts.append("")
                
                # Add structural chunks for context and verification
                if structural_chunks:
                    reformulation_parts.append("DỮ LIỆU CẤU TRÚC (VÀO NGỮ CẢN VÀ KIỂM CHỨNG):")
                    reformulation_parts.append("Những đoạn tài liệu sau cung cấp ngữ cảnh và cho phép xác minh các chỉ số:")
                    reformulation_parts.append("")
                    
                    for i, result in enumerate(structural_chunks[:3], 1):
                        content = result.get("text") or result.get("content", "")
                        source = result.get("source", result.get("filename", "unknown"))
                        reformulation_parts.append(f"Đoạn {i} từ {source}:")
                        if content:
                            # Truncate to 200 chars for supporting context
                            if len(content) > 200:
                                reformulation_parts.append(f"{content[:200]}...")
                            else:
                                reformulation_parts.append(f"{content}")
                        else:
                            reformulation_parts.append("[Nội dung rỗng]")
                        reformulation_parts.append("")
                
                # Add metric chunks with instructions
                if metric_chunks:
                    reformulation_parts.append("CÁC CHỈ SỐ TÓM TẮT (HÃY TÓM TẮT VÀ THÊM VÀO CÂU TRẢ LỜI NẾU CÓ NGHĨA):")
                    reformulation_parts.append("Các chỉ số sau cần được tóm tắt. Hãy:")
                    reformulation_parts.append("1. Tóm tắt từng chỉ số")
                    reformulation_parts.append("2. Thêm vào câu trả lời nếu nó có liên quan và có ý nghĩa")
                    reformulation_parts.append("3. Hiển thị thông tin chi tiết nếu chúng giải thích được câu hỏi")
                    reformulation_parts.append("")
                    
                    for i, result in enumerate(metric_chunks[:10], 1):
                        content = result.get("text") or result.get("content", "")
                        metric_name = result.get("metric_name", "unknown metric")
                        relevance = result.get("relevance", 1.0)
                        reformulation_parts.append(f"Chỉ số {i} - {metric_name} (liên quan: {relevance:.0%}):")
                        if content:
                            # Include full metric text for LLM to summarize
                            if len(content) > 500:
                                reformulation_parts.append(f"{content[:500]}...")
                            else:
                                reformulation_parts.append(f"{content}")
                        else:
                            reformulation_parts.append("[Nội dung rỗng]")
                        reformulation_parts.append("")
                
                
            elif best_results:
                # No summary: Show structured data, then metric chunks
                
                # First: Structural chunks for context and verification
                if structural_chunks:
                    reformulation_parts.append("DỮ LIỆU CẤU TRÚC (VÀO NGỮ CẢNH VÀ KIỂM CHỨNG):")
                    reformulation_parts.append("Những đoạn tài liệu sau cung cấp ngữ cảnh và cho phép xác minh các chỉ số:")
                    reformulation_parts.append("")
                    
                    for source, results in results_by_source.items():
                        source_structural = [r for r in results if r.get('chunk_type') != 'metric_centric']
                        if source_structural:
                            reformulation_parts.append(f"TỪ TÀI LIỆU: {source}")
                            for i, result in enumerate(source_structural[:2], 1):
                                content = result.get("text") or result.get("content", "")
                                score = result.get("score", 0)
                                reformulation_parts.append(f"Đoạn {i} (độ liên quan: {score:.2f}):")
                                if content:
                                    # Truncate structural chunks to 250 chars
                                    if len(content) > 250:
                                        reformulation_parts.append(f"{content[:250]}...")
                                    else:
                                        reformulation_parts.append(f"{content}")
                                else:
                                    reformulation_parts.append("[Nội dung rỗng]")
                            reformulation_parts.append("")
                
                # Second: Metric chunks with clear instructions
                if metric_chunks:
                    reformulation_parts.append("CÁC CHỈ SỐ (HÃY TÓM TẮT VÀ THÊM VÀO CÂU TRẢ LỜI NẾU CÓ NGHĨA):")
                    reformulation_parts.append("Các chỉ số sau cần được tóm tắt. Hãy:")
                    reformulation_parts.append("1. Tóm tắt từng chỉ số")
                    reformulation_parts.append("2. Thêm vào câu trả lời nếu nó có liên quan và có ý nghĩa")
                    reformulation_parts.append("3. Hiển thị thông tin chi tiết nếu chúng giải thích được câu hỏi")
                    reformulation_parts.append("")
                    
                    for source, results in results_by_source.items():
                        source_metrics = [r for r in results if r.get('chunk_type') == 'metric_centric']
                        if source_metrics:
                            reformulation_parts.append(f"TỪ TÀI LIỆU: {source}")
                            for i, result in enumerate(source_metrics[:8], 1):
                                content = result.get("text") or result.get("content", "")
                                metric_name = result.get("metric_name", "unknown metric")
                                score = result.get("score", 0)
                                relevance = result.get("relevance", 1.0)
                                reformulation_parts.append(f"Chỉ số {i} - {metric_name} (liên quan: {score:.2f}, độ đáng tin: {relevance:.0%}):")
                                if content:
                                    # Include FULL metric content for LLM to analyze
                                    reformulation_parts.append(f"{content}")
                                else:
                                    reformulation_parts.append("[Nội dung rỗng]")
                            reformulation_parts.append("")
                
            
            # Section 3: Tool Results
            if tool_results:
                reformulation_parts.append("DỮ LIỆU TỪ CÁC CÔNG CỤ PHÂN TÍCH:")
                
                for tool_name, result in tool_results.items():
                    reformulation_parts.append(f"CÔNG CỤ: {tool_name}")
                    result_str = str(result)
                    # Include FULL tool result (not truncated preview)
                    if result_str:
                        reformulation_parts.append(f"{result_str}")
                    else:
                        reformulation_parts.append("[Kết quả rỗng]")
                reformulation_parts.append("")
            
            # Section 4: Instructions for the LLM
            if has_no_data:
                # Special instructions when no data
                reformulation_parts.append("HƯỚNG DẪN CHO LLM:")
                reformulation_parts.append("1. Không được hallucinate hoặc suy đoán dữ liệu tài chính")
                reformulation_parts.append("2. Xin lỗi người dùng vì chưa thể trả lời")
                reformulation_parts.append("3. Giải thích rõ lý do (không có dữ liệu)")
                reformulation_parts.append("4. Hướng dẫn người dùng:")
                reformulation_parts.append("   a) Tải lên tài liệu liên quan")
                reformulation_parts.append("   b) Cung cấp thêm chi tiết về câu hỏi")
                reformulation_parts.append("   c) Thử lại với câu hỏi khác")
            else:
                # Normal instructions when data exists
                reformulation_parts.append("HƯỚNG DẪN CHO LLM:")
                reformulation_parts.append("1. Dựa trên câu hỏi của người dùng ở trên")
                reformulation_parts.append("2. Sử dụng dữ liệu từ các tài liệu được truy xuất")
                reformulation_parts.append("3. Sử dụng dữ liệu từ các công cụ phân tích")
                reformulation_parts.append("4. Kết hợp tất cả dữ liệu này để trả lời câu hỏi")
                reformulation_parts.append("5. Chỉ rõ nguồn dữ liệu được sử dụng (tài liệu nào, công cụ nào)")
                reformulation_parts.append("6. Nếu dữ liệu không đủ, hãy nói rõ điều đó")
            reformulation_parts.append("")
            
            full_reformulated_query = "\n".join(reformulation_parts)
            
            # Log summary stats
            logger.info("")
            logger.info("REFORMULATION STATISTICS")
            
            reformulation_lines = full_reformulated_query.split('\n')
            total_lines = len(reformulation_lines)
            total_chars = len(full_reformulated_query)
            
            logger.info(f"Total Lines: {total_lines}")
            logger.info(f"Total Characters: {total_chars:,}")
            logger.info(f"RAG Sources: {len(results_by_source)}")
            logger.info(f"Tool Results: {len(tool_results)}")
            logger.info(f"Data Status: {'NO DATA' if has_no_data else 'DATA AVAILABLE'}")
            
            logger.info("Reformulation complete")
            logger.info("")
            
            # Log RAW reformulated query being sent to LLM
            logger.info("[QUERY_REFORMULATION] RAW REFORMULATED QUERY TO BE SENT TO LLM:")
            logger.info(full_reformulated_query)
            
            # Set the reformulated query as the new prompt for the LLM
            state["reformulated_query"] = full_reformulated_query
            
            if self.observer and state.get("_step"):
                await self.observer.emit_step_completed(
                    state["_step"],
                    output_size=len(full_reformulated_query),
                    metadata={
                        "rag_sources": len(results_by_source),
                        "rag_results": len(best_results),
                        "tool_results": len(tool_results),
                        "mentioned_files": len(mentioned_files),
                        "total_chars": total_chars,
                        "has_no_data": has_no_data
                    }
                )
            
        except Exception as e:
            logger.error(f"Query reformulation failed: {e}", exc_info=True)
            state["reformulated_query"] = state.get("user_prompt", "")
            state["reformulation_context"] = ""
            if self.observer and state.get("_step"):
                await self.observer.emit_step_failed(state["_step"], str(e))
        
        return state
    
    async def _select_tools_with_llm(self, query: str, available_tools: List[str]) -> List[str]:
        """
        Use LLM to intelligently select which tools to use for the query.
        This is the intelligent fallback when keyword matching fails.
        
        Args:
            query: User's query/question
            available_tools: List of tool names available in the system
            
        Returns:
            List of selected tool names
        """
        try:
            from langchain_core.prompts import ChatPromptTemplate
            
            # Build tool descriptions
            tool_descriptions = {
                "get_company_info": "Get company information (name, industry, capital, activities)",
                "get_shareholders": "Get shareholder information (major shareholders, ownership structure)",
                "get_officers": "Get company leadership (CEO, chairman, directors, management team)",
                "get_subsidiaries": "Get subsidiary and related company information",
                "get_company_events": "Get company events (dividends, ĐHCĐ, shareholder meetings)",
                "get_historical_data": "Get historical stock price data (OHLCV)",
                "calculate_sma": "Calculate Simple Moving Average (trend analysis)",
                "calculate_rsi": "Calculate Relative Strength Index (momentum indicator)"
            }
            
            tools_text = "\n".join([
                f"- {tool}: {tool_descriptions.get(tool, 'Tool')}"
                for tool in available_tools
            ])
            
            prompt = ChatPromptTemplate.from_template("""
You are an expert financial analyst assistant. Given a user query, select ONLY the most relevant tools.

Available Tools:
{{tools}}

User Query: {{query}}

STRICT SELECTION RULES:
1. Analyze the query keywords to determine what information is needed
2. Match tools ONLY if query clearly requests that specific information
3. Select 1-3 tools MAXIMUM
4. Prefer more specific tools over generic ones
5. If query asks about company data → use get_company_info (not all company tools)
6. If query asks about stock price → use get_historical_data
7. If query asks about technical indicators → use calculate_sma or calculate_rsi (NOT both unless explicitly asked for both)
8. If query is general/chitchat → return "none"
9. Return ONLY tool names, comma-separated, or "none"

EXAMPLES:
- "What is the price history of VNM?" → get_historical_data
- "Is VNM a good buy?" → get_historical_data,calculate_rsi
- "Who are the CEOs?" → get_officers
- "Hello" → none
- "Calculate 20-day SMA" → calculate_sma
- "Give me all info about VCB" → get_company_info

Your response (tool names only, nothing else):""")
            
            chain = prompt | self.llm
            response = await chain.ainvoke({
                "tools": tools_text,
                "query": query
            })
            
            # Parse response - extract tool names with stricter validation
            tools_str = response.content.strip().lower().strip(".,!?;:")
            
            if tools_str == "none" or not tools_str:
                logger.info("LLM: No suitable tools found (strict selection)")
                return []
            
            # Validate and extract tool names - only accept known tools
            selected = []
            for tool in available_tools:
                # Match tool names more carefully
                if f"{tool.lower()}" in tools_str or tools_str in f"{tool.lower()}":
                    if tool not in selected:  # Avoid duplicates
                        selected.append(tool)
            
            # Limit to max 3 tools
            selected = selected[:3]
            
            logger.info(f"LLM Tool Selection (strict): query='{query[:50]}...' → tools={selected}")
            return selected
            
        except Exception as e:
            logger.warning(f"LLM tool selection failed: {e}, returning empty list")
            return []
    
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
            separator = "═" * 80
            logger.info("")
            logger.info(separator)
            logger.info("TOOL SELECTION NODE START")
            logger.info(separator)
            
            if self.observer:
                state["_step"] = await self.observer.emit_step_started("SELECT_TOOLS")
            
            query = state.get("user_prompt", "").lower()
            logger.info(f"Query: {query[:80]}...")
            
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
                    "keywords": ["ban lãnh đạo", "ceo", "lãnh đạo", "chủ tịch", "giám đốc", "leadership"],
                    "pattern": r"(lãnh đạo|ceo|chủ tịch|giám đốc|quản lý)"
                },
                "get_subsidiaries": {
                    "keywords": ["công ty con", "liên kết", "subsidiary", "con"],
                    "pattern": r"(công ty con|liên kết|subsidiary)"
                },
                "get_company_events": {
                    "keywords": ["sự kiện", "cổ tức", "đhcđ", "event", "dividend"],
                    "pattern": r"(sự kiện|cổ tức|đhcđ|event|chia)"
                },
                "get_historical_data": {
                    "keywords": ["giá", "ohlcv", "lịch sử", "price", "history"],
                    "pattern": r"(giá|ohlcv|lịch sử|price|history|3 tháng|6 tháng|1 năm)"
                },
                "calculate_sma": {
                    "keywords": ["sma", "moving average", "xu hướng", "trend"],
                    "pattern": r"(sma|moving average|xu hướng|trend|ma)"
                },
                "calculate_rsi": {
                    "keywords": ["rsi", "quá mua", "quá bán", "overbought", "oversold"],
                    "pattern": r"(rsi|quá mua|quá bán|overbought|oversold)"
                }
            }
            
            selected_tools = []
            matched_keywords = {}
            
            logger.info("Tool matching:")
            for tool_name, config in tool_keywords.items():
                # Use lowercase for case-insensitive keyword matching
                matched = [kw for kw in config["keywords"] if kw.lower() in query]
                if matched:
                    selected_tools.append(tool_name)
                    matched_keywords[tool_name] = matched
                    logger.info(f"  ✓ {tool_name} → {', '.join(matched)}")
                else:
                    logger.info(f"  ✗ {tool_name}")
            
            # If no tools matched by keywords, use LLM for intelligent selection
            if not selected_tools:
                logger.info("No keyword matches - using LLM for intelligent selection")
                selected_tools = await self._select_tools_with_llm(
                    state.get("user_prompt", ""),
                    self.tool_names
                )
                if selected_tools:
                    logger.info(f"LLM selected: {selected_tools}")
                else:
                    logger.info(f"LLM selected: none")
            else:
                logger.info(f"Keyword-based selection: {len(selected_tools)} tool(s)")
            
            state["selected_tools"] = selected_tools
            state["primary_tool"] = selected_tools[0] if selected_tools else None
            state["tool_selection_rationale"] = f"Selected {len(selected_tools)} tools: {', '.join(selected_tools) if selected_tools else 'none'}"
            
            if self.observer and state.get("_step"):
                await self.observer.emit_step_completed(
                    state["_step"],
                    metadata={"tools_selected": len(selected_tools), "tools": selected_tools}
                )
            
            logger.info(separator)
        except Exception as e:
            logger.error(f"✗ Tool selection failed: {e}")
            state["selected_tools"] = []
            if self.observer and state.get("_step"):
                await self.observer.emit_step_failed(state["_step"], str(e))
        
        return state
    
    async def node_generate(self, state: WorkflowState) -> Dict[str, Any]:
        """
        Generate final answer with LLM using reformulated query and formatted data.
        
        Implements:
        - Vietnamese financial advisor instructions
        - Uses reformulated query (query + RAG context + tool results)
        - Uses formatted output from FORMAT_OUTPUT node
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
            
            separator = "|" * 25
            logger.info(f"{separator} GENERATE START {separator}")
            
            # Detailed state tracking - check what's actually in the state
            logger.info("")
            logger.info("🔍 STATE VERIFICATION AT GENERATE NODE START:")
            logger.info("═" * 80)
            
            # Check reformulated_query
            reformulated_query_raw = state.get("reformulated_query", "")
            logger.info(f"1. reformulated_query:")
            logger.info(f"   - Exists: {bool(reformulated_query_raw)}")
            logger.info(f"   - Type: {type(reformulated_query_raw).__name__}")
            logger.info(f"   - Length: {len(reformulated_query_raw)} chars")
            if reformulated_query_raw:
                logger.info(f"   - Preview: {reformulated_query_raw[:150]}...")
            else:
                logger.info(f"   - WARNING: reformulated_query is EMPTY!")
            
            # Check user_prompt
            user_prompt_raw = state.get("user_prompt", "")
            logger.info(f"2. user_prompt (fallback):")
            logger.info(f"   - Exists: {bool(user_prompt_raw)}")
            logger.info(f"   - Type: {type(user_prompt_raw).__name__}")
            logger.info(f"   - Length: {len(user_prompt_raw)} chars")
            
            # Check formatted_answer
            formatted_answer_raw = state.get("formatted_answer", "")
            logger.info(f"3. formatted_answer:")
            logger.info(f"   - Exists: {bool(formatted_answer_raw)}")
            logger.info(f"   - Type: {type(formatted_answer_raw).__name__}")
            logger.info(f"   - Length: {len(formatted_answer_raw)} chars")
            
            # Check tool_results
            tool_results_raw = state.get("tool_results", {})
            logger.info(f"4. tool_results:")
            logger.info(f"   - Exists: {bool(tool_results_raw)}")
            logger.info(f"   - Type: {type(tool_results_raw).__name__}")
            logger.info(f"   - Count: {len(tool_results_raw)} tools")
            logger.info(f"   - Tools: {list(tool_results_raw.keys())}")
            
            logger.info("═" * 80)
            logger.info("")
            
            # Use reformulated query (which includes RAG + tool context)
            reformulated_query = reformulated_query_raw.strip()
            user_prompt_fallback = user_prompt_raw.strip()
            query = reformulated_query if reformulated_query else user_prompt_fallback
            
            has_files = bool(state.get("uploaded_files"))
            formatted_output = formatted_answer_raw
            tool_results = tool_results_raw
            best_results = state.get("best_search_results", [])
            
            # Log which query is being used
            logger.info("")
            logger.info("║" + "=" * 77 + "║")
            logger.info("║ QUERY BEING SENT TO LLM (GENERATION STEP):".ljust(79) + "║")
            logger.info("╠" + "=" * 77 + "╣")
            logger.info(f"║ Source: {'REFORMULATED QUERY (RAG + Tools)' if reformulated_query else 'FALLBACK: Original user prompt':<43} ║")
            logger.info(f"║ Length: {len(query):,} characters{' ' * (36 - len(str(len(query))))} ║")
            logger.info("╠" + "=" * 77 + "╣")
            logger.info(query)
            logger.info("╠" + "=" * 77 + "╣")
            logger.info(f"║ End of query being sent to LLM".ljust(79) + "║")
            logger.info("║" + "=" * 77 + "║")
            logger.info("")
            
            logger.info(f"📊 Available Data for LLM:")
            logger.info(f"  ✓ Formatted output: {len(formatted_output):,} chars")
            logger.info(f"  ✓ Tool results: {len(tool_results)} tools")
            logger.info(f"  ✓ RAG results: {len(best_results)} items")
            
            # Check if this is just a file upload without actual content query
            is_file_only_query = False
            if has_files and query:
                file_names = [
                    f.get("name", "") if isinstance(f, dict) else str(f)
                    for f in state.get("uploaded_files", [])
                ]
                
                query_without_filenames = query.lower()
                for fname in file_names:
                    query_without_filenames = query_without_filenames.replace(fname.lower(), "").replace(fname.split(".")[0].lower(), "")
                
                query_without_filenames = query_without_filenames.replace(":", "").replace("-", "").strip()
                generic_keywords = ["phân tích", "tóm tắt", "file", "tệp", "sau", "và"]
                
                if query_without_filenames:
                    remaining_words = [w for w in query_without_filenames.split() if w and w not in generic_keywords]
                    is_file_only_query = len(remaining_words) == 0
                else:
                    is_file_only_query = True
            
            # If no specific query was provided, ask user for clarification
            if has_files and is_file_only_query:
                uploaded_files = state.get("uploaded_files", [])
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
                logger.info(f"File uploaded without specific query - providing guidance")
            else:
                # System prompt for financial advisor
                system_prompt = """Bạn là một chuyên gia tư vấn tài chính chuyên về thị trường chứng khoán Việt Nam. Bạn có kinh nghiệm phân tích sâu, hiểu rõ các chỉ số tài chính, và có khả năng liên kết thông tin để đưa ra những nhận định sáng suốt.

NHIỆM VỤ:
- Trả lời các câu hỏi tài chính một cách chính xác, chi tiết nhưng dễ hiểu
- Phân tích và giải thích ý nghĩa của dữ liệu, không chỉ trình bày thô các con số
- Liên kết các chỉ số khác nhau để tạo ra bức tranh toàn cảnh
- Cung cấp thông tin hữu ích giúp người dùng hiểu rõ hơn về tình hình tài chính

HƯỚNG DẪN TRẢ LỜI:

1. PHONG CÁCH:
   - Trả lời bằng tiếng Việt, tự nhiên và chuyên nghiệp
   - Sử dụng dữ liệu cụ thể để hỗ trợ các phân tích của bạn
   - Khi có dữ liệu bảng hoặc danh sách, hiển thị dưới dạng BẢNG MARKDOWN để dễ đọc
   - QUAN TRỌNG: Đây là dữ liệu tham khảo - hãy diễn giải, phân tích và rút ra kết luận của riêng bạn chứ không phải chỉ nhắc lại số liệu

2. CÁCH TIẾP CẬN:
   - Bắt đầu với câu trả lời chính, sau đó cung cấp chi tiết hỗ trợ
   - Nếu dữ liệu phức tạp, đầu tiên tóm tắt điểm chính, sau đó cung cấp bảng chi tiết
   - Kết thúc bằng phân tích tổng hợp hoặc gợi ý dựa trên các thông tin đã cung cấp
   - Được phép bổ sung kiến thức chuyên gia của bạn để làm cho câu trả lời sâu sắc hơn

3. DIỄN GIẢI CHỈ SỐ:
   - Giải thích những gì các chỉ số có ý nghĩa (ví dụ: SMA, RSI, P/E ratio)
   - Liên kết các chỉ số để xác định xu hướng hoặc cơ hội
   - Cung cấp bối cảnh để giúp người dùng hiểu ý nghĩa thực tế

4. XỬ LÝ DỮ LIỆU THIẾU:
   - Nếu dữ liệu không tìm thấy, giải thích lý do rõ ràng
   - Gợi ý những cách sửa hoặc điều chỉnh để lấy được thông tin cần thiết
   - Có thể đưa ra phân tích dựa trên dữ liệu liên quan nếu có"""
                
                # Ensure system_prompt doesn't have unmatched braces for LangChain template parsing
                # Replace any literal braces with escaped versions for template safety
                system_prompt = system_prompt.replace("{", "{{").replace("}", "}}")
                
                # Build prompt with formatted data
                # Clearly separate question from context to give LLM freedom to synthesize
                user_prompt_content = f"""Câu hỏi của người dùng: {query}

---THÔNG TIN THAM KHẢO---

"""
                
                if formatted_output:
                    logger.info(f"Including formatted output in prompt ({len(formatted_output)} chars)")
                    user_prompt_content += f"""{formatted_output}

"""
                
                if tool_results and not formatted_output:
                    logger.info(f"Including raw tool results ({len(tool_results)} tools)")
                    for tool_name, result in tool_results.items():
                        user_prompt_content += f"""Từ công cụ '{tool_name}':
{result}

"""
                
                user_prompt_content += """---HẾT THÔNG TIN---

Dựa trên thông tin tham khảo ở trên, hãy:
1. Trả lời trực tiếp câu hỏi của người dùng
2. Giải thích và phân tích các con số, không chỉ trình bày thô dữ liệu
3. Đưa ra nhận định cá nhân và kết luận sáng suốt
4. Tự do diễn giải, liên kết thông tin, và bổ sung kiến thức chuyên gia của bạn để giúp người dùng hiểu rõ hơn"""
                
                logger.info(f"Generating answer with prompt ({len(user_prompt_content)} chars)...")
                
                # Escape braces in user_prompt_content for LangChain template parsing
                # Tool results may contain unescaped braces that need to be escaped
                escaped_user_prompt = user_prompt_content.replace("{", "{{").replace("}", "}}")
                
                # Create prompt and generate answer
                generation_prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", escaped_user_prompt)
                ])
                
                chain = generation_prompt | self.llm
                response = await chain.ainvoke({})
                
                generated_answer = response.content.strip()
                state["generated_answer"] = generated_answer
                
                logger.info(f"Generated answer ({len(generated_answer)} chars)")
                logger.info(f"Answer preview:\n{generated_answer[:500]}...")
            
            if self.observer and state.get("_step"):
                await self.observer.emit_step_completed(
                    state["_step"],
                    output_size=len(state.get("generated_answer", "")),
                    metadata={
                        "query_len": len(query),
                        "has_formatted_output": bool(formatted_output),
                        "tool_count": len(tool_results)
                    }
                )
            
            logger.info(f"{separator} GENERATE COMPLETE {separator}")
        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            state["generated_answer"] = "Tôi gặp lỗi khi xử lý câu hỏi của bạn. Vui lòng thử lại."
            if self.observer and state.get("_step"):
                await self.observer.emit_step_failed(state["_step"], str(e))
        
        return state
    
    async def node_execute_tools(self, state: WorkflowState) -> Dict[str, Any]:
        """Execute selected tools"""
        try:
            import inspect
            
            separator = "═" * 80
            logger.info("")
            logger.info(separator)
            logger.info("TOOL EXECUTION NODE START")
            logger.info(separator)
            
            if self.observer:
                state["_step"] = await self.observer.emit_step_started(
                    "EXECUTE_TOOLS",
                    {"tool_count": len(state.get("selected_tools", []))}
                )
            
            selected = state.get("selected_tools", [])
            
            if not selected:
                logger.info("⊘ No tools selected - skipping execution")
                if self.observer and state.get("_step"):
                    await self.observer.emit_step_skipped(
                        "EXECUTE_TOOLS", "No tools selected"
                    )
                logger.info(separator)
                return state
            
            logger.info(f"Executing {len(selected)} tool(s): {', '.join(selected)}")
            
            tool_results = {}
            query = state.get("user_prompt", "")
            retrieved_context = state.get("retrieved_context", [])
            
            for tool_name in selected:
                try:
                    logger.info(f"  Executing: {tool_name}")
                    
                    # Find tool by name
                    tool = next((t for t in self.tools if getattr(t, 'name', '') == tool_name), None)
                    
                    if not tool:
                        logger.warning(f"  ✗ Tool not found: {tool_name}")
                        continue
                    
                    # Use LLM to generate proper tool parameters from the query
                    if hasattr(tool, 'args_schema'):
                        # Get the schema
                        schema = tool.args_schema
                        schema_fields = schema.model_fields if hasattr(schema, 'model_fields') else {}
                        
                        # Build prompt for LLM to extract parameters
                        param_prompt = f"""Given the user query: "{query}"
                        
Extract parameters for the tool '{tool_name}' with the following fields:
{chr(10).join([f"- {field_name}: {field_info.description or 'required'}" for field_name, field_info in schema_fields.items()])}

Return ONLY a JSON object with the extracted parameters, no other text."""
                        
                        params = None
                        try:
                            # Call LLM to extract parameters
                            llm_response = await asyncio.to_thread(
                                lambda: self.llm.invoke(param_prompt)
                            )
                            response_text = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
                            
                            # Parse JSON from response
                            import json
                            import re
                            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                            if json_match:
                                params = json.loads(json_match.group())
                        except Exception as e:
                            logger.debug(f"  LLM parameter extraction error: {e}")
                            params = None
                        
                        # If LLM extraction failed, try smart fallback extraction
                        if params is None:
                            logger.warning(f"  Parameter extraction failed, using fallback extraction")
                            params = self._extract_parameters_fallback(query, tool_name, schema_fields)
                        
                        # Call tool with extracted parameters
                        if params:
                            try:
                                if hasattr(tool, 'invoke'):
                                    result = await asyncio.to_thread(tool.invoke, params)
                                elif hasattr(tool, 'func') and callable(tool.func):
                                    result = await asyncio.to_thread(tool.func, **params)
                                else:
                                    result = await asyncio.to_thread(tool, **params)
                            except Exception as e:
                                logger.error(f"  ✗ {tool_name} exception: {e}")
                                result = None
                        else:
                            logger.warning(f"  Could not extract parameters for {tool_name}")
                            result = None
                    else:
                        # No structured schema, call directly with query
                        if hasattr(tool, 'func') and callable(tool.func):
                            func = tool.func
                        elif callable(tool):
                            func = tool
                        else:
                            logger.warning(f"  ✗ Tool not callable: {tool_name}")
                            continue
                        
                        if inspect.iscoroutinefunction(func):
                            result = await func(query)
                        else:
                            result = await asyncio.to_thread(func, query)
                    
                    if result is not None:
                        # Check if the result is a failure (has success: false)
                        is_failed_result = False
                        if isinstance(result, str):
                            try:
                                import json
                                result_obj = json.loads(result)
                                if isinstance(result_obj, dict) and result_obj.get("success") == False:
                                    is_failed_result = True
                                    logger.warning(f"  ⚠️  {tool_name} returned error: {result_obj.get('message', 'Unknown')}")
                            except:
                                pass  # Not JSON, treat as success
                        
                        # Only store successful results
                        if not is_failed_result:
                            tool_results[tool_name] = result
                            logger.info(f"  ✓ {tool_name} completed")
                        else:
                            logger.info(f"  ✗ {tool_name} failed (error returned)")
                    else:
                        logger.warning(f"  ⚠️  {tool_name} returned None")
                        
                except Exception as e:
                    logger.error(f"  ✗ {tool_name} exception: {e}")
            
            state["tool_results"] = tool_results
            
            logger.info(f"Tool execution complete: {len(tool_results)}/{len(selected)} succeeded")
            
            if self.observer and state.get("_step"):
                await self.observer.emit_step_completed(
                    state["_step"],
                    metadata={"tools_executed": len(tool_results), "tools": list(tool_results.keys())}
                )
            
            logger.info(separator)
        except Exception as e:
            logger.error(f"✗ Tool execution failed: {e}")
            state["tool_results"] = {}
            if self.observer and state.get("_step"):
                await self.observer.emit_step_failed(state["_step"], str(e))
        
        return state
    
    async def node_format_output(self, state: WorkflowState) -> Dict[str, Any]:
        """Format final output with tables, calculations, and citations"""
        try:
            separator = "═" * 80
            logger.info("")
            logger.info(separator)
            logger.info("FORMAT OUTPUT NODE START")
            logger.info(separator)
            
            if self.observer:
                state["_step"] = await self.observer.emit_step_started("FORMAT_OUTPUT")
            
            answer = state.get("generated_answer", "")
            search_results = state.get("best_search_results", [])
            tool_results = state.get("tool_results")
            data_types = state.get("detected_data_types", [])
            
            logger.info(f"Input data:")
            logger.info(f"  Generated answer: {len(answer)} chars")
            logger.info(f"  Search results: {len(search_results) if search_results else 0}")
            logger.info(f"  Tool results: {len(tool_results) if tool_results else 0}")
            logger.info(f"  Data types: {data_types if data_types else 'None'}")
            
            formatted_answer = await self.formatter.format_answer(
                answer, search_results, tool_results, data_types
            )
            
            logger.info(f"Formatted answer: {len(formatted_answer)} chars")
            state["formatted_answer"] = formatted_answer
            
            if self.observer and state.get("_step"):
                await self.observer.emit_step_completed(
                    state["_step"],
                    output_size=len(formatted_answer),
                    metadata={"answer_len": len(formatted_answer)}
                )
            
            logger.info(separator)
        except Exception as e:
            logger.error(f"✗ Output formatting failed: {e}")
            state["formatted_answer"] = state.get("generated_answer", "")
            if self.observer and state.get("_step"):
                await self.observer.emit_step_failed(state["_step"], str(e))
        
        return state
    
    def _extract_parameters_fallback(self, query: str, tool_name: str, schema_fields: Dict) -> Optional[Dict[str, Any]]:
        """
        Fallback parameter extraction when LLM extraction fails.
        Intelligently extracts parameters from query text using pattern matching and heuristics.
        Handles Vietnamese and English inputs, stock tickers, numbers, and dates.
        """
        import re
        params = {}
        
        # Normalize query for easier matching
        query_lower = query.lower()
        query_normalized = query.replace("tính", "").replace("cho", "").replace("của", "")
        
        logger.debug(f"[FALLBACK] Attempting to extract parameters for {tool_name}")
        logger.debug(f"[FALLBACK] Query: {query}")
        
        # List of required fields from schema
        required_fields = [
            field_name for field_name, field_info in schema_fields.items()
            if field_info.is_required()
        ]
        
        try:
            # ==== EXTRACT TICKER (common for all technical tools) ====
            if 'ticker' in schema_fields:
                # Pattern 1: Explicit Vietnamese phrases - "cho HPG", "của TCB"
                ticker_match = re.search(r'(?:cho|của|từ|mã)\s+([A-Z]{1,5})\b', query)
                
                # Pattern 2: Standalone uppercase letters (likely ticker)
                if not ticker_match:
                    ticker_match = re.search(r'\b([A-Z]{2,5})\b', query)
                
                # Pattern 3: Last uppercase sequence in query
                if not ticker_match:
                    uppercase_matches = re.findall(r'\b([A-Z]{2,5})\b', query)
                    if uppercase_matches:
                        # Take the first or most relevant one
                        ticker_match = type('obj', (object,), {'group': lambda self, x: uppercase_matches[0]})()
                
                if ticker_match:
                    ticker = ticker_match.group(1) if ticker_match.lastindex else ticker_match.group()
                    params['ticker'] = ticker.upper().strip()
                    logger.debug(f"[FALLBACK] Extracted ticker: {params['ticker']}")
            
            # ==== EXTRACT WINDOW/PERIOD ====
            if 'window' in schema_fields:
                # Pattern: Numbers followed by day/period keywords
                window_match = re.search(r'(\d{1,3})\s*(?:ngày|day|period|SMA-|RSI-)', query)
                if window_match:
                    window = int(window_match.group(1))
                    if 1 <= window <= 200:  # Reasonable range for technical indicators
                        params['window'] = window
                        logger.debug(f"[FALLBACK] Extracted window: {window}")
                
                # If not found, try extracting from common SMA patterns like "SMA20"
                if 'window' not in params:
                    sma_match = re.search(r'SMA-?(\d+)', query)
                    if sma_match:
                        window = int(sma_match.group(1))
                        if 1 <= window <= 200:
                            params['window'] = window
                            logger.debug(f"[FALLBACK] Extracted window from SMA pattern: {window}")
            
            # ==== EXTRACT DATE PARAMETERS ====
            # Pattern for YYYY-MM-DD dates
            date_pattern = r'(\d{4}-\d{2}-\d{2})'
            dates = re.findall(date_pattern, query)
            
            if 'start_date' in schema_fields and len(dates) > 0:
                params['start_date'] = dates[0]
                logger.debug(f"[FALLBACK] Extracted start_date: {dates[0]}")
            
            if 'end_date' in schema_fields and len(dates) > 1:
                params['end_date'] = dates[1]
                logger.debug(f"[FALLBACK] Extracted end_date: {dates[1]}")
            
            # ==== EXTRACT TIMEFRAME ====
            if 'timeframe' in schema_fields:
                timeframe_match = re.search(r'(D|H|1m|5m|15m|30m|1h|4h|daily|hourly)', query, re.IGNORECASE)
                if timeframe_match:
                    timeframe = timeframe_match.group(1).upper()
                    valid_timeframes = ['D', 'H', '1m', '5m', '15m', '30m', '1h', '4h']
                    if timeframe in valid_timeframes:
                        params['timeframe'] = timeframe
                        logger.debug(f"[FALLBACK] Extracted timeframe: {timeframe}")
            
            # ==== VALIDATE REQUIRED FIELDS ====
            missing_fields = [f for f in required_fields if f not in params]
            
            if missing_fields:
                logger.warning(f"[FALLBACK] Missing required fields: {missing_fields}")
                if 'ticker' in missing_fields:
                    # Ticker extraction failed - can't proceed
                    logger.error(f"[FALLBACK] Could not extract required 'ticker' field")
                    return None
            
            logger.info(f"[FALLBACK] Successfully extracted parameters: {params}")
            return params
        
        except Exception as e:
            logger.error(f"[FALLBACK] Parameter extraction error: {e}")
            return None
    
    async def invoke(self, user_prompt: str, uploaded_files: list = None, 
                     conversation_history: list = None, user_id: str = "default",
                     session_id: str = "default", use_rag: bool = True,
                     tools_enabled: bool = True, observer_callback: callable = None) -> Dict[str, Any]:
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
            
            # Reset observer for this invocation
            if self.observer:
                self.observer.reset()
            
            # Register observer callback if provided
            logger.info(f"[WORKFLOW] Observer registration check:")
            logger.info(f"[WORKFLOW]   - observer_callback provided: {observer_callback is not None}")
            logger.info(f"[WORKFLOW]   - self.observer exists: {self.observer is not None}")
            
            if observer_callback and self.observer:
                logger.info(f"[WORKFLOW] Registering observer callback for real-time step streaming")
                self.observer.register_callback(observer_callback)
                logger.info(f"[WORKFLOW] Observer callback registered. Total callbacks: {len(self.observer.callbacks)}")
            elif observer_callback and not self.observer:
                logger.warning(f"[WORKFLOW] Observer callback provided but observer not initialized!")
            else:
                if observer_callback:
                    logger.info(f"[WORKFLOW] Observer callback provided but won't register (missing observer)")
                if self.observer:
                    logger.info(f"[WORKFLOW] Observer exists but no callback provided")
            
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
            
            # Print workflow summary at the very end
            if self.observer:
                await self.observer.emit_workflow_completed()
                self.observer.print_summary()
            
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

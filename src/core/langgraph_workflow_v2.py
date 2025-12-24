"""
ENHANCED LangGraphWorkflow - File Handling Integration

Adds file processing nodes while maintaining core 2-node agent behavior:
- Extract Data node: Parses uploaded files
- Ingest File node: Stores parsed files in RAG
- Agent node: LLM decision making + tool selection
- Tools node: Execute selected tools

This maintains behavioral parity with the original system while moving
file handling into the workflow.
"""

import logging
import asyncio
from typing import Optional, List, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage

from .workflow_state import WorkflowState, create_initial_state

logger = logging.getLogger(__name__)


class LangGraphWorkflow:
    """
    Enhanced workflow with file processing integration.
    
    Nodes:
    1. EXTRACT_DATA: Parse uploaded files using FileProcessingPipeline
    2. INGEST_FILE: Store extracted data in personal RAG
    3. AGENT: LLM decision making (tool selection)
    4. TOOLS: Execute selected tools
    
    Flow:
    - EXTRACT_DATA → INGEST_FILE → AGENT → (TOOLS) → AGENT → END
    """
    
    def __init__(self, agent_executor):
        """
        Initialize workflow with agent executor.
        
        Args:
            agent_executor: FinancialAgent instance with tools, LLM, etc.
        """
        self.agent = agent_executor
        self.graph = self._build_graph()
        logger.info("LangGraphWorkflow initialized with file handling integration")
    
    def _build_graph(self) -> StateGraph:
        """
        Build the enhanced workflow with file processing.
        
        Flow:
        EXTRACT_DATA → INGEST_FILE → AGENT → (TOOLS) → AGENT → END
        """
        workflow = StateGraph(WorkflowState)
        
        # Add file processing nodes
        workflow.add_node("extract_data", self.node_extract_data)
        workflow.add_node("ingest_file", self.node_ingest_file)
        
        # Add core agent nodes
        workflow.add_node("agent", self.node_agent)
        workflow.add_node("tools", self.node_tools)
        
        # Set entry point
        workflow.set_entry_point("extract_data")
        
        # File processing flow
        workflow.add_edge("extract_data", "ingest_file")
        workflow.add_edge("ingest_file", "agent")
        
        # Agent conditional flow: if tools selected → tools node, else → END
        workflow.add_conditional_edges(
            "agent",
            self._route_after_agent,
            {
                "tools": "tools",
                "end": END,
            }
        )
        
        # After tools, always back to agent for final synthesis
        workflow.add_edge("tools", "agent")
        
        return workflow.compile()
    
    # ========== FILE PROCESSING NODES ==========
    
    async def node_extract_data(self, state: WorkflowState) -> Dict[str, Any]:
        """
        EXTRACT_DATA node: Parse uploaded files.
        
        Uses FileProcessingPipeline to extract structured data from:
        - PDF files
        - Excel files  
        - Image files (via OCR)
        
        Returns extracted data and updates workflow state.
        """
        logger.info("\n" + "="*60)
        logger.info(">>> EXTRACT_DATA NODE")
        logger.info("="*60)
        
        uploaded_files = state.get("uploaded_files", [])
        
        if not uploaded_files:
            logger.info("✓ No files to extract")
            return {
                "extracted_file_data": None,
                "needs_file_processing": False,
            }
        
        logger.info(f"📄 Processing {len(uploaded_files)} file(s)")
        
        try:
            from ..core.file_processing_pipeline import FileProcessingPipeline
            
            pipeline = FileProcessingPipeline()
            extracted_data = {}
            
            for file_info in uploaded_files:
                file_path = file_info.get("path")
                file_name = file_info.get("name")
                file_type = file_info.get("type")
                
                logger.info(f"  • Extracting: {file_name} ({file_type})")
                
                try:
                    # Process file using pipeline
                    result = pipeline.process(
                        file_path=file_path,
                        file_type=file_type,
                        file_name=file_name
                    )
                    
                    extracted_data[file_name] = {
                        "success": True,
                        "content": result.get("text", ""),
                        "chunks": result.get("chunks", []),
                        "metadata": {
                            "file_name": file_name,
                            "file_type": file_type,
                            "chunk_count": len(result.get("chunks", []))
                        }
                    }
                    
                    logger.info(f"    ✓ Extracted {len(result.get('chunks', []))} chunks")
                    
                except Exception as e:
                    logger.error(f"    ✗ Extraction failed: {e}")
                    extracted_data[file_name] = {
                        "success": False,
                        "error": str(e)
                    }
            
            logger.info(f"✓ File extraction complete: {len(extracted_data)} file(s) processed")
            
            return {
                "extracted_file_data": extracted_data if extracted_data else None,
                "needs_file_processing": len(extracted_data) > 0,
            }
            
        except Exception as e:
            logger.error(f"❌ File extraction failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "extracted_file_data": None,
                "needs_file_processing": False,
            }
    
    async def node_ingest_file(self, state: WorkflowState) -> Dict[str, Any]:
        """
        INGEST_FILE node: Store extracted data in personal RAG.
        
        Uses FileIngestionService to:
        - Embed extracted chunks
        - Store in personal conversation-isolated vectordb
        - Record file IDs for retrieval
        
        Returns list of ingested file IDs.
        """
        logger.info("\n" + "="*60)
        logger.info(">>> INGEST_FILE NODE")
        logger.info("="*60)
        
        extracted_data = state.get("extracted_file_data")
        user_id = state.get("user_id", "default")
        session_id = state.get("session_id", "default")
        
        if not extracted_data:
            logger.info("✓ No extracted data to ingest")
            return {
                "ingested_file_ids": [],
                "metadata": {
                    **state.get("metadata", {}),
                    "ingestion_status": "no_files"
                }
            }
        
        logger.info(f"💾 Ingesting {len(extracted_data)} file(s) to personal RAG")
        
        try:
            from ..services.file_ingestion_service import FileIngestionService
            from ..services.multi_collection_rag_service import get_rag_service
            
            ingestion_service = FileIngestionService()
            rag_service = get_rag_service()
            ingested_file_ids = []
            total_chunks_added = 0
            
            for file_name, file_data in extracted_data.items():
                if not file_data.get("success"):
                    logger.warning(f"  ⊘ Skipping {file_name}: extraction failed")
                    continue
                
                logger.info(f"  • Ingesting: {file_name}")
                
                try:
                    chunks = file_data.get("chunks", [])
                    content = file_data.get("content", "")
                    
                    # Add to RAG using the service
                    # The service handles embedding and storage
                    for i, chunk in enumerate(chunks):
                        # Add chunk to personal collection (conversation isolated)
                        rag_service.add_document(
                            user_id=user_id,
                            chat_session_id=session_id,
                            text=chunk.get("content", chunk) if isinstance(chunk, dict) else chunk,
                            title=f"{file_name} - Part {i+1}",
                            source=file_name,
                            metadata={
                                "file_type": file_data.get("metadata", {}).get("file_type"),
                                "chunk_index": i,
                                "total_chunks": len(chunks),
                                "session_id": session_id
                            }
                        )
                    
                    ingested_file_ids.append(file_name)
                    total_chunks_added += len(chunks)
                    logger.info(f"    ✓ Ingested {len(chunks)} chunks")
                    
                except Exception as e:
                    logger.error(f"    ✗ Ingestion failed: {e}")
            
            logger.info(f"✓ Ingestion complete: {len(ingested_file_ids)} file(s), {total_chunks_added} chunks total")
            
            return {
                "ingested_file_ids": ingested_file_ids,
                "metadata": {
                    **state.get("metadata", {}),
                    "ingestion_status": "complete",
                    "files_ingested": len(ingested_file_ids),
                    "chunks_added": total_chunks_added
                }
            }
            
        except Exception as e:
            logger.error(f"❌ File ingestion failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "ingested_file_ids": [],
                "metadata": {
                    **state.get("metadata", {}),
                    "ingestion_status": "failed",
                    "error": str(e)
                }
            }
    
    # ========== ROUTING FUNCTION ==========
    
    def _route_after_agent(self, state: WorkflowState) -> str:
        """
        Route after AGENT node.
        If tool calls detected, go to TOOLS.
        Otherwise, go to END.
        """
        if state.get("conversation_history"):
            messages = state["conversation_history"]
            if messages:
                last_msg = messages[-1]
                if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                    return "tools"
        
        return "end"
    
    # ========== CORE AGENT NODES ==========
    
    async def node_agent(self, state: WorkflowState) -> Dict[str, Any]:
        """
        AGENT node: LLM decision making + first/final answer synthesis.
        
        Handles:
        - First pass: LLM with tools (decide whether to call tools)
        - Second pass: Final answer synthesis (after tool results)
        """
        logger.info("\n" + "="*60)
        logger.info(">>> AGENT NODE INVOKED")
        logger.info("="*60)
        
        # Get current state
        messages = state.get("conversation_history", [])
        query = state["user_prompt"]
        rag_context = state.get("best_search_results", [])
        
        # Log current state
        logger.info(f"📬 Message count in state: {len(messages)}")
        logger.info(f"📝 Last message type: {type(messages[-1]).__name__ if messages else 'None'}")
        if messages:
            last_preview = str(messages[-1].content)[:100] if hasattr(messages[-1], 'content') else str(messages[-1])[:100]
            logger.info(f"    Preview: content='{last_preview}'")
        
        # Message breakdown
        message_types = {}
        for msg in messages:
            msg_type = type(msg).__name__
            message_types[msg_type] = message_types.get(msg_type, 0) + 1
        logger.info(f"📊 Message breakdown: {message_types}")
        
        # Check if this is a follow-up (after tools were executed)
        is_followup = any(type(msg).__name__ == 'ToolMessage' for msg in messages)
        
        if is_followup:
            # Second pass: final synthesis with tool results
            logger.info("🔧 TOOL MESSAGE DETECTED - Processing tool result")
            tool_result_len = len(str(messages[-1].content)) if hasattr(messages[-1], 'content') else 0
            logger.info(f"    Result length: {tool_result_len} chars")
            logger.info("    📊 RAG context detected + tool result - merging both sources" if rag_context else "")
        else:
            # First pass: prepare for LLM invocation
            logger.info("🤖 PREPARING LLM INVOCATION")
            logger.info(f"    System prompt size: 12962 chars")
            logger.info(f"    Tools available: {len(self.agent.tools)}")
            
            # Check if tools should be enabled
            should_allow_tools = state.get("tools_enabled", True)
            logger.info(f"    Tools allowed: {should_allow_tools}")
            
            if rag_context:
                logger.info("    ✓ RAG context detected in messages")
                logger.info("    ℹ️  RAG present with tools enabled - will decide based on relevance")
            else:
                logger.info("    ✗ No RAG context in messages")
        
        # Prepare LLM invocation
        logger.info("⚙️  INVOKING LLM...")
        
        try:
            # Build prompt with full system context
            from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
            
            # System prompt with tool descriptions
            system_text = """Bạn là một trợ lý tư vấn tài chính chuyên về thị trường chứng khoán Việt Nam.

Khả năng của bạn:
- Cung cấp thông tin công ty và dữ liệu tài chính
- Phân tích giá cổ phiếu và xu hướng thị trường
- Tính toán các chỉ số kỹ thuật (SMA, RSI, MACD, v.v.)
- Giải thích các khái niệm tài chính

Hướng dẫn:
- Nếu bạn có tài liệu liên quan, hãy sử dụng chúng làm nguồn
- QUAN TRỌNG: Nếu người dùng yêu cầu tính toán (SMA, RSI) hoặc dữ liệu giá cổ phiếu, PHẢI GỌI công cụ phù hợp
- Nếu không có đủ thông tin, hãy nêu rõ điều đó
- Luôn cung cấp các giải thích chi tiết
- Sử dụng tiếng Việt trừ khi người dùng yêu cầu ngôn ngữ khác
"""
            
            # Add RAG context if available
            if rag_context:
                system_text += "\n📚 Tài liệu liên quan:\n"
                for i, doc in enumerate(rag_context[:5], 1):
                    title = doc.get('title', 'Unknown')
                    score = doc.get('score', doc.get('similarity', 0))
                    system_text += f"  {i}. {title} (score: {score:.1%})\n"
            
            # Create prompt template
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_text),
                MessagesPlaceholder(variable_name="messages"),
            ])
            
            # Bind tools to LLM to enable tool calling
            allow_tools = state.get("tools_enabled", True)
            
            if allow_tools:
                llm_with_tools = self.agent.llm.bind_tools(self.agent.tools)
                chain = prompt | llm_with_tools
            else:
                logger.info("    🛑 Tools disabled - LLM will answer without tools")
                chain = prompt | self.agent.llm
            
            # Invoke LLM
            response = await asyncio.to_thread(chain.invoke, {"messages": messages})
            
            # Add response to messages
            updated_messages = messages + [response]
            
            # Check for tool calls
            tool_calls = []
            if allow_tools and hasattr(response, 'tool_calls') and response.tool_calls:
                tool_calls = response.tool_calls
                logger.info(f"✓ TOOL CALLS DETECTED: {[tc.get('name', 'unknown') for tc in tool_calls]}")
                logger.info(f"    Tool count: {len(tool_calls)}")
                for i, tc in enumerate(tool_calls, 1):
                    logger.info(f"    [{i}] {tc.get('name', 'unknown')}")
                logger.info(f"🔄 ROUTING DECISION: Tool calls found → TOOLS NODE")
                logger.info(f"    Tools to execute: {[tc.get('name', 'unknown') for tc in tool_calls]}")
            else:
                logger.info("✓ NO TOOL CALLS")
                logger.info("    → LLM decided to answer directly")
                logger.info("✅ ROUTING DECISION: No tool calls → END")
            
            return {
                "conversation_history": updated_messages,
                "generated_answer": response.content if hasattr(response, 'content') else str(response),
            }
            
        except Exception as e:
            logger.error(f"❌ Agent invocation failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "conversation_history": messages,
                "generated_answer": f"Error: {str(e)}"
            }
    
    async def node_tools(self, state: WorkflowState) -> Dict[str, Any]:
        """
        TOOLS node: Execute selected tools and return results.
        
        Uses LangGraph's ToolNode for proper tool execution.
        """
        logger.info("\n" + "="*60)
        logger.info(">>> TOOLS NODE INVOKED")
        logger.info("="*60)
        
        messages = state.get("conversation_history", [])
        
        if not messages:
            logger.warning("No messages in state")
            return {"conversation_history": messages}
        
        # Get last AI message with tool calls
        last_ai_message = None
        for msg in reversed(messages):
            if type(msg).__name__ == 'AIMessage':
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    last_ai_message = msg
                    break
        
        if not last_ai_message or not last_ai_message.tool_calls:
            logger.warning("No tool calls found in last message")
            return {"conversation_history": messages}
        
        # Use LangGraph's ToolNode for proper tool execution
        tool_node = ToolNode(self.agent.tools)
        
        try:
            logger.info(f"Executing {len(last_ai_message.tool_calls)} tool call(s)...")
            
            # Create a minimal state dict for ToolNode.invoke
            tool_state = {
                "messages": messages
            }
            
            # Invoke ToolNode (runs synchronously)
            result_state = await asyncio.to_thread(
                tool_node.invoke,
                tool_state
            )
            
            # Extract updated messages from result
            updated_messages = result_state.get("messages", messages)
            
            logger.info(f"✓ Tools executed successfully")
            logger.info(f"✓ {len(updated_messages) - len(messages)} result message(s) added to context")
            
            return {"conversation_history": updated_messages}
            
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Create error message
            error_message = AIMessage(
                content=f"Xin lỗi, công cụ gặp lỗi: {str(e)}"
            )
            return {"conversation_history": messages + [error_message]}
    
    async def invoke(
        self,
        user_prompt: Optional[str] = None,
        uploaded_files: Optional[List[Dict]] = None,
        conversation_history: Optional[List[Dict]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        rag_results: Optional[List[Dict]] = None,
        tools_enabled: bool = True
    ) -> WorkflowState:
        """
        Main entry point - invoke the enhanced workflow.
        
        Flow:
        1. EXTRACT_DATA: Parse uploaded files
        2. INGEST_FILE: Store to personal RAG
        3. AGENT: LLM decision making
        4. TOOLS: Execute selected tools
        5. AGENT: Final synthesis
        
        Args:
            user_prompt: User's question (already rewritten at API level)
            uploaded_files: Files uploaded (new!)
            conversation_history: Previous messages (assembled at API level)
            user_id: User identifier
            session_id: Session identifier
            rag_results: Pre-retrieved RAG results from API level
            tools_enabled: Whether to allow tool calling
            
        Returns:
            Final WorkflowState with generated_answer populated
        """
        logger.info("\n" + "="*60)
        logger.info("🚀 LANGGRAPH WORKFLOW INVOKED (WITH FILE HANDLING)")
        logger.info("="*60)
        
        # Create initial state
        initial_state = create_initial_state(
            user_prompt=user_prompt,
            uploaded_files=uploaded_files,
            conversation_history=conversation_history
        )
        
        # Add additional context from API preprocessing
        initial_state["user_id"] = user_id or "default"
        initial_state["session_id"] = session_id or "default"
        initial_state["best_search_results"] = rag_results or []
        initial_state["tools_enabled"] = tools_enabled
        
        # Run workflow
        final_state = await self.graph.ainvoke(initial_state)
        
        logger.info("="*60)
        logger.info("✅ LANGGRAPH WORKFLOW COMPLETED")
        logger.info("="*60 + "\n")
        
        return final_state


def get_langgraph_workflow(agent_executor) -> LangGraphWorkflow:
    """Factory function to get workflow instance"""
    return LangGraphWorkflow(agent_executor)

"""
ENHANCED LangGraphWorkflow - File Handling Integration

Adds file processing nodes while maintaining core 2-node agent behavior:
- Extract Data node: Parses uploaded files
- Ingest File node: Stores parsed files in RAG
- Agent node: LLM decision making + tool selection
- Tools node: Execute selected tools

This maintains behavioral parity with the original system while moving
file handling into the workflow.

Flow: EXTRACT_DATA → INGEST_FILE → AGENT → (TOOLS) → AGENT → END
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
    EXTRACT_DATA → INGEST_FILE → AGENT → (TOOLS) → AGENT → END
    """
    
    def __init__(self, agent_executor):
        """
        Initialize workflow with agent executor.
        
        Args:
            agent_executor: FinancialAgent instance with tools, LLM, etc.
        """
        self.agent = agent_executor
        self.graph = self._build_graph()
        logger.info("LangGraphWorkflow initialized with simplified agent-centric architecture")
    
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
        
        # Set entry point to file extraction
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
        logger.info("="*30)
        logger.info(">>> EXTRACT_DATA NODE")
        logger.info("="*30)
        
        uploaded_files = state.get("uploaded_files", [])
        
        if not uploaded_files:
            logger.info("✓ No files to extract")
            logger.info("📋 EXTRACT_DATA NODE SUMMARY")
            logger.info("="*30)
            logger.info("Files to process: 0")
            logger.info("Status: SKIPPED (no files)")
            logger.info("="*30)
            return {
                "extracted_file_data": None,
                "needs_file_processing": False,
            }
        
        logger.info(f"📄 Processing {len(uploaded_files)} file(s)")
        
        try:
            from ..services.file_processing_pipeline import FileProcessingPipeline
            
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
            
            # Add node summary with full file info
            logger.info("📋 EXTRACT_DATA NODE SUMMARY")
            logger.info("="*30)
            logger.info(f"Files to process: {len(uploaded_files)}")
            successful_files = sum(1 for f in extracted_data.values() if f.get("success"))
            logger.info(f"Successfully extracted: {successful_files}")
            total_chunks = sum(len(f.get("chunks", [])) for f in extracted_data.values() if f.get("success"))
            logger.info(f"Total chunks created: {total_chunks}")
            for file_name, file_data in extracted_data.items():
                if file_data.get("success"):
                    logger.info(f"File: {file_name} | Type: {file_data.get('metadata', {}).get('file_type')} | Chunks: {len(file_data.get('chunks', []))}")
                    logger.info(f"  Content preview: {file_data.get('content', '')[:200]}...")
            logger.info("="*30)
            
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
        logger.info("="*30)
        logger.info(">>> INGEST_FILE NODE")
        logger.info("="*30)
        
        extracted_data = state.get("extracted_file_data")
        user_id = state.get("user_id", "default")
        session_id = state.get("session_id", "default")
        
        if not extracted_data:
            logger.info("✓ No extracted data to ingest")
            logger.info("📋 INGEST_FILE NODE SUMMARY")
            logger.info("="*30)
            logger.info("Files to ingest: 0")
            logger.info("Status: SKIPPED (no files)")
            logger.info("="*30)
            return {
                "ingested_file_ids": [],
                "metadata": {
                    **state.get("metadata", {}),
                    "ingestion_status": "no_files"
                }
            }
        
        logger.info(f"💾 Ingesting {len(extracted_data)} file(s) to personal RAG")
        
        try:
            from ..services.multi_collection_rag_service import get_rag_service
            
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
                    
                    # Add chunk to personal collection (conversation isolated)
                    for i, chunk in enumerate(chunks):
                        # Extract chunk content
                        if isinstance(chunk, dict):
                            chunk_content = chunk.get("content", str(chunk))
                        else:
                            chunk_content = str(chunk)
                        
                        # Add to RAG service
                        rag_service.add_document(
                            user_id=user_id,
                            chat_session_id=session_id,
                            text=chunk_content,
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
            
            # Add node summary with full ingestion details
            logger.info("📋 INGEST_FILE NODE SUMMARY")
            logger.info("="*30)
            logger.info(f"Files to ingest: {len(extracted_data)}")
            logger.info(f"Successfully ingested: {len(ingested_file_ids)}")
            logger.info(f"Total chunks ingested: {total_chunks_added}")
            logger.info(f"User: {user_id}")
            logger.info(f"Session: {session_id}")
            for file_id in ingested_file_ids:
                logger.info(f"Ingested file: {file_id}")
            logger.info("="*30)
            
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
        If tool calls detected in conversation history, go to TOOLS.
        Otherwise, go to END.
        """
        # Check if the last AI message has tool_calls
        if state.get("conversation_history"):
            # Look at message history for tool calls
            messages = state["conversation_history"]
            if messages:
                last_msg = messages[-1]
                # If it's an AIMessage with tool_calls, execute tools
                if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                    return "tools"
        
        return "end"
    
    # ========== NODE IMPLEMENTATIONS (2 nodes matching actual behavior) ==========
    
    async def node_agent(self, state: WorkflowState) -> Dict[str, Any]:
        """
        AGENT node: LLM decision making + first/final answer synthesis.
        
        From logs:
        - Calls LLM with 8 tools available (or disabled)
        - Decides whether to use tools or answer directly
        - On second pass (after tools), synthesizes final answer
        - Updates conversation_history with messages
        """
        logger.info("="*30)
        logger.info(">>> AGENT NODE INVOKED")
        logger.info("="*30)
        
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
            
            # Add node summary with full answer
            full_answer = response.content if hasattr(response, 'content') else str(response)
            logger.info("📋 AGENT NODE SUMMARY")
            logger.info("="*30)
            logger.info(f"Input: {len(messages)} messages")
            logger.info(f"LLM Decision: {'Call tools' if tool_calls else 'Answer directly'}")
            logger.info(f"Tool Calls: {len(tool_calls)} tools")
            for i, tc in enumerate(tool_calls, 1):
                logger.info(f"  [{i}] {tc.get('name', 'unknown')} - Args: {tc.get('args', {})}")
            logger.info(f"Full Answer:")
            logger.info(f"{full_answer}")
            logger.info(f"RAG Used: {'Yes' if rag_context else 'No'}")
            logger.info("="*30)
            
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
    
    def _agent_llm_invocation(self, messages, query, rag_context, allow_tools):
        """
        Internal helper to handle LLM invocation with tool binding.
        This is where the actual LLM call happens with the agent's tools.
        
        CRITICAL: Use .bind_tools() to enable tool calling!
        """
        # Build context if RAG results available
        rag_text = ""
        if rag_context:
            rag_text = "\n📚 Retrieved Documents:\n"
            for i, doc in enumerate(rag_context[:5], 1):
                title = doc.get('title', 'Unknown')
                score = doc.get('score', doc.get('similarity', 0))
                rag_text += f"  {i}. {title} (score: {score:.1%})\n"
        
        # System prompt with tool descriptions
        system_prompt = f"""You are a professional financial advisor specializing in Vietnamese stock market.

Available Tools:
"""
        # Add tool descriptions
        for tool in self.agent.tools:
            tool_name = getattr(tool, 'name', 'unknown')
            tool_desc = getattr(tool, 'description', 'No description')
            system_prompt += f"- {tool_name}: {tool_desc}\n"
        
        system_prompt += f"""
Your capabilities:
- Provide company and financial data analysis
- Analyze stock prices and market trends
- Calculate technical indicators (SMA, RSI, MACD, etc.)
- Explain financial concepts

Guidelines:
- Use provided documents as primary source when available
- Only call tools when necessary
- State clearly if you lack information
- Always provide detailed explanations
- Respond in Vietnamese unless user requests otherwise
- If query asks for calculations (SMA, RSI, price data), USE TOOLS

{f"Documents:{rag_text}" if rag_text else ""}"""
        
        # Build prompt with messages
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        # CRITICAL: Bind tools to LLM to enable tool calling
        if allow_tools:
            llm_with_tools = self.agent.llm.bind_tools(self.agent.tools)
            chain = prompt | llm_with_tools
        else:
            # Tools disabled - use plain LLM
            chain = prompt | self.agent.llm
        
        # Invoke the chain
        response = chain.invoke({"messages": messages})
        
        # Extract tool calls from response
        tool_calls = []
        if allow_tools and hasattr(response, 'tool_calls') and response.tool_calls:
            tool_calls = response.tool_calls
        
        return {
            "messages": messages + [response],
            "tool_calls": tool_calls,
            "answer": response.content if hasattr(response, 'content') else str(response)
        }
    
    async def node_tools(self, state: WorkflowState) -> Dict[str, Any]:
        """
        TOOLS node: Execute selected tools and return results.
        
        Uses LangGraph's ToolNode for proper tool execution.
        
        From logs:
        - Extracts tool calls from last AI message
        - Executes tools (e.g., get_company_info, calculate_sma)
        - Returns tool results as ToolMessage
        - Routes back to AGENT for final synthesis
        """
        logger.info("="*30)
        logger.info(">>> TOOLS NODE INVOKED")
        logger.info("="*30)
        
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
        from langgraph.prebuilt import ToolNode
        
        tool_node = ToolNode(self.agent.tools)
        
        # ToolNode.invoke expects the full state dict
        # It will execute tools and add ToolMessages to the messages
        try:
            logger.info(f"Executing {len(last_ai_message.tool_calls)} tool call(s)...")
            
            # Create a minimal state dict for ToolNode.invoke
            tool_state = {
                "messages": messages
            }
            
            logger.info(f"📤 Initial message count: {len(messages)}")
            
            # Invoke ToolNode (runs synchronously)
            result_state = await asyncio.to_thread(
                tool_node.invoke,
                tool_state
            )
            
            # Extract updated messages from result
            updated_messages = result_state.get("messages", messages)
            messages_added = max(0, len(updated_messages) - len(messages))  # Ensure non-negative
            
            logger.info(f"✓ Tools executed successfully")
            logger.info(f"✓ {messages_added} result message(s) added to context (total: {len(updated_messages)} messages)")
            
            # Add node summary with full tool results
            logger.info("📋 TOOLS NODE SUMMARY")
            logger.info("="*30)
            logger.info(f"Input: {len(messages)} messages")
            logger.info(f"Tool Calls Executed: {len(last_ai_message.tool_calls)}")
            for i, tc in enumerate(last_ai_message.tool_calls, 1):
                logger.info(f"  [{i}] {tc.get('name', 'unknown')} - Args: {tc.get('args', {})}")
            logger.info(f"Tool Results: {messages_added} message(s) added")
            logger.info(f"Output: {len(updated_messages)} total messages")
            # Log full tool results
            for msg in updated_messages[-messages_added:]:
                if hasattr(msg, 'content'):
                    logger.info(f"Tool Result:")
                    logger.info(f"{msg.content}")
            logger.info("="*30)
            
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
        Main entry point - invoke the simplified workflow.
        
        NOTE: This workflow receives PREPROCESSED input:
        - Prompt rewriting already done at API level
        - RAG retrieval already done at API level
        - Conversation history already assembled
        
        Workflow responsibility:
        - LLM agent decision making (tool selection)
        - Tool execution
        - Final answer synthesis
        
        Args:
            user_prompt: User's question (already rewritten at API level)
            uploaded_files: Files uploaded (optional)
            conversation_history: Previous messages (assembled at API level)
            user_id: User identifier
            session_id: Session identifier
            rag_results: Pre-retrieved RAG results from API level
            tools_enabled: Whether to allow tool calling
            
        Returns:
            Final WorkflowState with generated_answer populated
        """
        logger.info("="*30)
        logger.info("🚀 LANGGRAPH WORKFLOW INVOKED")
        logger.info("="*30)
        logger.info(f"User Prompt: {user_prompt}")
        logger.info(f"Files: {len(uploaded_files) if uploaded_files else 0}")
        logger.info(f"History messages: {len(conversation_history) if conversation_history else 0}")
        logger.info(f"RAG results: {len(rag_results) if rag_results else 0}")
        
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
        
        logger.info("="*30)
        logger.info("✅ LANGGRAPH WORKFLOW COMPLETED")
        logger.info(f"Final Answer:")
        logger.info(f"{final_state.get('generated_answer', 'No answer generated')}")
        logger.info("="*30)
        
        return final_state


def get_langgraph_workflow(agent_executor) -> LangGraphWorkflow:
    """Factory function to get workflow instance"""
    return LangGraphWorkflow(agent_executor)

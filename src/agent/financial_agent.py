"""
Financial Agent - Agent chính sử dụng LangGraph và ReAct pattern
"""

import logging
import os
import asyncio
from typing import Literal
from pathlib import Path

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import json

from .state import AgentState
from ..llm import LLMFactory
from ..tools import get_all_tools
from ..core.summarization import summarize_tool_result
from ..core.tool_config import ToolsConfig, DEFAULT_TOOLS_CONFIG

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FinancialAgent:
    """
    Financial Agent sử dụng LangGraph và ReAct pattern
    
    Workflow:
    1. User input → HumanMessage
    2. Agent node: LLM phân tích và quyết định dùng tool nào
    3. Tool node: Thực thi tools (get_company_info, get_historical_data, calculate_sma, calculate_rsi)
    4. Agent node: Nhận kết quả tool, tổng hợp và trả lời
    5. End: Trả về câu trả lời cuối cùng
    """
    
    def __init__(self, config: ToolsConfig = None):
        """
        Initialize Financial Agent.
        
        Args:
            config: Optional ToolsConfig for tool and feature configuration
        """
        logger.info("Initializing Financial Agent...")
        
        # Use provided config or default
        self.config = config or DEFAULT_TOOLS_CONFIG
        logger.info(f"Using config: tools={self.config.enabled_tools}, RAG={self.config.allow_rag}")
        
        # Get LLM from factory
        self.llm = LLMFactory.get_llm()
        
        # Initialize RAG service for workflow access
        try:
            from ..services.multi_collection_rag_service import get_rag_service
            self.rag_service = get_rag_service(llm=self.llm)
            logger.info("✅ RAG service initialized with LLM for metric extraction")
        except Exception as e:
            logger.warning(f"❌ RAG service initialization failed: {e}")
            self.rag_service = None
        
        # Get all tools (config-filtered)
        self.tools = get_all_tools(self.config)
        tool_names = []
        for t in self.tools:
            if hasattr(t, 'name'):
                tool_names.append(t.name)
            elif isinstance(t, dict) and 'name' in t:
                tool_names.append(t['name'])
            else:
                tool_names.append(str(t))
        logger.info(f"Loaded {len(self.tools)} tools: {tool_names}")
        
        # Load system prompt
        prompts_dir = Path(__file__).parent / "prompts"
        with open(prompts_dir / "system_prompt.txt", "r", encoding="utf-8") as f:
            self.system_prompt = f.read().strip()
        
        # Create old-style LangGraph workflow (kept for backward compatibility)
        self.app = self._create_graph()
        
        # Import configuration for workflow version selection
        from ..core.config import settings
        
        # Load all available workflows
        # Create legacy 10-node LangGraph workflow (fallback)
        from ..core.langgraph_workflow import get_langgraph_workflow
        self.langgraph_workflow = get_langgraph_workflow(self)
        
        # Create V3 workflow (8-node)
        try:
            from ..core.langgraph_workflow_v3 import LangGraphWorkflowV3
            self.langgraph_workflow_v3 = LangGraphWorkflowV3(self)
            logger.info("✅ V3 workflow loaded (8-node enhanced architecture)")
        except Exception as e:
            logger.warning(f"❌ V3 workflow not available: {e}")
            self.langgraph_workflow_v3 = None
        
        # Create V4 workflow (18-node - latest)
        try:
            from ..core.langgraph_workflow_v4 import LangGraphWorkflowV4
            self.langgraph_workflow_v4 = LangGraphWorkflowV4(self, enable_observer=settings.WORKFLOW_OBSERVER_ENABLED)
            logger.info("✅ V4 workflow loaded (18-node complete architecture)")
        except Exception as e:
            logger.warning(f"❌ V4 workflow not available: {e}")
            self.langgraph_workflow_v4 = None
        
        # Determine default workflow version based on configuration
        self.workflow_version = settings.WORKFLOW_VERSION
        self.canary_rollout_percentage = settings.CANARY_ROLLOUT_PERCENTAGE
        
        logger.info(f"📊 Workflow Configuration:")
        logger.info(f"   Default version: {self.workflow_version}")
        logger.info(f"   Canary rollout: {self.canary_rollout_percentage}%")
        logger.info(f"   Observer enabled: {settings.WORKFLOW_OBSERVER_ENABLED}")
        
        logger.info("✅ Financial Agent initialized successfully!")
        logger.info("   - Old-style workflow: self.app (backward compatible)")
        logger.info("   - Legacy 10-node: self.langgraph_workflow (fallback)")
        if self.langgraph_workflow_v3:
            logger.info("   - V3 (8-node): self.langgraph_workflow_v3")
        if self.langgraph_workflow_v4:
            logger.info("   - V4 (18-node): self.langgraph_workflow_v4 ⭐")
    
    def _get_tool_descriptions(self) -> str:
        """Get formatted tool descriptions for prompt"""
        descriptions = []
        for tool in self.tools:
            try:
                # Handle both ToolCall and regular tools
                tool_name = getattr(tool, 'name', 'unknown')
                tool_desc = getattr(tool, 'description', 'No description')
                
                if hasattr(tool, 'args_schema') and tool.args_schema:
                    args = tool.args_schema.schema().get('properties', {}).keys()
                    args_str = ', '.join(args) if args else 'None'
                else:
                    args_str = 'None'
                
                descriptions.append(
                    f"- {tool_name}: {tool_desc}\n"
                    f"  Arguments: {args_str}"
                )
            except Exception as e:
                logger.warning(f"Error getting tool description: {e}")
                continue
        
        return "\n\n".join(descriptions)
    
    async def _agent_node(self, state: AgentState) -> AgentState:
        """
        Agent node - LLM phân tích và quyết định
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with LLM response
        """
        logger.info("="*30)
        logger.info(">>> AGENT NODE INVOKED")
        logger.info("="*30)
        
        # Log the incoming message state for debugging
        message_count = len(state["messages"])
        logger.info(f"📬 Message count in state: {message_count}")
        
        # Context window awareness: check token estimate
        try:
            from ..utils.summarization import estimate_message_tokens, should_compress_history
            token_count = estimate_message_tokens(state["messages"])
            logger.info(f"📊 Estimated tokens: {token_count}/6000")
            
            # If approaching context limit, log warning
            if token_count > 5000:
                logger.warning(f"⚠️ Context window approaching limit ({token_count} tokens)")
        except Exception as e:
            logger.debug(f"Token estimation skipped: {e}")
        
        # CHECK FOR GREETING FIRST - before processing any messages
        # List of simple greetings and non-financial queries that shouldn't trigger tool calls
        greetings = [
            "hello", "hi", "hey", "xin chào", "chào", "chào bạn",
            "how are you", "bạn khỏe không", "như thế nào",
            "thanks", "cảm ơn", "thank you", "thank",
            "help me", "giúp tôi", "hỗ trợ",
            "ok", "okay", "được", "tốt"
        ]
        
        # Check if the last message is a human greeting (not a tool result)
        if state["messages"]:
            last_msg = state["messages"][-1]
            if isinstance(last_msg, HumanMessage):
                query_text = str(last_msg.content).lower().strip()
                
                # Check if query is EXACTLY a greeting or is a greeting with only punctuation
                # e.g., "hello" or "hello?" or "hello!" should match, but "hello world" should not
                query_clean = query_text.rstrip('?!.,;:')  # Remove trailing punctuation
                
                is_greeting = any(
                    query_clean == g or  # Exact match after removing punctuation
                    (query_text == g) or  # Exact match as-is
                    (query_clean.startswith(g + ' ') and len(query_clean.split()) == 2 and 
                     query_clean.split()[1] in ['?', '!', '.', ',', ';', ':'])  # Greeting + punctuation
                    for g in greetings
                )
                
                if is_greeting:
                    logger.info(f"✓ DETECTED GREETING QUERY: '{query_text}'")
                    logger.info(f"--- Responding conversationally without tools ---")
                    conversational_response = AIMessage(
                        content="Tôi là trợ lý tư vấn tài chính. Bạn có câu hỏi gì về chứng khoán Việt Nam không? Tôi có thể giúp bạn với: thông tin công ty, dữ liệu giá cổ phiếu, phân tích kỹ thuật (SMA, RSI), thông tin cổ đông, ban lãnh đạo, v.v."
                    )
                    return {"messages": state["messages"] + [conversational_response]}
        
        if state["messages"]:
            last_msg = state["messages"][-1]
            msg_type = type(last_msg).__name__
            msg_preview = str(last_msg.content)[:100] if hasattr(last_msg, 'content') else str(last_msg)[:100]
            logger.info(f"📝 Last message type: {msg_type}")
            logger.info(f"   Preview: {msg_preview}")
            
            # Count message types in state
            msg_types_count = {
                'HumanMessage': sum(1 for m in state["messages"] if isinstance(m, HumanMessage)),
                'AIMessage': sum(1 for m in state["messages"] if isinstance(m, AIMessage)),
                'ToolMessage': sum(1 for m in state["messages"] if isinstance(m, ToolMessage))
            }
            logger.info(f"📊 Message breakdown: {msg_types_count}")
            
            # If the last message is a ToolMessage (tool result), don't call tools again
            # Just process the result and generate final answer
            if isinstance(last_msg, ToolMessage):
                logger.info("🔧 TOOL MESSAGE DETECTED - Processing tool result")
                logger.info(f"   Tool name: {last_msg.tool_calls[0]['name'] if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls else 'unknown'}")
                logger.info(f"   Result length: {len(str(last_msg.content))} chars")
                
                # Check if the tool message contains an error
                tool_content = str(last_msg.content).lower()
                if "error" in tool_content or last_msg.content.startswith("Error"):
                    logger.warning(f"Tool execution failed - Full error: {last_msg.content}")
                    # Don't ask LLM to process error - just return it directly
                    error_response = AIMessage(
                        content=f"Xin lỗi, công cụ gặp lỗi: {last_msg.content}\n\nVui lòng thử lại với các tham số khác hoặc kiểm tra mã chứng khoán."
                    )
                    return {"messages": state["messages"] + [error_response]}
                
                # Check if we have RAG context - if yes, merge results
                has_rag = any("📚 Related Documents" in str(msg.content) for msg in state["messages"] if hasattr(msg, 'content'))
                
                # Summarization happens AFTER merging (if RAG present)
                # For now, just prepare tool content without summarizing
                tool_content = str(last_msg.content)
                
                if has_rag:
                    logger.info("   📊 RAG context detected + tool result - merging both sources")
                    # Extract RAG documents from messages
                    rag_docs_raw = None
                    original_question = None
                    for msg in state["messages"]:
                        if hasattr(msg, 'content') and "📚 Related Documents" in str(msg.content):
                            rag_docs_raw = str(msg.content)
                            # Extract original question (before RAG section)
                            original_question = rag_docs_raw.split("📚 Related Documents")[0].strip()
                            break
                    
                    if rag_docs_raw and original_question:
                        try:
                            # Re-retrieve RAG documents if available from state
                            rag_docs = state.get("_rag_documents", [])
                            if rag_docs:
                                merged_answer = await self._merge_rag_and_tool_results(
                                    rag_docs, 
                                    tool_content,  # Raw tool result, not summarized yet
                                    original_question
                                )
                                
                                # NOW summarize the merged answer if needed
                                if state.get("summarize_results", True) and len(merged_answer) > 500:
                                    try:
                                        from ..core.summarization import summarize_tool_result
                                        summary = summarize_tool_result({"content": merged_answer}, self.llm)
                                        if summary:
                                            logger.info(f"📌 Merged answer summarized: {summary[:80]}...")
                                            merged_answer = summary  # Replace entire answer with summary
                                    except Exception as e:
                                        logger.warning(f"Merged answer summarization skipped: {e}")
                                elif state.get("summarize_results") is False:
                                    logger.info("📋 Answer summarization disabled by user")
                                
                                final_response = AIMessage(content=merged_answer)
                                return {"messages": state["messages"] + [final_response]}
                        except Exception as e:
                            logger.warning(f"Result merging failed: {e}, will use tool result only")
                
                # No RAG context - summarize tool result directly
                logger.info("   → Generating final answer based on tool output...")
                
                if state.get("summarize_results", True) and len(tool_content) > 500:
                    try:
                        from ..core.summarization import summarize_tool_result
                        summary = summarize_tool_result({"content": tool_content}, self.llm)
                        if summary:
                            logger.info(f"📌 Tool result summarized: {summary[:80]}...")
                            tool_content = tool_content + f"\n\n📌 **Tóm tắt**: {summary}"
                    except Exception as e:
                        logger.warning(f"Tool result summarization skipped: {e}")
                elif state.get("summarize_results") is False:
                    logger.info("📋 Tool result summarization disabled by user")
                
                # Update last_msg with summarized content if changed
                if tool_content != str(last_msg.content):
                    last_msg = ToolMessage(
                        tool_call_id=last_msg.tool_call_id if hasattr(last_msg, 'tool_call_id') else "",
                        content=tool_content
                    )
                    state["messages"] = state["messages"][:-1] + [last_msg]
                
                
                # IMPORTANT: Use a STRICT result-only prompt to ensure LLM ONLY displays results
                # without explanations, code examples, or tool usage explanations
                result_only_prompt = """Bạn vừa nhận được kết quả từ một công cụ tài chính.

MỤC TIÊU DUY NHẤT: Hiển thị kết quả dữ liệu cho người dùng dưới dạng bảng Markdown rõ ràng và dễ đọc.

⚠️ TUYỆT ĐỐI KHÔNG ĐƯỢC:
- KHÔNG giải thích cấu trúc dữ liệu hay JSON
- KHÔNG nêu tên các trường như "success", "data", "date", "open", "close", v.v.
- KHÔNG viết code Python hay tham khảo json.loads, parsing
- KHÔNG nói "Dữ liệu bao gồm...", "Kết quả trả về...", "Để sử dụng..."
- KHÔNG giải thích cách gọi công cụ hoặc tham số
- KHÔNG trả lời bằng ví dụ mã code

✅ PHẢI LÀM NGAY:
1. Tạo bảng Markdown với dữ liệu THỰC TẾ (không phải mẫu/ví dụ)
2. Tiêu đề cột: Tiếng Việt dễ hiểu (ví dụ: "Ngày", "Giá", "RSI", "Trạng thái")
3. Mỗi hàng là dữ liệu thực
4. SAU bảng: Viết phân tích hoặc kết luận ngắn nếu cần thiết

💡 VÍ DỤ ĐÚNG (CHỈ HIỂN THỊ BẢNG VÀ PHÂN TÍCH):
| Ngày | Giá đóng cửa | RSI-14 | Trạng thái |
|------|-------------|--------|-----------|
| 2024-12-20 | 45,200 | 68.5 | Quá mua |
| 2024-12-19 | 44,800 | 65.2 | Qua mua |
| 2024-12-18 | 44,500 | 62.1 | Trung tính |

Nhận xét: VIC hiện đang ở vùng quá mua (RSI > 70), có thể điều chỉnh trong ngắn hạn.

❌ VÍ DỤ SAI (KHÔNG PHẢI TRẢ LỜI NHƯ DƯƠI ĐÂY):
"Kết quả là một object JSON chứa..."
"Dữ liệu bao gồm các trường..."
"Để giải quyết vấn đề này, bạn cần..."
[rồi mới hiển thị bảng]

HÀNH ĐỘNG NGAY: Đọc dữ liệu công cụ trả về, tạo bảng Markdown, giải thích kết quả. KHÔNG GIẢI THÍCH CẤU TRÚC DỮ LIỆU.
"""
                
                prompt = ChatPromptTemplate.from_messages([
                    ("system", result_only_prompt),
                    MessagesPlaceholder(variable_name="messages"),
                ])
                
                # Don't bind tools - just ask LLM to process the result
                chain = prompt | self.llm
                response = await chain.ainvoke({"messages": state["messages"]})
                content = response.content if hasattr(response, 'content') else str(response)
                
                # CRITICAL FIX: Validate that LLM didn't return an explanation instead of data
                # If it did, try to extract and format the data directly
                is_valid_response = self._validate_response_is_not_explanation(content)
                if not is_valid_response:
                    logger.error(f"❌ LLM returned explanation instead of formatted data. Attempting to extract and format data...")
                    # Try to extract actual data from tool message and format it
                    formatted_content = await self._extract_and_format_tool_data(last_msg.content)
                    if formatted_content:
                        logger.info(f"✓ Successfully extracted and formatted tool data")
                        final_response = AIMessage(content=formatted_content)
                        return {"messages": state["messages"] + [final_response]}
                    else:
                        logger.warning(f"Could not extract data, returning LLM response as-is")
                        final_response = AIMessage(content=content)
                        return {"messages": state["messages"] + [final_response]}
                
                # Create AIMessage with the processed result
                final_response = AIMessage(content=content)
                return {"messages": state["messages"] + [final_response]}
        
        # Prepare prompt with tool descriptions
        tool_descriptions = self._get_tool_descriptions()
        system_text = self.system_prompt.replace("{tool_descriptions}", tool_descriptions)
        
        logger.info("🤖 PREPARING LLM INVOCATION")
        logger.info(f"   System prompt size: {len(system_text)} chars")
        logger.info(f"   Tools available: {len(self.tools)}")
        logger.info(f"   Tools allowed: {state.get('allow_tools', True)}")
        
        has_rag_context = any("📚 Related Documents" in str(msg.content) for msg in state["messages"] if hasattr(msg, 'content'))
        if has_rag_context:
            logger.info("   ✓ RAG context detected in messages")
        else:
            logger.info("   ✗ No RAG context in messages")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_text),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        allow_tools = state.get('allow_tools', True)
        
        if allow_tools and has_rag_context:
            logger.info("   ℹ️  RAG present with tools enabled - will decide based on relevance")
            llm_with_tools = self.llm.bind_tools(self.tools)
            chain = prompt | llm_with_tools
        elif allow_tools and not has_rag_context:
            logger.info("   ✓ No RAG - tools available")
            llm_with_tools = self.llm.bind_tools(self.tools)
            chain = prompt | llm_with_tools
        else:
            logger.info("   🛑 Tools disabled - LLM will answer without tools")
            chain = prompt | self.llm
        
        try:
            logger.info("⚙️  INVOKING LLM...")
            # Invoke LLM
            response = await chain.ainvoke({"messages": state["messages"]})
            
            # ===== TOOL SELECTION REASONING LOG =====
            logger.info("[SELECT] Tool selection reasoning:")
            logger.info(f"  RAG results: {'YES' if has_rag_context else 'NO'}, score={rag_score:.2f if has_rag_context else 'N/A'} > threshold=0.50")
            
            if hasattr(response, 'tool_calls') and response.tool_calls:
                tool_names = [tc.get('name', 'unknown') for tc in response.tool_calls]
                logger.info(f"  Decision: Using TOOLS (LLM requested execution)")
                logger.info(f"  Tools: {', '.join(tool_names)}")
                
                # Check if we have RAG context - if yes, warn about unnecessary tool calls
                if has_rag_context:
                    logger.warning(f"⚠️  POTENTIAL ISSUE: Tools called despite RAG context present!")
                    logger.warning(f"   Tools: {tool_names}")
                    logger.warning(f"   This might mean RAG context wasn't sufficient or system prompt wasn't followed")
                
                logger.info(f"\n✓ TOOL CALLS DETECTED: {tool_names}")
                logger.info(f"   Tool count: {len(response.tool_calls)}")
                for i, tc in enumerate(response.tool_calls, 1):
                    tool_name = tc.get('name', 'unknown')
                    logger.info(f"   [{i}] {tool_name}")
            else:
                logger.info(f"  Decision: Using RAG or LLM-only")
                if has_rag_context:
                    logger.info(f"  Reason: RAG context sufficient (score={rag_score:.3f})")
                else:
                    logger.info(f"  Reason: General query, LLM-only response")
                logger.info(f"\n✓ NO TOOL CALLS")
                logger.info("   → LLM decided to answer directly")
            
            return {"messages": state["messages"] + [response]}
            
        except Exception as e:
            logger.error(f"Error in agent node: {e}")
            error_message = AIMessage(
                content=f"Xin lỗi, đã có lỗi xảy ra khi xử lý câu hỏi của bạn: {str(e)}"
            )
            return {"messages": state["messages"] + [error_message]}
    
    def _should_continue(self, state: AgentState) -> Literal["tools", "end"]:
        """
        Quyết định tiếp tục hay kết thúc
        
        Args:
            state: Current agent state
            
        Returns:
            "tools" nếu cần gọi tools, "end" nếu kết thúc
        """
        last_message = state["messages"][-1] if state["messages"] else None
        
        if not last_message:
            logger.info("⛔ ROUTING DECISION: No messages → END")
            return "end"
        
        # Check if last message has tool calls
        if isinstance(last_message, AIMessage) and hasattr(last_message, "tool_calls") and last_message.tool_calls:
            logger.info("🔄 ROUTING DECISION: Tool calls found → TOOLS NODE")
            logger.info(f"   Tools to execute: {[tc.get('name', 'unknown') for tc in last_message.tool_calls]}")
            return "tools"
        
        logger.info("✅ ROUTING DECISION: No tool calls → END")
        return "end"
    
    def _tools_node(self, state: AgentState) -> AgentState:
        """Custom tools node with answer-level summarization.
        
        Executes tools and summarizes results if >500 chars.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with tool results and summaries
        """
        # Use standard ToolNode to execute
        standard_tools = ToolNode(self.tools)
        result_state = standard_tools.invoke(state)
        
        # Post-process: add summaries to tool results if long
        if "messages" in result_state:
            messages = list(result_state["messages"])
            new_messages = []
            
            # Get tool names from previous AIMessage for context
            tool_names_map = {}
            for msg in messages:
                if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_id = tc.get('id', tc.get('tool_call_id', ''))
                        tool_name = tc.get('name', 'unknown')
                        if tool_id:
                            tool_names_map[tool_id] = tool_name
            
            for msg in messages:
                if isinstance(msg, ToolMessage):
                    result_content = msg.content
                    result_len = len(str(result_content))
                    
                    # Get tool name from mapping, fallback to message attribute or "unknown"
                    tool_name = tool_names_map.get(msg.tool_call_id, getattr(msg, 'name', 'unknown'))
                    
                    # Check if user wants summarization
                    should_summarize = state.get("summarize_results", True)
                    
                    # If tool result >500 chars and summarization enabled, add summary
                    if should_summarize and result_len > 500:
                        try:
                            summary = summarize_tool_result({
                                "data": result_content,
                                "tool": tool_name
                            }, self.llm)
                            if summary:
                                # Append summary to tool result
                                enhanced_content = f"{result_content}\n\n📌 **Tóm tắt**: {summary}"
                                new_messages.append(ToolMessage(
                                    content=enhanced_content,
                                    tool_call_id=msg.tool_call_id,
                                    name=tool_name
                                ))
                            else:
                                new_messages.append(msg)
                        except Exception as e:
                            logger.warning(f"Tool result summarization skipped: {e}")
                            new_messages.append(msg)
                    elif should_summarize is False:
                        logger.info(f"📋 Tool result summarization disabled by user")
                        new_messages.append(msg)
                    else:
                        new_messages.append(msg)
                else:
                    new_messages.append(msg)
            
            result_state["messages"] = new_messages
        
        return result_state
    
    def _create_graph(self):
        """
        Tạo LangGraph workflow
        
        Returns:
            Compiled graph app
        """
        logger.info("Creating LangGraph workflow...")
        
        # Create state graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", self._tools_node)
        
        # Set entry point
        workflow.set_entry_point("agent")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "tools": "tools",
                "end": END
            }
        )
        
        # After tools, always go back to agent
        workflow.add_edge("tools", "agent")
        
        # Compile graph
        app = workflow.compile()
        
        logger.info("LangGraph workflow created successfully!")
        return app
    
    async def aquery(self, question: str, user_id: str = None, session_id: str = None, conversation_history: list = None, uploaded_files: list = None, rag_documents: list = None, allow_tools: bool = True, use_rag: bool = True, summarize_results: bool = True, observer_callback: callable = None) -> tuple:
        """
        Async query - Main entry point using simplified 2-node LangGraph workflow.
        
        ARCHITECTURE:
        The workflow handles ALL processing:
        1. File extraction (if files uploaded)
        2. File ingestion into vectordb (if files uploaded)
        3. RAG retrieval (if RAG enabled)
        4. Query classification and routing
        5. Tool selection and execution
        6. Response generation
        7. Output formatting
        
        Args:
            question: User's question (Vietnamese)
            user_id: User ID for context
            session_id: Session ID for context
            conversation_history: Previous messages (list of dicts with 'role' and 'content')
            uploaded_files: List of file metadata dicts (name, type, path, size, extension) for workflow
            rag_documents: Pre-retrieved RAG documents (optional, legacy support)
            allow_tools: Whether tools can be called
            use_rag: Whether RAG is enabled
            summarize_results: Whether to summarize tool outputs
            
        Returns:
            Tuple of (answer, thinking_steps) for display
        """
        try:
            logger.info(f"{'='*30}")
            logger.info(f"Processing question for user {user_id} in session {session_id}: {question}")
            logger.info(f"{'='*30}")
            
            # ========== PHASE 0: MINIMAL PREPROCESSING (OPTIONAL) ==========
            # Only handle query rewriting if it's a complex multi-part query
            # Most processing moves to workflow
            
            rewritten_question = question
            
            # Skip rewriting for greetings
            import re
            greeting_patterns = [
                r"^\s*(hello|hi|xin chào|chào|how are you|thanks|thank you|cảm ơn|goodbye|bye|tạm biệt|what'?s?\s+up|who are you|bạn là ai)\s*[\.\?\!]*\s*$",
                r"^(sao thế)\s*[\.\?\!]*\s*$"
            ]
            is_greeting = any(re.match(p, question.lower().strip(), re.IGNORECASE) for p in greeting_patterns)
            
            if not is_greeting:
                logger.info(f"Original query: {question}")
            else:
                logger.info(f"Query: {question} (greeting detected)")
            
            # ========== PHASE 1: BUILD MESSAGE HISTORY ==========
            
            # Convert conversation_history to LangChain message objects if needed
            from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
            
            # Build message history for the workflow
            workflow_messages = []
            
            if conversation_history:
                for msg in conversation_history:
                    if isinstance(msg, BaseMessage):
                        workflow_messages.append(msg)
                    elif isinstance(msg, dict):
                        role = msg.get('role', 'human')
                        content = msg.get('content', '')
                        if role == 'user' or role == 'human':
                            workflow_messages.append(HumanMessage(content=content))
                        elif role == 'assistant':
                            workflow_messages.append(AIMessage(content=content))
            
            # Add current question as HumanMessage
            workflow_messages.append(HumanMessage(content=question))
            
            # ========== PHASE 2: INVOKE WORKFLOW IMMEDIATELY ==========
            logger.info("="*50)
            logger.info("🚀 WORKFLOW INVOCATION (IMMEDIATE)")
            logger.info("="*50)
            logger.info(f"User prompt: {question}")
            logger.info(f"Files: {len(uploaded_files) if uploaded_files else 0}")
            logger.info(f"History messages: {len(workflow_messages)}")
            logger.info(f"RAG enabled: {use_rag}")
            logger.info("Note: All processing (file ingestion, RAG, tools) handled by workflow")
            
            # Determine which workflow version to use for this user
            from ..core.config import settings
            selected_version = settings.should_use_workflow_version(user_id)
            logger.info(f"📌 Workflow Selection:")
            logger.info(f"   User: {user_id}")
            logger.info(f"   Default: {self.workflow_version}")
            logger.info(f"   Canary %: {self.canary_rollout_percentage}%")
            logger.info(f"   Selected: {selected_version} ⭐")
            
            # Invoke the selected workflow
            if selected_version == "v4" and self.langgraph_workflow_v4:
                logger.info(f"➡️  Using V4 (18-node complete architecture)")
                logger.info(f"[INVOKE] Passing to workflow:")
                logger.info(f"[INVOKE]   - uploaded_files type: {type(uploaded_files)}")
                logger.info(f"[INVOKE]   - uploaded_files length: {len(uploaded_files) if uploaded_files else 0}")
                logger.info(f"[INVOKE]   - observer_callback: {'enabled' if observer_callback else 'disabled'}")
                if uploaded_files:
                    logger.info(f"[INVOKE]   - uploaded_files content: {uploaded_files}")
                final_state = await self.langgraph_workflow_v4.invoke(
                    user_prompt=question,
                    uploaded_files=uploaded_files or [],
                    conversation_history=workflow_messages,
                    user_id=user_id or "default",
                    session_id=session_id or "default",
                    use_rag=use_rag,
                    tools_enabled=allow_tools,
                    observer_callback=observer_callback
                )
            elif selected_version == "v3" and self.langgraph_workflow_v3:
                logger.info(f"\n➡️  Using V3 (8-node enhanced architecture)")
                final_state = await self.langgraph_workflow_v3.invoke(
                    user_prompt=question,
                    uploaded_files=uploaded_files or [],
                    conversation_history=workflow_messages,
                    user_id=user_id or "default",
                    session_id=session_id or "default",
                    use_rag=use_rag,
                    tools_enabled=allow_tools,
                    observer_callback=observer_callback
                )
            else:
                logger.info(f"\n➡️  Fallback to legacy 10-node (simplified)")
                final_state = await self.langgraph_workflow.invoke(
                    user_prompt=question,
                    uploaded_files=uploaded_files or [],
                    conversation_history=workflow_messages,
                    user_id=user_id or "default",
                    session_id=session_id or "default",
                    use_rag=use_rag,
                    tools_enabled=allow_tools,
                    observer_callback=observer_callback
                )
            
            # ========== PHASE 3: EXTRACT AND RETURN ANSWER ==========
            logger.info("="*30)
            logger.info("📝 EXTRACTING FINAL ANSWER")
            logger.info("="*30)
            
            answer = final_state.get("generated_answer", "")
            metadata = final_state.get("metadata", {})
            
            logger.info(f"Final message type: AIMessage")
            logger.info(f"Answer length: {len(answer)} chars")
            logger.info(f"Full Answer:")
            logger.info(f"{answer}")
            logger.info(f"✅ ANSWER READY FOR USER {user_id}")
            logger.info(f"   Final length: {len(answer)} chars")
            
            # Build thinking steps from workflow result
            thinking_steps = [
                {
                    "step": 1,
                    "title": "🔍 Query Processing",
                    "description": "Classified and processed query in workflow"
                },
                {
                    "step": 2,
                    "title": "📚 Document Retrieval",
                    "description": "Retrieved relevant documents via RAG (if enabled)"
                },
                {
                    "step": 3,
                    "title": "🤖 Processing with Agent",
                    "description": "Agent analyzed context and executed tools"
                },
                {
                    "step": 4,
                    "title": "✅ Answer Ready",
                    "description": "Generated final answer"
                }
            ]
            
            return answer, thinking_steps
            
        except Exception as e:
            logger.error(f"Error in aquery: {e}")
            import traceback
            traceback.print_exc()
            return f"Error processing query: {str(e)}", []
    
    async def aquery_legacy(self, question: str, user_id: str = None, session_id: str = None, conversation_history: list = None, rag_documents: list = None, allow_tools: bool = True, use_rag: bool = True, summarize_results: bool = True) -> tuple:
        """
        Legacy async query method (kept for backward compatibility).
        Uses the old-style self.app workflow.
        
        Args:
            question: Câu hỏi tiếng Việt
            user_id: User ID (optional, for context)
            session_id: Session ID (optional, for context)
            conversation_history: List of previous messages (optional)
            rag_documents: List of RAG document chunks to include in context (optional)
            allow_tools: Whether to allow tool calls (default: True)
            use_rag: Whether to use RAG documents (default: True)
            summarize_results: Whether to summarize tool results (default: True)
            
        Returns:
            Tuple of (answer, thinking_steps)
        """
        try:
            logger.info(f"Processing question for user {user_id} in session {session_id}: {question}")
            thinking_steps = []
            
            query_lower = question.lower()
            force_no_tools = any(kw in query_lower for kw in ["no tool", "dont use tool", "without tool", "không dùng tool", "không sử dụng tool"])
            force_tools = any(kw in query_lower for kw in ["use tool", "use this tool", "dùng tool", "sử dụng tool"])
            force_no_rag = any(kw in query_lower for kw in ["no rag", "dont use rag", "without rag", "không dùng rag"])
            
            if force_no_tools:
                allow_tools = False
                logger.info("🚫 User explicitly disabled tools")
            elif force_tools:
                allow_tools = True
                logger.info("✓ User explicitly enabled tools")
            
            if force_no_rag:
                use_rag = False
                rag_documents = None
                logger.info("🚫 User explicitly disabled RAG")
            
            # Step 1: Rewrite query with context
            thinking_steps.append({
                "step": 1,
                "title": "🔄 Rewriting Query",
                "description": "Analyzing conversation context..."
            })
            
            rewritten_question, rewrite_reason = await self.rewrite_query_with_context(question, conversation_history)
            if rewritten_question != question:
                thinking_steps[-1]["result"] = f"Query rewritten: '{rewritten_question[:80]}...'"
                logger.info(f"Query rewritten: {question} → {rewritten_question}")
            else:
                thinking_steps[-1]["result"] = "No context needed - using original query"
            
            # Step 2: Filter RAG results if available and use_rag enabled
            if rag_documents and use_rag:
                thinking_steps.append({
                    "step": 2,
                    "title": "🔍 Filtering Search Results",
                    "description": f"Evaluating {len(rag_documents)} retrieved documents..."
                })
                
                filtered_docs = self.filter_rag_results(rewritten_question, rag_documents)
                thinking_steps[-1]["result"] = f"Selected {len(filtered_docs)}/{len(rag_documents)} relevant documents"
                
                # Use only filtered docs
                if filtered_docs:
                    rag_documents = filtered_docs
                else:
                    thinking_steps[-1]["result"] = "No sufficiently relevant documents found, proceeding without RAG"
                    rag_documents = None
            elif rag_documents and not use_rag:
                logger.info("RAG disabled by user preference")
                rag_documents = None
            
            # Step 3: Prepare for workflow invocation
            thinking_steps.append({
                "step": 3,
                "title": "💭 Processing with Agent",
                "description": "Invoking LangGraph workflow..."
            })
            
            # Build message list with conversation history
            messages = []
            
            # Add previous messages from conversation history if provided (ONLY last 2-3 turns for context, not all)
            if conversation_history:
                # Keep only the last 2 exchanges (4 messages max: user, assistant, user, assistant)
                recent_history = conversation_history[-4:] if len(conversation_history) > 4 else conversation_history
                for msg in recent_history:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role == "assistant":
                        messages.append(AIMessage(content=content))
                    else:
                        messages.append(HumanMessage(content=content))
                logger.info(f"📚 CONVERSATION CONTEXT")
                logger.info(f"   Added {len(recent_history)} recent messages ({len(recent_history)//2} exchanges)")
            
            # Add current question (with RAG context if available and relevant)
            if rag_documents and len(rag_documents) > 0:
                # Format RAG documents as context
                rag_context = self._format_rag_context(rag_documents)
                enhanced_question = f"{rewritten_question}\n📚 Related Documents:\n{rag_context}"
                messages.append(HumanMessage(content=enhanced_question))
                logger.info(f"📖 RAG CONTEXT INTEGRATION")
                logger.info(f"   Attached {len(rag_documents)} relevant document(s) to question")
                logger.info(f"   Enhanced question length: {len(enhanced_question)} chars")
                logger.info(f"   RAG section preview:")
                for doc in rag_documents[:2]:  # Show first 2 docs
                    logger.info(f"     • {doc.get('title', 'Unknown')} (relevance: {doc.get('similarity', 0):.1%})")
            else:
                messages.append(HumanMessage(content=rewritten_question))
                if rag_documents is not None:
                    logger.info(f"\n⚠️  NO RAG DOCUMENTS")
                    logger.info(f"   RAG was enabled but no documents matched relevance threshold")
            
            has_rag = any("📚 Related Documents" in str(msg.content) for msg in messages if hasattr(msg, 'content'))
            
            # Step 3: Prepare initial state
            initial_state = {
                "messages": messages,
                "allow_tools": allow_tools,
                "has_rag_context": has_rag,
                "summarize_results": summarize_results,
                "_rag_documents": rag_documents if rag_documents else []
            }
            
            logger.info("="*30)
            logger.info("🚀 INVOKING LANGGRAPH WORKFLOW")
            logger.info("="*30)
            result = await self.app.ainvoke(initial_state)
            logger.info("="*30)
            logger.info("✅ LANGGRAPH WORKFLOW COMPLETED")
            logger.info("="*30)
            
            tool_calls_made = []
            for msg in result.get("messages", []):
                if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        tool_name = tool_call.get('name', 'unknown') if isinstance(tool_call, dict) else getattr(tool_call, 'name', 'unknown')
                        tool_calls_made.append(tool_name)
            
            # Update workflow step with results
            if tool_calls_made:
                unique_tools = list(set(tool_calls_made))
                thinking_steps[-1]["result"] = f"Used {len(unique_tools)} tool(s): {', '.join(unique_tools)}"
                thinking_steps[-1]["title"] = "⚙️  Agent Execution (with Tools)"
            else:
                thinking_steps[-1]["result"] = "Answered from knowledge without tools"
                thinking_steps[-1]["title"] = "⚙️  Agent Execution"
            
            # Add final answer generation step
            thinking_steps.append({
                "step": 4,
                "title": "✅ Answer Ready",
                "description": "Formatted and validated response..."
            })
            
            # Get final answer
            logger.info("="*30)
            logger.info("📝 EXTRACTING FINAL ANSWER")
            logger.info("="*30)
            final_message = result["messages"][-1]
            final_msg_type = type(final_message).__name__
            logger.info(f"Final message type: {final_msg_type}")
            
            content = final_message.content if hasattr(final_message, 'content') else str(final_message)
            
            # Handle case where content is a list of content blocks
            if isinstance(content, list):
                # Extract text from content blocks
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get('type') == 'text':
                        text_parts.append(block.get('text', ''))
                    elif isinstance(block, str):
                        text_parts.append(block)
                answer = ''.join(text_parts)
                logger.info(f"Extracted from {len(content)} content blocks")
            else:
                answer = str(content)
            
            logger.info(f"Answer length: {len(answer)} chars")
            logger.info(f"Answer preview: {answer[:150]}...")
            
            # Clean up JSON responses - if answer looks like JSON, convert to natural text
            answer = self._clean_json_response(answer)
            
            thinking_steps[-1]["result"] = "✅ Answer generated successfully"
            
            logger.info(f"\n✅ ANSWER READY FOR USER {user_id}")
            logger.info(f"   Final length: {len(answer)} chars")
            return answer, thinking_steps
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            error_steps = [
                {"step": 1, "title": "❌ Error", "result": str(e)}
            ]
            return f"Xin lỗi, đã có lỗi xảy ra: {str(e)}", error_steps
    
    def _format_rag_context(self, documents: list) -> str:
        """
        Format RAG documents for inclusion in the question
        
        Args:
            documents: List of document chunks
            
        Returns:
            Formatted context string (content-focused, without filenames that trigger tools)
        """
        context_parts = []
        for i, doc in enumerate(documents, 1):
            title = doc.get('title', 'Unknown')
            text = doc.get('text', '')
            similarity = doc.get('similarity', 0)
            # Note: Intentionally NOT including 'source' field to prevent LLM from 
            # thinking it should call analyze_excel_file or other file tools
            
            # Limit text length
            text_preview = text[:300] + "..." if len(text) > 300 else text
            
            context_parts.append(
                f"{i}. [{title}] (Relevance: {similarity:.1%})\n"
                f"   {text_preview}\n"
            )
        
        return "\n".join(context_parts)
    
    async def _merge_rag_and_tool_results(self, rag_docs: list, tool_result: str, original_question: str) -> str:
        """
        Merge RAG documents with tool execution results for comprehensive answer.
        
        Args:
            rag_docs: List of RAG documents
            tool_result: String result from tool execution
            original_question: Original user question
            
        Returns:
            Merged response combining both sources
        """
        from langchain_core.prompts import PromptTemplate
        
        rag_context = self._format_rag_context(rag_docs)
        
        merge_template = """Bạn vừa nhận được thông tin từ hai nguồn:

1. NGHIÊN CỨU TÀI LIỆU (từ cơ sở dữ liệu):
{rag_context}

2. DỮ LIỆU THỜI GIAN THỰC (từ công cụ):
{tool_result}

NHIỆM VỤ: Viết câu trả lời cân bằng kết hợp cả hai nguồn:
- Nêu phân tích từ tài liệu trước (bối cảnh, khái niệm, xu hướng)
- Rồi đưa dữ liệu thời gian thực từ công cụ (số liệu cụ thể, chỉ số)
- Kết luận: So sánh hay nhận xét từ cả hai

Câu hỏi gốc: {original_question}

Viết câu trả lời thực tế, không lặp lại "dữ liệu bao gồm..." hay "kết quả trả về..."
"""
        
        prompt = PromptTemplate(
            template=merge_template,
            input_variables=["rag_context", "tool_result", "original_question"]
        )
        
        chain = prompt | self.llm
        response = await chain.ainvoke({
            "rag_context": rag_context,
            "tool_result": tool_result,
            "original_question": original_question
        })
        
        merged_answer = response.content if hasattr(response, 'content') else str(response)
        logger.info(f"📊 MERGED RESULT: Combined {len(rag_docs)} RAG docs + tool result")
        
        return merged_answer
    
    def _clean_json_response(self, answer: str) -> str:
        """
        Detect and clean JSON-formatted responses or explanations of JSON structure.
        If answer is explaining JSON instead of showing a table, try to extract and 
        reformat the data.
        
        Args:
            answer: Raw answer from LLM
            
        Returns:
            Cleaned answer (natural language or original if not JSON)
        """
        import json
        import re
        
        answer = answer.strip()
        
        # Check if answer looks like it's explaining JSON structure or data format
        json_explanation_keywords = [
            "đối tượng json",
            "cấu trúc json", 
            "json chứa",
            "mảng dữ liệu",
            "từng phần tử",
            "bản ghi dữ liệu",
            "thử viện json",
            "json parsing",
            "json.loads",
            "để sử dụng dữ liệu",
            "duyệt qua mảng",
            "dữ liệu này bao gồm",
            "dữ liệu bao gồm",
            "có thể được sử dụng",
            "để phân tích",
            "dữ liệu này chứa",
            "dữ liệu này có",
            "mảng chứa",
            "bao gồm các khóa",
            "khóa như",
        ]
        
        # Check if answer is explaining how to use a tool instead of showing results
        tool_explanation_keywords = [
            "sửa đổi mã",
            "cách sửa",
            "thêm tham số",
            "cần đảm bảo",
            "để giải quyết",
            "def get_",
            "officers = get_officers",
            "để gọi công cụ",
            "trong ví dụ trên",
            "hãy cho tôi biết để",
            "nếu bạn vẫn gặp",
            "chương trình python",
            "mã nguồn của chương trình",
            "import requests",
            "import json",
        ]
        
        answer_lower = answer.lower()
        is_json_explanation = any(keyword in answer_lower for keyword in json_explanation_keywords)
        is_tool_explanation = any(keyword in answer_lower for keyword in tool_explanation_keywords)
        
        if is_json_explanation or is_tool_explanation:
            logger.error(f"❌ CRITICAL: JSON/Tool explanation detected instead of results: {answer[:100]}")
            logger.error(f"❌ This means the LLM ignored the system prompt instructions!")
            
            if is_tool_explanation:
                logger.error(f"❌ Agent is explaining tool usage instead of showing results")
            
            # Try to find JSON data embedded in the explanation text
            json_patterns = [
                r'\{[^{}]*"success"[^{}]*\}',  # JSON with success field
                r'\[\s*\{[^}]*\}\s*(?:,\s*\{[^}]*\})*\s*\]',  # Array of objects
                r'\{[^{}]*\}',  # Any JSON object
            ]
            
            json_matches = []
            for pattern in json_patterns:
                json_matches = re.findall(pattern, answer)
                if json_matches:
                    break
            
            if json_matches:
                for json_str in json_matches:
                    try:
                        # Try to parse each match
                        data = json.loads(json_str)
                        logger.error(f"✓ Found and extracted JSON data from explanation")
                        
                        # Format the extracted data as a proper response
                        formatted = self._format_tool_result(data)
                        if formatted and formatted != answer:
                            logger.error(f"✓ Replacing explanation with formatted result")
                            return formatted
                    except json.JSONDecodeError as e:
                        logger.warning(f"Could not parse JSON: {str(e)[:100]}")
                        continue
            
            # If we couldn't extract JSON, just return the answer as-is and flag the issue
            logger.error(f"⚠️ Could not extract JSON from explanation - returning answer as-is")
            logger.error(f"⚠️ System prompt failed to prevent tool explanation. LLM returned explanation instead of results.")
            return answer
        
        # Check if answer looks like JSON (starts with { or [)
        if answer.startswith('{') or answer.startswith('['):
            try:
                parsed = json.loads(answer)
                
                # If it's a simple object like {"name": "hello"...}, extract values
                if isinstance(parsed, dict):
                    # Try to extract a meaningful message
                    for key in ['content', 'message', 'text', 'answer', 'response']:
                        if key in parsed and isinstance(parsed[key], str):
                            return parsed[key]
                    
                    # If no standard key, return a default message
                    logger.warning(f"JSON response detected and cleaned: {answer[:100]}")
                    return "Tôi là trợ lý tư vấn tài chính. Bạn có câu hỏi gì về chứng khoán Việt Nam không?"
                
                return answer
            except json.JSONDecodeError:
                # Not valid JSON, return as-is
                return answer
        
        return answer
    
    def _format_tool_result(self, data: dict) -> str:
        """
        Format extracted tool result as a proper Markdown table response.
        This is a fallback method to convert JSON data to table format
        when the LLM ignores formatting instructions.
        
        Args:
            data: Extracted tool result data
            
        Returns:
            Formatted Markdown table or message
        """
        try:
            # Check if it's an RSI result
            if "indicator" in data and data.get("indicator", "").startswith("RSI"):
                if "detailed_data" in data and isinstance(data["detailed_data"], list):
                    logger.info(f"Formatting RSI result as table")
                    rows = ["|Ngày|Giá đóng cửa|RSI|Trạng thái|"]
                    rows.append("|---|---|---|---|")
                    for item in data["detailed_data"][:10]:  # Limit to 10 rows
                        date = item.get("date", "")
                        close = item.get("close", "")
                        rsi = item.get("rsi_14", "")
                        status = item.get("status", "")
                        rows.append(f"|{date}|{close}|{rsi}|{status}|")
                    
                    # Add summary
                    summary = "\n"
                    if "analysis" in data:
                        analysis = data["analysis"]
                        if "status" in analysis:
                            status_text = analysis["status"]
                            if status_text == "OVERBOUGHT":
                                summary += "**Nhận xét**: Cổ phiếu đang ở vùng quá mua (RSI > 70), có thể điều chỉnh trong ngắn hạn."
                            elif status_text == "OVERSOLD":
                                summary += "**Nhận xét**: Cổ phiếu đang ở vùng quá bán (RSI < 30), có cơ hội phục hồi."
                            else:
                                summary += "**Nhận xét**: Cổ phiếu đang ở vùng trung tính."
                    
                    return "\n".join(rows) + summary
            
            # Check if it's a SMA result
            if "indicator" in data and data.get("indicator", "").startswith("SMA"):
                if "detailed_data" in data and isinstance(data["detailed_data"], list):
                    logger.info(f"Formatting SMA result as table")
                    rows = ["|Ngày|Giá đóng cửa|SMA|Chênh lệch|% Chênh lệch|"]
                    rows.append("|---|---|---|---|---|")
                    for item in data["detailed_data"][:10]:  # Limit to 10 rows
                        date = item.get("date", "")
                        close = item.get("close", "")
                        sma_key = [k for k in item.keys() if k.startswith("sma_")]
                        sma_val = item.get(sma_key[0], "") if sma_key else ""
                        diff = item.get("difference", "")
                        diff_pct = item.get("difference_percent", "")
                        rows.append(f"|{date}|{close}|{sma_val}|{diff}|{diff_pct}|")
                    
                    # Add summary
                    summary = "\n"
                    if "analysis" in data:
                        analysis = data["analysis"]
                        trend = analysis.get("trend", "")
                        if trend:
                            summary += f"**Nhận xét**: {trend}"
                    
                    return "\n".join(rows) + summary
            
            # Check if it's a historical data result (detailed_data key)
            if "detailed_data" in data and isinstance(data.get("detailed_data"), list):
                logger.info(f"Formatting historical data result as table")
                rows = ["|Ngày|Mở|Cao|Thấp|Đóng|Khối lượng|"]
                rows.append("|---|---|---|---|---|---|")
                for item in data["detailed_data"][:10]:  # Limit to 10 rows
                    if isinstance(item, dict):
                        date = item.get("date", "")
                        open_p = item.get("open", "")
                        high = item.get("high", "")
                        low = item.get("low", "")
                        close = item.get("close", "")
                        volume = item.get("volume", "")
                        rows.append(f"|{date}|{open_p}|{high}|{low}|{close}|{volume}|")
                
                return "\n".join(rows)
            
            # Check if it's a historical data result (data key - older format)
            if "data" in data and isinstance(data.get("data"), list):
                logger.info(f"Formatting historical data result (old format) as table")
                rows = ["|Ngày|Mở|Cao|Thấp|Đóng|Khối lượng|"]
                rows.append("|---|---|---|---|---|---|")
                for item in data["data"][:10]:  # Limit to 10 rows
                    if isinstance(item, dict):
                        date = item.get("date", "")
                        open_p = item.get("open", "")
                        high = item.get("high", "")
                        low = item.get("low", "")
                        close = item.get("close", "")
                        volume = item.get("volume", "")
                        rows.append(f"|{date}|{open_p}|{high}|{low}|{close}|{volume}|")
                
                return "\n".join(rows)
            
            # For other data structures, try generic formatting
            logger.warning(f"Could not identify specific tool result format, returning generic response")
            return f"Kết quả từ công cụ được trả về nhưng định dạng không được nhận diện. Vui lòng thử lại."
            
        except Exception as e:
            logger.error(f"Error formatting tool result: {str(e)}")
            return f"Lỗi khi định dạng kết quả: {str(e)}"
    
    def _validate_response_is_not_explanation(self, response: str) -> bool:
        """
        Validate that the response is actual formatted data, not an explanation of JSON structure.
        
        Args:
            response: The response from LLM
            
        Returns:
            True if response contains actual data/table, False if it's explaining structure
        """
        import re
        
        response_lower = response.lower()
        
        # Keywords that indicate explanation instead of data display
        bad_keywords = [
            "đối tượng json",
            "cấu trúc json", 
            "json chứa",
            "mảng dữ liệu",
            "từng phần tử",
            "bản ghi dữ liệu",
            "json.loads",
            "json parsing",
            "dữ liệu này chứa",
            "dữ liệu bao gồm",
            "có các trường",
            "các khóa",
            "mảng chứa",
            "đây là một",
            "kết quả là một",
            "để sử dụng",
            "duyệt qua",
        ]
        
        # Check if response contains explanation keywords
        for keyword in bad_keywords:
            if keyword in response_lower:
                logger.error(f"❌ Found explanation keyword: '{keyword}'")
                return False
        
        # Check if response contains actual table markdown (valid response)
        if re.search(r'\|\s*[A-Za-z0-9_\u0080-\uffff\s]+\s*\|', response):
            logger.info(f"✓ Response contains Markdown table - likely valid")
            return True
        
        # Check if response starts with common explanation patterns
        first_sentence = response.split('\n')[0].strip() if response else ""
        explanation_starters = [
            "dữ liệu này",
            "kết quả trả",
            "để ",
            "ngoài ra",
            "theo như",
        ]
        
        for starter in explanation_starters:
            if first_sentence.lower().startswith(starter):
                logger.error(f"❌ Response starts with explanation pattern: '{starter}'")
                return False
        
        # If response is very short and doesn't contain meaningful data, it's likely invalid
        if len(response.strip()) < 50 and not re.search(r'\d{4}-\d{2}', response):
            logger.warning(f"⚠️ Response is very short and contains no date patterns")
            # Could be valid for some types of responses, so don't fail here
        
        logger.info(f"✓ Response appears to be valid formatted data")
        return True
    
    async def _extract_and_format_tool_data(self, tool_content: str) -> str:
        """
        Fallback: Extract JSON data from tool result and format it as a proper table.
        This is used when the LLM fails to format the data correctly.
        
        Args:
            tool_content: Raw tool message content (usually JSON string)
            
        Returns:
            Formatted Markdown table or empty string if extraction fails
        """
        import json
        
        logger.info("🔧 Attempting to extract and format tool data directly...")
        
        try:
            # Try to parse the tool content as JSON
            if isinstance(tool_content, str):
                # First try direct JSON parse
                try:
                    data = json.loads(tool_content)
                except json.JSONDecodeError:
                    # If not valid JSON, try to extract JSON from the string
                    import re
                    json_match = re.search(r'\{.*\}', tool_content, re.DOTALL)
                    if json_match:
                        data = json.loads(json_match.group())
                    else:
                        logger.error(f"Could not find JSON in tool content")
                        return ""
            else:
                data = tool_content
            
            # Format based on data type
            formatted = self._format_tool_result(data)
            return formatted
            
        except Exception as e:
            logger.error(f"Failed to extract and format tool data: {e}")
            return ""
    
    async def rewrite_query_with_context(self, question: str, conversation_history: list = None) -> tuple:
        """
        Rewrite user query with context from conversation history.
        
        CRITICAL GUARDS:
        1. Max 1 rewrite per query (tracked by rewrite_count in state)
        2. Skip if filename found in query
        3. Only rewrite if ambiguous AND conversation context available
        4. Limit history to last 2 exchanges (4 messages) to prevent subject drift
        
        Args:
            question: Original user question
            conversation_history: List of previous messages (will limit to last 4)
            
        Returns:
            Tuple of (rewritten_query, reasoning)
        """
        logger.info("="*30)
        logger.info("🔄 QUERY REWRITING PHASE")
        logger.info("="*30)
        logger.info(f"Original query: {question}")
        
        # GUARD 1: No history available
        if not conversation_history or len(conversation_history) == 0:
            logger.info("⛔ No conversation history - using original query")
            return question, "No conversation history"
        
        # GUARD 2: Check if query is clear and specific (no ambiguous pronouns)
        is_clear = self._is_query_clear(question)
        if is_clear:
            logger.info("✓ Query is clear and specific - no rewriting needed")
            return question, "Query clear and specific"
        
        # GUARD 3: Limit history to last 2 exchanges (4 messages) to prevent subject drift
        recent_context = conversation_history[-4:] if conversation_history else []
        
        if not recent_context:
            logger.info("⛔ Insufficient conversation history - using original query")
            return question, "Insufficient history"
        
        logger.info(f"🤔 Ambiguous reference detected - attempting clarification...")
        logger.info(f"Using last {len(recent_context)} messages from history")
        
        # Format context safely
        context_str = ""
        for msg in recent_context:
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "")[:100]
            context_str += f"{role}: {content}\n"
        
        logger.info(f"History context:\n{context_str[:200]}...")
        
        rewrite_prompt = f"""Based on the recent conversation, clarify what the ambiguous reference refers to.

Recent Conversation:
{context_str}

User's New Query: {question}

Rules:
1. Only clarify ambiguous pronouns (it, this, that, they, etc.)
2. Keep the query as close to original as possible
3. Replace ambiguous words with what they refer to
4. Respond ONLY with the clarified query (no explanation)

Clarified Query:"""
        
        try:
            # Use sync method wrapped in async
            import asyncio
            response = await asyncio.to_thread(
                self.llm.invoke,
                rewrite_prompt
            )
            rewritten = response.content if hasattr(response, 'content') else str(response)
            rewritten = rewritten.strip()
            
            logger.info(f"✓ Query rewritten successfully:")
            logger.info(f"  Before: {question[:60]}...")
            logger.info(f"  After:  {rewritten[:60]}...")
            return rewritten, "Query clarified with context"
        except Exception as e:
            logger.warning(f"❌ Query rewrite failed: {e}")
            logger.info("   → Proceeding with original query")
            return question, f"Rewrite failed: {str(e)[:50]}"
    
    async def _rewrite_query_if_needed(self, state: dict) -> dict:
        """
        Rewrite query ONCE if ambiguous, with guards:
        1. Skip if filename already present in query
        2. Skip if rewrite_count >= 1 (max 1 rewrite per query)
        3. Limit history to last 2 exchanges (4 messages)
        """
        user_prompt = state.get("user_prompt", "")
        uploaded_files = state.get("uploaded_files", [])
        conversation_history = state.get("conversation_history", [])
        rewrite_count = state.get("rewrite_count", 0)
        
        logger.info("====================")
        logger.info("🔄 QUERY REWRITING PHASE")
        logger.info("====================")
        logger.info(f"Original query: {user_prompt[:80]}")
        
        # GUARD 1: Check if filename already in query
        has_filename_in_query = False
        if uploaded_files:
            first_file = uploaded_files[0].get("filename", "")
            if first_file and first_file in user_prompt:
                has_filename_in_query = True
                logger.info(f"✓ Filename found in query, skipping rewrite")
        
        # GUARD 2: Check rewrite count
        if rewrite_count >= 1:
            logger.info(f"⚠️  Rewrite limit reached ({rewrite_count}/1), skipping")
            return state
        
        # GUARD 3: Check if query is clear/specific
        is_clear = self._is_query_clear(user_prompt)
        if is_clear and (has_filename_in_query or not uploaded_files):
            logger.info("✓ Query is clear and specific - no rewriting needed")
            state["rewritten_prompt"] = user_prompt
            state["rewrite_count"] = 0
            return state
        
        # GUARD 4: Only rewrite if ambiguous AND needs context
        if not has_filename_in_query and uploaded_files:
            # Limit history to last 2 exchanges (4 messages)
            recent_history = conversation_history[-4:] if conversation_history else []
            
            logger.info(f"🤔 Ambiguous reference detected - attempting clarification...")
            logger.info(f"Context from history: {len(recent_history)} recent messages")
            
            rewritten = await self._call_rewrite_agent(user_prompt, recent_history, uploaded_files)
            state["rewritten_prompt"] = rewritten
            state["rewrite_count"] = 1
            
            logger.info(f"✓ Query rewritten successfully:")
            logger.info(f"  Before: {user_prompt[:60]}...")
            logger.info(f"  After:  {rewritten[:60]}...")
        else:
            state["rewritten_prompt"] = user_prompt
            state["rewrite_count"] = 0
        
        return state
    
    def _is_query_clear(self, query: str) -> bool:
        """
        Detect if query is clear/specific or ambiguous.
        Clear: contains specific company/metric names or clear intent
        Ambiguous: mostly pronouns/vague terms without specific subject
        
        Rules:
        - If query contains ambiguous pronouns alone → AMBIGUOUS
        - If > 30% ambiguous terms → AMBIGUOUS  
        - Otherwise → CLEAR
        """
        ambiguous_terms = ["this", "that", "it", "it's", "summarize", "analyze", "explain", 
                          "what is it", "what about it", "tell me about it"]
        
        lowercase = query.lower()
        
        # Direct matches for common ambiguous patterns
        for term in ambiguous_terms:
            if term in lowercase:
                # Count total significant words
                words = [w for w in lowercase.split() if len(w) > 2]  # Ignore short words
                
                # If query is mostly just ambiguous pattern
                if len(words) <= 3 and (term in lowercase):
                    return False  # Ambiguous: mostly pronouns
        
        # Check proportion of ambiguous terms (lower threshold)
        ambiguous_count = sum(1 for term in ambiguous_terms if term in lowercase)
        words = [w for w in lowercase.split() if len(w) > 2]
        
        if len(words) > 0 and (ambiguous_count / len(words)) > 0.30:
            return False  # More than 30% ambiguous
        
        return True  # Clear and specific

    async def _call_rewrite_agent(
        self, 
        query: str, 
        history: List[dict], 
        uploaded_files: List[dict]
    ) -> str:
        """Rewrite query using LLM with limited history context."""
        first_file = uploaded_files[0].get("filename", "") if uploaded_files else ""
        
        history_text = ""
        if history:
            history_text = "\n".join([
                f"User: {h.get('content', '')[:100]}" if h.get('role') == 'user' 
                else f"Assistant: {h.get('content', '')[:100]}"
                for h in history
            ])
        
        prompt = f"""You are a query clarification agent. The user uploaded a file: {first_file}

Recent conversation:
{history_text or "(No recent conversation)"}

User's current ambiguous query: {query}

Your task: Clarify what the user is asking about regarding the uploaded file.
Return ONLY the clarified query (1-2 sentences), nothing else.

Clarified query:"""
    
        response = await self.llm.ainvoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)

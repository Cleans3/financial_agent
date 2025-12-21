"""
Financial Agent - Agent chính sử dụng LangGraph và ReAct pattern
"""

import logging
import os
from typing import Literal
from pathlib import Path

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from .state import AgentState
from ..llm import LLMFactory
from ..tools import get_all_tools

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
    
    def __init__(self):
        """Initialize Financial Agent"""
        logger.info("Initializing Financial Agent...")
        
        # Get LLM from factory
        self.llm = LLMFactory.get_llm()
        
        # Get all tools
        self.tools = get_all_tools()
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
        
        # Create workflow graph
        self.app = self._create_graph()
        
        logger.info("Financial Agent initialized successfully!")
    
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
        logger.info("--- AGENT NODE: Analyzing query and selecting tools ---")
        
        # Log the incoming message state for debugging
        if state["messages"]:
            last_msg = state["messages"][-1]
            msg_type = type(last_msg).__name__
            msg_preview = str(last_msg.content)[:80] if hasattr(last_msg, 'content') else str(last_msg)[:80]
            logger.info(f"Last message ({msg_type}): {msg_preview}")
            
            # If the last message is a ToolMessage (tool result), don't call tools again
            # Just process the result and generate final answer
            if isinstance(last_msg, ToolMessage):
                logger.info("--- AGENT: Received tool result, generating final answer based on tool output ---")
                
                # Check if the tool message contains an error
                tool_content = str(last_msg.content).lower()
                if "error" in tool_content or last_msg.content.startswith("Error"):
                    logger.warning(f"Tool execution failed - Full error: {last_msg.content}")
                    # Don't ask LLM to process error - just return it directly
                    error_response = AIMessage(
                        content=f"Xin lỗi, công cụ gặp lỗi: {last_msg.content}\n\nVui lòng thử lại với các tham số khác hoặc kiểm tra mã chứng khoán."
                    )
                    return {"messages": state["messages"] + [error_response]}
                
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

❌ VÍ DỤ SAI (KHÔNG PHẢI TRẢ LỜI NHƯ DƯỚI ĐÂY):
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
                
                # Create AIMessage with the processed result
                final_response = AIMessage(content=content)
                return {"messages": state["messages"] + [final_response]}
        
        # Check if the last message is a simple greeting/non-financial query
        last_message = state["messages"][-1] if state["messages"] else None
        if last_message and hasattr(last_message, 'content'):
            query_text = str(last_message.content).lower().strip()
            
            # List of simple greetings and non-financial queries that shouldn't trigger tool calls
            greetings = [
                "hello", "hi", "hey", "xin chào", "chào", "chào bạn",
                "how are you", "bạn khỏe không", "như thế nào",
                "thanks", "cảm ơn", "thank you", "thank",
                "help me", "giúp tôi", "hỗ trợ",
                "ok", "okay", "được", "tốt"
            ]
            
            # Check if query is just a greeting
            is_greeting = any(query_text.startswith(g) or query_text == g for g in greetings)
            
            if is_greeting:
                logger.info(f"--- AGENT: Detected greeting query, responding conversationally without tools ---")
                conversational_response = AIMessage(
                    content="Tôi là trợ lý tư vấn tài chính. Bạn có câu hỏi gì về chứng khoán Việt Nam không? Tôi có thể giúp bạn với: thông tin công ty, dữ liệu giá cổ phiếu, phân tích kỹ thuật (SMA, RSI), thông tin cổ đông, ban lãnh đạo, v.v."
                )
                return {"messages": state["messages"] + [conversational_response]}
        
        # Prepare prompt with tool descriptions
        tool_descriptions = self._get_tool_descriptions()
        # Avoid using str.format because the system prompt may contain braces
        # (e.g. examples or conditional snippets). Use a safe replacement for
        # the single placeholder {tool_descriptions} only.
        system_text = self.system_prompt.replace("{tool_descriptions}", tool_descriptions)
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_text),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        # Bind tools to LLM
        llm_with_tools = self.llm.bind_tools(self.tools)
        chain = prompt | llm_with_tools
        
        try:
            # Invoke LLM
            response = await chain.ainvoke({"messages": state["messages"]})
            
            # Log tool calls if any
            if hasattr(response, 'tool_calls') and response.tool_calls:
                tool_names = [tc.get('name', 'unknown') for tc in response.tool_calls]
                logger.info(f"--- AGENT: Tool calls detected: {tool_names} ---")
            else:
                logger.info("--- AGENT: No tool calls, generating final answer ---")
            
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
            logger.info("--- DECISION: No messages, ending ---")
            return "end"
        
        # Check if last message has tool calls
        if isinstance(last_message, AIMessage) and hasattr(last_message, "tool_calls") and last_message.tool_calls:
            logger.info("--- DECISION: Tool calls found, going to tools ---")
            return "tools"
        
        logger.info("--- DECISION: No tool calls, ending ---")
        return "end"
    
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
        workflow.add_node("tools", ToolNode(self.tools))
        
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
    
    async def aquery(self, question: str, user_id: str = None, session_id: str = None, conversation_history: list = None, rag_documents: list = None) -> tuple:
        """
        Async query - Xử lý câu hỏi của người dùng with Agentic RAG
        
        Args:
            question: Câu hỏi tiếng Việt
            user_id: User ID (optional, for context)
            session_id: Session ID (optional, for context)
            conversation_history: List of previous messages (optional)
            rag_documents: List of RAG document chunks to include in context (optional)
            
        Returns:
            Tuple of (answer, thinking_steps)
            - answer: Câu trả lời từ agent
            - thinking_steps: List of reasoning steps for visualization
        """
        try:
            logger.info(f"Processing question for user {user_id} in session {session_id}: {question}")
            thinking_steps = []
            
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
            
            # Step 2: Filter RAG results if available
            if rag_documents:
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
            
            # Step 3: Generate answer
            thinking_steps.append({
                "step": 3,
                "title": "💭 Generating Answer",
                "description": "Calling LLM with optimized query..."
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
                logger.info(f"Added {len(recent_history)} recent messages for context")
            
            # Add current question (with RAG context if available and relevant)
            if rag_documents and len(rag_documents) > 0:
                # Format RAG documents as context
                rag_context = self._format_rag_context(rag_documents)
                enhanced_question = f"{rewritten_question}\n\n📚 Related Documents:\n{rag_context}"
                messages.append(HumanMessage(content=enhanced_question))
                logger.info(f"Enhanced question with {len(rag_documents)} relevant RAG documents")
            else:
                messages.append(HumanMessage(content=rewritten_question))
            
            # Create initial state with conversation context
            initial_state = {
                "messages": messages
            }
            
            # Run graph
            result = await self.app.ainvoke(initial_state)
            
            # Get final answer
            final_message = result["messages"][-1]
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
            else:
                answer = str(content)
            
            # Clean up JSON responses - if answer looks like JSON, convert to natural text
            answer = self._clean_json_response(answer)
            
            thinking_steps[-1]["result"] = "✅ Answer generated successfully"
            
            logger.info(f"Answer generated for user {user_id}: {answer[:100]}...")
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
            Formatted context string
        """
        context_parts = []
        for i, doc in enumerate(documents, 1):
            title = doc.get('title', 'Unknown')
            text = doc.get('text', '')
            similarity = doc.get('similarity', 0)
            source = doc.get('source', '')
            
            # Limit text length
            text_preview = text[:300] + "..." if len(text) > 300 else text
            
            context_parts.append(
                f"{i}. [{title}] (Relevance: {similarity:.1%})\n"
                f"   Source: {source}\n"
                f"   {text_preview}\n"
            )
        
        return "\n".join(context_parts)
    
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
    
    async def rewrite_query_with_context(self, question: str, conversation_history: list = None) -> tuple:
        """
        Rewrite user query with context from conversation history
        Only rewrites if query has ambiguous references that need clarification
        
        Args:
            question: Original user question
            conversation_history: List of previous messages
            
        Returns:
            Tuple of (rewritten_query, reasoning)
        """
        if not conversation_history or len(conversation_history) == 0:
            return question, "No conversation history - using original query"
        
        # Check if query needs context (contains ambiguous references)
        ambiguous_words = ["it", "this", "that", "it's", "they", "them", "those", "these", "nó", "cái này", "cái kia", "nó là", "họ"]
        has_ambiguous_ref = any(word.lower() in question.lower() for word in ambiguous_words)
        
        # If query is clear and specific, don't rewrite it
        if not has_ambiguous_ref:
            return question, "Query is clear and specific - no rewriting needed"
        
        # Get last turn for context (user + assistant)
        recent_context = []
        for msg in conversation_history[-2:]:  # Only last user-assistant pair
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "")[:100]  # Limit length
            recent_context.append(f"{role}: {content}")
        
        context_str = "\n".join(recent_context)
        
        rewrite_prompt = f"""Based on the previous message, clarify what the ambiguous reference refers to.

Previous Context:
{context_str}

New Query: {question}

Rules:
1. Only clarify ambiguous pronouns (it, this, that, etc.)
2. Keep the query as close to original as possible
3. Just replace the ambiguous word with what it refers to
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
            return rewritten.strip(), "Query clarified with context"
        except Exception as e:
            logger.warning(f"Query rewrite failed: {e}")
            return question, f"Rewrite skipped due to error"
    
    def filter_rag_results(self, question: str, documents: list, min_relevance: float = 0.4) -> list:
        """
        Filter RAG results based on relevance to the question
        
        Args:
            question: User question
            documents: List of retrieved documents with similarity scores
            min_relevance: Minimum relevance threshold (0-1)
            
        Returns:
            Filtered documents, or empty list if results are not relevant
        """
        if not documents:
            return []
        
        # Filter by similarity threshold
        filtered = [doc for doc in documents if doc.get('similarity', 0) >= min_relevance]
        
        # Additional check: if top result similarity is very low, filter all
        if documents and documents[0].get('similarity', 0) < 0.3:
            logger.info(f"Top result similarity too low ({documents[0].get('similarity', 0):.1%}), excluding all documents")
            return []
        
        if len(filtered) > len(documents) * 0.5:  # Keep if at least 50% are relevant
            logger.info(f"Filtered RAG results: {len(documents)} → {len(filtered)} relevant documents")
            return filtered
        else:
            logger.info(f"Too few relevant results ({len(filtered)}/{len(documents)}), using all or none")
            return [] if len(filtered) < 2 else filtered
    
    def query(self, question: str, user_id: str = None, session_id: str = None, conversation_history: list = None, rag_documents: list = None) -> str:
        """
        Sync query - Wrapper for async query
        
        Args:
            question: Câu hỏi tiếng Việt
            user_id: User ID (optional, for context)
            session_id: Session ID (optional, for context)
            conversation_history: List of previous messages (optional)
            rag_documents: List of RAG document chunks (optional)
            
        Returns:
            Câu trả lời từ agent (only answer, not thinking steps)
        """
        import asyncio
        
        try:
            # Get or create event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run async query - aquery now returns (answer, thinking_steps)
            answer, thinking_steps = loop.run_until_complete(
                self.aquery(question, user_id=user_id, session_id=session_id, conversation_history=conversation_history, rag_documents=rag_documents)
            )
            
            # Return only the answer for sync compatibility
            return answer
            
        except Exception as e:
            logger.error(f"Error in sync query: {e}")
            return f"Xin lỗi, đã có lỗi xảy ra: {str(e)}"

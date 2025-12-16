"""
Financial Agent - Agent chính sử dụng LangGraph và ReAct pattern
"""

import logging
import os
from typing import Literal
from pathlib import Path

from langchain_core.messages import HumanMessage, AIMessage
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
    
    async def aquery(self, question: str, user_id: str = None, session_id: str = None, conversation_history: list = None, rag_documents: list = None) -> str:
        """
        Async query - Xử lý câu hỏi của người dùng
        
        Args:
            question: Câu hỏi tiếng Việt
            user_id: User ID (optional, for context)
            session_id: Session ID (optional, for context)
            conversation_history: List of previous messages (optional)
                Each message: {"role": "user"|"assistant", "content": str}
            rag_documents: List of RAG document chunks to include in context (optional)
                Each doc: {"text": str, "title": str, "source": str, "similarity": float}
            
        Returns:
            Câu trả lời từ agent
        """
        try:
            logger.info(f"Processing question for user {user_id} in session {session_id}: {question}")
            
            # Build message list with conversation history
            messages = []
            
            # Add previous messages from conversation history if provided
            if conversation_history:
                for msg in conversation_history:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role == "assistant":
                        messages.append(AIMessage(content=content))
                    else:
                        messages.append(HumanMessage(content=content))
            
            # Add current question (with RAG context if available)
            if rag_documents and len(rag_documents) > 0:
                # Format RAG documents as context
                rag_context = self._format_rag_context(rag_documents)
                enhanced_question = f"{question}\n\n📚 Related Documents:\n{rag_context}"
                messages.append(HumanMessage(content=enhanced_question))
                logger.info(f"Enhanced question with {len(rag_documents)} RAG documents")
            else:
                messages.append(HumanMessage(content=question))
            
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
            
            logger.info(f"Answer generated for user {user_id}: {answer[:100]}...")
            return answer
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"Xin lỗi, đã có lỗi xảy ra: {str(e)}"
    
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
        
        answer_lower = answer.lower()
        is_json_explanation = any(keyword in answer_lower for keyword in json_explanation_keywords)
        
        if is_json_explanation:
            logger.warning(f"JSON explanation detected, attempting to extract embedded JSON: {answer[:100]}")
            
            # Try to find JSON embedded in the explanation text
            json_pattern = r'\{[^{}]*\}|\[[^\[\]]*\]'
            json_matches = re.findall(json_pattern, answer)
            
            if json_matches:
                for json_str in json_matches:
                    try:
                        # Try to parse each match
                        data = json.loads(json_str)
                        logger.info(f"Extracted JSON from explanation: {str(data)[:100]}")
                        
                        # If we found data, still return the original answer
                        # The system prompt should have taught it better
                        # But if this keeps happening, we could try conversion here
                        break
                    except json.JSONDecodeError:
                        continue
            
            # For now, just log and return original answer
            # The system prompt is supposed to prevent this
            logger.warning(f"Returning original answer despite JSON explanation pattern")
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
            Câu trả lời từ agent
        """
        import asyncio
        
        try:
            # Get or create event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run async query
            return loop.run_until_complete(
                self.aquery(question, user_id=user_id, session_id=session_id, conversation_history=conversation_history, rag_documents=rag_documents)
            )
            
        except Exception as e:
            logger.error(f"Error in sync query: {e}")
            return f"Xin lỗi, đã có lỗi xảy ra: {str(e)}"

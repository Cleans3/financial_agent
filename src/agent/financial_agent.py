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
    
    async def aquery(self, question: str) -> str:
        """
        Async query - Xử lý câu hỏi của người dùng
        
        Args:
            question: Câu hỏi tiếng Việt
            
        Returns:
            Câu trả lời từ agent
        """
        try:
            logger.info(f"Processing question: {question}")
            
            # Create initial state with user question
            initial_state = {
                "messages": [HumanMessage(content=question)]
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
            
            logger.info(f"Answer generated: {answer[:100]}...")
            return answer
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"Xin lỗi, đã có lỗi xảy ra: {str(e)}"
    
    def query(self, question: str) -> str:
        """
        Sync query - Wrapper for async query
        
        Args:
            question: Câu hỏi tiếng Việt
            
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
            return loop.run_until_complete(self.aquery(question))
            
        except Exception as e:
            logger.error(f"Error in sync query: {e}")
            return f"Xin lỗi, đã có lỗi xảy ra: {str(e)}"

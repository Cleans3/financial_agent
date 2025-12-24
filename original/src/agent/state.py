"""
Agent State - Định nghĩa state cho LangGraph
"""

from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    State của Financial Agent
    
    Attributes:
        messages: Lịch sử conversation (HumanMessage, AIMessage, ToolMessage)
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]

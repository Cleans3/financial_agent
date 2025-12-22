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
        allow_tools: Whether tools are allowed in this query
        has_rag_context: Whether RAG documents are attached
        _rag_documents: Raw RAG documents for merging with tool results
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    allow_tools: bool = True
    has_rag_context: bool = False
    _rag_documents: list = []

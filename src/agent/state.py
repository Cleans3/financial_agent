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
        summarize_results: User preference for tool result summarization (True/False/None=auto)
        _rag_documents: Raw RAG documents for merging with tool results
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    allow_tools: bool = True
    has_rag_context: bool = False
    summarize_results: bool = True  # True=always summarize, False=never, True=default (auto >500 chars)
    _rag_documents: list = []

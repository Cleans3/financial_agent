"""
Backward compatibility module for summarization functions.
All functions have been consolidated into src/core/summarization.py
This module re-exports them for backward compatibility.
"""

from src.core.summarization import (
    summarize_messages,
    summarize_tool_result,
    extract_financial_metrics,
    create_enhanced_tool_result,
    create_rag_summary,
    estimate_message_tokens,
    should_compress_history,
)

__all__ = [
    "summarize_messages",
    "summarize_tool_result",
    "extract_financial_metrics",
    "create_enhanced_tool_result",
    "create_rag_summary",
    "estimate_message_tokens",
    "should_compress_history",
]

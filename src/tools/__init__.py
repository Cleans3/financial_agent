"""
Tools Module - Export query-time tools for Financial Agent.

NOTE: File processing tools (PDF, Excel, Image) are NOT included in registry.
They are handled by FileProcessingPipeline for document ingestion.
"""

from typing import Optional
from .vnstock_tools import (
    get_vnstock_tools,
    get_company_info,
    get_historical_data,
    get_shareholders,
    get_officers,
    get_subsidiaries,
    get_company_events
)
from .technical_tools import get_technical_tools, calculate_sma, calculate_rsi


def get_all_tools(config=None):
    """
    Get query-time tools based on configuration.
    
    File processing tools (PDF, Excel, Image) are NOT included.
    Those are handled by FileProcessingPipeline for ingestion.
    
    Args:
        config: Optional ToolsConfig for filtering enabled tools
        
    Returns:
        List of available tools
    """
    if config is None:
        from src.core.tool_config import DEFAULT_TOOLS_CONFIG
        config = DEFAULT_TOOLS_CONFIG
    
    tools = []
    
    # Only include configured tools
    for tool_category in config.enabled_tools:
        if tool_category == "vnstock_tools":
            tools.extend(get_vnstock_tools())
        elif tool_category == "technical_tools":
            tools.extend(get_technical_tools())
    
    # NOTE: File processing tools NOT included
    # - get_financial_report_tools() → handled by FileProcessingPipeline
    # - get_excel_tools() → handled by FileProcessingPipeline
    
    return tools


__all__ = [
    "get_all_tools",
    "get_vnstock_tools",
    "get_technical_tools",
    "get_company_info",
    "get_historical_data",
    "get_shareholders",
    "get_officers",
    "get_subsidiaries",
    "get_company_events",
    "calculate_sma",
    "calculate_rsi",
]

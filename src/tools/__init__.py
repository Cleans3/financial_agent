"""
Tools Module - Export query-time tools for Financial Agent.

NOTE: File processing tools (PDF, Excel, Image) are NOT included in registry.
They are handled by FileProcessingPipeline for document ingestion.
"""

from typing import Optional
from .vnstock_tools import get_vnstock_tools
from .technical_tools import get_technical_tools


def get_all_tools(config=None):
    """
    Get query-time tools based on configuration.
    
    File processing tools (PDF, Excel, Image) are NOT included.
    Those are handled by FileProcessingPipeline for ingestion.
    
    Args:
        config: Optional ToolsConfig for filtering enabled tools
        
    Returns:
        List of available tools (deduplicated by tool name)
    """
    if config is None:
        from src.core.tool_config import DEFAULT_TOOLS_CONFIG
        config = DEFAULT_TOOLS_CONFIG
    
    tools = []
    seen_tool_names = set()  # Track registered tool names to prevent duplicates
    
    # Only include configured tools
    for tool_category in config.enabled_tools:
        tools_to_add = []
        if tool_category == "vnstock_tools":
            tools_to_add = get_vnstock_tools()
        elif tool_category == "technical_tools":
            tools_to_add = get_technical_tools()
        
        # Add tools, skipping duplicates by name
        for tool in tools_to_add:
            tool_name = getattr(tool, 'name', str(tool))
            if tool_name not in seen_tool_names:
                tools.append(tool)
                seen_tool_names.add(tool_name)
    
    # NOTE: File processing tools NOT included
    # - get_financial_report_tools() → handled by FileProcessingPipeline
    # - get_excel_tools() → handled by FileProcessingPipeline
    
    return tools


__all__ = [
    "get_all_tools",
    "get_vnstock_tools",
    "get_technical_tools",
]

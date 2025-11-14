"""
Tools Module - Export tất cả tools cho Financial Agent
"""

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


def get_all_tools():
    """
    Get all tools for the Financial Agent
    
    Returns:
        List of all available tools
    """
    tools = []
    tools.extend(get_vnstock_tools())
    tools.extend(get_technical_tools())
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

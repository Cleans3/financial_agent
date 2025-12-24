"""
Tool Result Formatting - Convert tool outputs to markdown tables for display
"""

import json
from typing import Dict, List, Any, Union


def format_tool_result_as_table(result_json: str) -> str:
    """
    Convert tool JSON result to markdown table format if it contains tabular data.
    
    Args:
        result_json: JSON string from tool execution
        
    Returns:
        Formatted string with tables if data is tabular, otherwise original JSON
    """
    try:
        data = json.loads(result_json)
    except:
        return result_json
    
    # Handle different tool result formats
    if isinstance(data, dict):
        # Check for detailed_data field (SMA, RSI, etc.)
        if "detailed_data" in data and isinstance(data["detailed_data"], list):
            table_md = _create_markdown_table(data["detailed_data"])
            
            # Build formatted response
            output = ""
            
            # Add analysis/interpretation
            if "analysis" in data:
                output += "## Phân Tích\n\n"
                analysis = data["analysis"]
                if "trend" in analysis:
                    output += f"**Xu hướng**: {analysis['trend']}\n\n"
                if "interpretation" in analysis:
                    output += f"**Diễn giải**: {analysis['interpretation']}\n\n"
            
            # Add current values summary
            if "current_values" in data:
                output += "## Giá Trị Hiện Tại\n\n"
                cv = data["current_values"]
                for key, value in cv.items():
                    if isinstance(value, float):
                        output += f"- **{key}**: {value:.2f}\n"
                    else:
                        output += f"- **{key}**: {value}\n"
                output += "\n"
            
            # Add table
            output += "## Dữ Liệu Chi Tiết\n\n"
            output += table_md
            
            # Add message
            if "message" in data:
                output += f"\n\n*{data['message']}*\n"
            
            return output
        
        # Check for other table-like data
        elif any(isinstance(v, list) for v in data.values()):
            for key, value in data.items():
                if isinstance(value, list) and value and isinstance(value[0], dict):
                    output = f"## {key.replace('_', ' ').title()}\n\n"
                    output += _create_markdown_table(value)
                    return output
    
    # If no table found, return formatted JSON
    return json.dumps(data, ensure_ascii=False, indent=2)


def _create_markdown_table(records: List[Dict[str, Any]]) -> str:
    """
    Create a markdown table from a list of dictionaries.
    
    Args:
        records: List of dictionaries with the same keys
        
    Returns:
        Markdown formatted table string
    """
    if not records:
        return ""
    
    # Get headers from first record
    headers = list(records[0].keys())
    
    # Create header row
    header_line = "| " + " | ".join(str(h).replace("_", " ").title() for h in headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    
    # Create data rows
    data_lines = []
    for record in records:
        row_values = []
        for header in headers:
            value = record.get(header, "")
            # Format numbers
            if isinstance(value, float):
                value = f"{value:.2f}"
            row_values.append(str(value))
        data_lines.append("| " + " | ".join(row_values) + " |")
    
    # Combine all parts
    table = "\n".join([header_line, separator_line] + data_lines)
    return table


def format_tool_call_results(tool_calls_results: List[tuple]) -> str:
    """
    Format multiple tool call results for display.
    
    Args:
        tool_calls_results: List of (tool_name, result_json) tuples
        
    Returns:
        Formatted markdown string with all results
    """
    output = "## 🔧 Kết Quả Công Cụ\n\n"
    
    for tool_name, result in tool_calls_results:
        output += f"### {tool_name}\n\n"
        formatted = format_tool_result_as_table(result)
        output += formatted + "\n\n"
    
    return output

"""
Output Formatter - Formats answers for tables, calculations, and complex data
Implements: FORMAT_OUTPUT node logic
"""

import logging
import re
from typing import Dict, Any, List, Optional
from .workflow_state import DataType

logger = logging.getLogger(__name__)


class OutputFormatter:
    """
    Formats LLM-generated answers with structured data formatting.
    
    Handles:
    1. Table formatting (markdown tables)
    2. Calculation formatting (with units and precision)
    3. Text formatting (emphasis, links)
    4. Combined data + narrative formats
    """
    
    def __init__(self, use_markdown: bool = True):
        """
        Initialize output formatter.
        
        Args:
            use_markdown: Whether to use markdown formatting (default True)
        """
        self.use_markdown = use_markdown
        self.logger = logging.getLogger(__name__)
    
    async def format_answer(
        self,
        generated_answer: str,
        search_results: List[Dict[str, Any]],
        tool_results: Optional[Dict[str, Any]] = None,
        detected_data_types: Optional[List[DataType]] = None
    ) -> str:
        """
        Format final answer with structured data where appropriate.
        
        Args:
            generated_answer: LLM-generated text answer
            search_results: Search results with metadata
            tool_results: Results from executed tools
            detected_data_types: Data types found in results
            
        Returns:
            Formatted answer string (markdown or plain text)
        """
        if not generated_answer:
            return "No answer generated."
        
        formatted = generated_answer
        
        # Step 1: Extract and format tables from search results
        if detected_data_types and DataType.TABLE in detected_data_types:
            table_section = await self._format_tables_from_results(search_results)
            if table_section:
                formatted = f"{formatted}\n\n{table_section}"
        
        # Step 2: Format calculation results
        if tool_results and "calculator" in tool_results:
            calc_section = await self._format_calculations(tool_results["calculator"])
            if calc_section:
                formatted = f"{formatted}\n\n{calc_section}"
        
        # Step 3: Add source citations
        citation_section = self._format_sources(search_results)
        if citation_section:
            formatted = f"{formatted}\n\n{citation_section}"
        
        # Step 4: Clean up formatting
        formatted = self._cleanup_formatting(formatted)
        
        self.logger.info(f"Formatted answer ({len(formatted)} chars)")
        return formatted
    
    async def _format_tables_from_results(
        self,
        search_results: List[Dict[str, Any]]
    ) -> Optional[str]:
        """
        Extract and format tables from search results.
        
        Args:
            search_results: List of search result objects
            
        Returns:
            Formatted table section or None
        """
        tables = []
        
        for result in search_results:
            content = result.get("content", "")
            source = result.get("source", "Unknown")
            
            # Check if content looks like a table
            if self._is_table_content(content):
                table = await self._parse_and_format_table(content)
                if table:
                    tables.append({
                        "table": table,
                        "source": source,
                        "title": self._extract_table_title(content)
                    })
        
        if not tables:
            return None
        
        # Build formatted section
        section = "### Data Tables\n\n"
        for i, table_info in enumerate(tables, 1):
            if table_info["title"]:
                section += f"**Table {i}: {table_info['title']}**\n\n"
            
            section += table_info["table"]
            section += f"\n*Source: {table_info['source']}*\n\n"
        
        return section
    
    async def _parse_and_format_table(self, content: str) -> Optional[str]:
        """
        Parse table content and format as markdown table.
        
        Args:
            content: Raw table content
            
        Returns:
            Formatted markdown table or None
        """
        # Check for markdown table
        if "|" in content:
            lines = content.split("\n")
            table_lines = [line for line in lines if "|" in line]
            
            if len(table_lines) >= 2:
                # Format as markdown table
                formatted = "\n".join(table_lines)
                
                # Ensure proper markdown table format
                if "|" in formatted:
                    rows = formatted.split("\n")
                    if len(rows) >= 2:
                        # Add separator if missing
                        if "---" not in rows[1] and "-" not in rows[1]:
                            cols = len(rows[0].split("|"))
                            separator = "|" + "|".join(["---"] * (cols - 2)) + "|"
                            rows.insert(1, separator)
                        
                        return "\n".join(rows)
        
        # Check for other table patterns (borders, etc.)
        if "─" in content or "═" in content or "━" in content:
            # ASCII table detected, try to convert
            return self._convert_ascii_table(content)
        
        return None
    
    def _convert_ascii_table(self, content: str) -> Optional[str]:
        """
        Convert ASCII art table to markdown.
        
        Args:
            content: ASCII table content
            
        Returns:
            Markdown table or None
        """
        lines = content.split("\n")
        
        # Extract data rows (skip borders)
        data_rows = []
        for line in lines:
            # Skip lines that are all borders
            if re.match(r'^[\s\-═─┌┐└┘├┤┬┴┼┋│┃]*$', line):
                continue
            
            # Extract columns (separated by | or spaces)
            if "|" in line:
                cells = [cell.strip() for cell in line.split("|")]
                cells = [c for c in cells if c]  # Remove empty cells
                if cells:
                    data_rows.append(cells)
        
        if len(data_rows) < 2:
            return None
        
        # Build markdown table
        md_table = "| " + " | ".join(data_rows[0]) + " |\n"
        md_table += "|" + "|".join(["---"] * len(data_rows[0])) + "|\n"
        
        for row in data_rows[1:]:
            if len(row) == len(data_rows[0]):
                md_table += "| " + " | ".join(row) + " |\n"
        
        return md_table
    
    def _extract_table_title(self, content: str) -> Optional[str]:
        """
        Extract table title from content.
        
        Args:
            content: Content potentially containing title
            
        Returns:
            Title string or None
        """
        # Look for "Table:" or "Summary:" patterns
        patterns = [
            r'(?:Table|Summary|Data|Report):\s*([^\n]+)',
            r'\*\*([^*]+)\*\*\s*\n\s*\|',  # Bold text before table
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    async def _format_calculations(
        self,
        calc_result: Any
    ) -> Optional[str]:
        """
        Format calculation results with units and precision.
        
        Args:
            calc_result: Calculation result (dict or value)
            
        Returns:
            Formatted calculation section or None
        """
        if not calc_result:
            return None
        
        section = "### Calculations\n\n"
        
        if isinstance(calc_result, dict):
            for key, value in calc_result.items():
                formatted_value = await self._format_value(key, value)
                section += f"**{key}**: {formatted_value}\n\n"
        else:
            formatted_value = await self._format_value("Result", calc_result)
            section += f"**Result**: {formatted_value}\n"
        
        return section
    
    async def _format_value(self, key: str, value: Any) -> str:
        """
        Format a single value with appropriate units.
        
        Args:
            key: Field name for context
            value: Value to format
            
        Returns:
            Formatted value string
        """
        if isinstance(value, (int, float)):
            # Detect type from key name
            if "percent" in key.lower() or "%" in str(value):
                return f"{value:.2f}%"
            elif "currency" in key.lower() or "$" in str(value) or "revenue" in key.lower():
                return f"${value:,.2f}"
            elif "growth" in key.lower() or "change" in key.lower():
                sign = "+" if value > 0 else ""
                return f"{sign}{value:.2f}%"
            elif isinstance(value, float):
                return f"{value:.2f}"
            else:
                return f"{value:,}"
        
        return str(value)
    
    def _format_sources(self, search_results: List[Dict[str, Any]]) -> Optional[str]:
        """
        Format source citations.
        
        Args:
            search_results: Search results with source info
            
        Returns:
            Sources section or None
        """
        if not search_results:
            return None
        
        section = "### Sources\n\n"
        sources = set()
        
        for i, result in enumerate(search_results[:5], 1):  # Top 5 sources
            source = result.get("source", "Unknown")
            doc_id = result.get("doc_id", "")
            
            if source:
                source_str = f"- {source}"
                if doc_id:
                    source_str += f" (ID: {doc_id})"
                sources.add(source_str)
        
        if not sources:
            return None
        
        section += "\n".join(sources)
        return section
    
    def _is_table_content(self, content: str) -> bool:
        """
        Detect if content contains table data.
        
        Args:
            content: Content to check
            
        Returns:
            True if table detected
        """
        # Check for markdown table
        if content.count("|") >= 4:  # At least 2 rows with separators
            return True
        
        # Check for ASCII table borders
        if re.search(r'[─═┌┐└┘├┤┬┴┼]', content):
            return True
        
        # Check for header + data pattern
        lines = content.split("\n")
        if len(lines) >= 2:
            # Lines with consistent column-like structure
            for i in range(min(2, len(lines))):
                if re.match(r'^[\w\s]+\s{2,}[\w\s]+', lines[i]):
                    return True
        
        return False
    
    def _cleanup_formatting(self, text: str) -> str:
        """
        Clean up formatting inconsistencies.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        # Remove multiple blank lines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Trim trailing whitespace
        text = '\n'.join(line.rstrip() for line in text.split('\n'))
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def get_formatter_summary(self) -> Dict[str, Any]:
        """
        Get summary of formatter capabilities.
        
        Returns:
            Dict with formatter info
        """
        return {
            "description": "Output Formatter for structured data",
            "capabilities": [
                "Table formatting (markdown)",
                "Calculation formatting (with units)",
                "Source citation",
                "Combined narrative + data",
                "ASCII table conversion"
            ],
            "supported_data_types": ["TABLE", "NUMERIC", "TEXT", "MIXED"]
        }

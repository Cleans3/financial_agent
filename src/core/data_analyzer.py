"""
Data Analyzer - Detects data types in search results
Implements: ANALYZE_RETRIEVED_RESULTS node logic
"""

import logging
import re
from typing import List, Dict, Any
from .workflow_state import DataType

logger = logging.getLogger(__name__)


class DataAnalyzer:
    """
    Analyzes retrieved search results to detect data types.
    
    Detects:
    - TABLE: Structured table data with columns/rows
    - NUMERIC: Numbers, percentages, currency, calculations
    - TEXT: Prose, explanations, narratives
    - MIXED: Combination of above
    
    Enables smart tool selection (e.g., select Calculator if numeric data found).
    """
    
    def __init__(self, llm=None):
        """
        Initialize analyzer.
        
        Args:
            llm: Optional LLM instance for complex data type detection
        """
        self.llm = llm
        self.logger = logging.getLogger(__name__)
    
    async def analyze_results(self, results: List[Dict]) -> Dict[str, Any]:
        """
        Analyze retrieved results to detect data types.
        
        Args:
            results: List of search results with 'content' field
            
        Returns:
            Dict with:
            {
                'has_table_data': bool,
                'has_numeric_data': bool,
                'text_only': bool,
                'detected_types': List[DataType],
                'details': {
                    'table_count': int,
                    'numeric_patterns_found': List[str],
                    'confidence_scores': Dict[str, float]
                }
            }
        """
        if not results:
            self.logger.info("No results to analyze")
            return self._empty_analysis()
        
        # Aggregate analysis from all results
        has_table = False
        has_numeric = False
        table_count = 0
        numeric_patterns = []
        
        for result in results:
            content = result.get('content', '')
            
            # Check for table patterns
            if self._detect_table(content):
                has_table = True
                table_count += 1
            
            # Check for numeric data
            numeric_found = self._detect_numeric(content)
            if numeric_found:
                has_numeric = True
                numeric_patterns.extend(numeric_found)
        
        # Determine final categorization
        detected_types = self._categorize_types(has_table, has_numeric)
        text_only = not (has_table or has_numeric)
        
        analysis = {
            'has_table_data': has_table,
            'has_numeric_data': has_numeric,
            'text_only': text_only,
            'detected_types': detected_types,
            'details': {
                'table_count': table_count,
                'numeric_patterns_found': list(set(numeric_patterns)),  # Unique patterns
                'confidence_scores': {
                    'table': 0.9 if has_table else 0.0,
                    'numeric': 0.9 if has_numeric else 0.0,
                    'text': 0.9 if text_only else 0.5
                }
            }
        }
        
        self.logger.info(
            f"Data analysis: tables={has_table} ({table_count}), "
            f"numeric={has_numeric} ({len(numeric_patterns)} patterns), "
            f"types={[t.value for t in detected_types]}"
        )
        
        return analysis
    
    def _detect_table(self, content: str) -> bool:
        """
        Detect if content contains table data.
        
        Looks for:
        - Markdown table syntax (| ... |)
        - Table borders (─, ═, etc.)
        - Multiple columns with aligned data
        - HTML table tags
        
        Args:
            content: Text content to analyze
            
        Returns:
            True if table detected
        """
        # Markdown table
        if re.search(r'\|.*\|.*\|', content):
            self.logger.debug("Detected markdown table")
            return True
        
        # Table borders
        if re.search(r'[\-─═]+\s+[\-─═]+', content):
            self.logger.debug("Detected table borders")
            return True
        
        # Grid pattern (multiple rows with consistent columns)
        lines = content.split('\n')
        if len(lines) > 2:
            pipe_lines = sum(1 for line in lines if '|' in line)
            if pipe_lines > 2:
                self.logger.debug("Detected grid pattern with pipes")
                return True
        
        # HTML table
        if '<table>' in content or '<tr>' in content:
            self.logger.debug("Detected HTML table")
            return True
        
        # CSV-like pattern (multiple lines with consistent delimiters)
        if re.search(r'\d+\s*,\s*\d+\s*,\s*\d+', content):
            self.logger.debug("Detected CSV-like pattern")
            return True
        
        return False
    
    def _detect_numeric(self, content: str) -> List[str]:
        """
        Detect numeric data in content.
        
        Looks for:
        - Percentages (50%)
        - Currency ($1,234.56)
        - Large numbers (1,000,000)
        - Decimals (3.14)
        - Fractions (1/2)
        
        Args:
            content: Text content to analyze
            
        Returns:
            List of numeric patterns found
        """
        patterns_found = []
        
        # Percentages
        if re.search(r'\b(\d+\.?\d*%)\b', content):
            patterns_found.append('percentages')
            self.logger.debug("Detected percentages")
        
        # Currency
        if re.search(r'[\$€£¥₹]\s*(\d+[,\d.]*)', content):
            patterns_found.append('currency')
            self.logger.debug("Detected currency")
        
        # Large numbers (1,000 format)
        if re.search(r'\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b', content):
            patterns_found.append('large_numbers')
            self.logger.debug("Detected large numbers")
        
        # Decimals
        if re.search(r'\b\d+\.\d+\b', content):
            patterns_found.append('decimals')
            self.logger.debug("Detected decimals")
        
        # Fractions
        if re.search(r'\d+/\d+', content):
            patterns_found.append('fractions')
            self.logger.debug("Detected fractions")
        
        # Year-over-year growth
        if re.search(r'[Gg]rowth|[Yy]oy|[Qq]oq', content) and re.search(r'\d+%', content):
            patterns_found.append('growth_metrics')
            self.logger.debug("Detected growth metrics")
        
        # Financial keywords with numbers
        financial_keywords = ['revenue', 'profit', 'loss', 'earnings', 'margin', 'ratio']
        for keyword in financial_keywords:
            if re.search(rf'{keyword}.*\d+', content, re.IGNORECASE):
                patterns_found.append(f'{keyword}_data')
                break
        
        return patterns_found
    
    def _categorize_types(self, has_table: bool, has_numeric: bool) -> List[DataType]:
        """
        Determine DataType categories based on detected features.
        
        Args:
            has_table: Whether table data found
            has_numeric: Whether numeric data found
            
        Returns:
            List of DataType enums
        """
        types = []
        
        if has_table:
            types.append(DataType.TABLE)
        
        if has_numeric:
            types.append(DataType.NUMERIC)
        
        if not has_table and not has_numeric:
            types.append(DataType.TEXT)
        elif has_table and has_numeric:
            types.append(DataType.MIXED)
        
        return types
    
    def _empty_analysis(self) -> Dict[str, Any]:
        """Return analysis for empty result set."""
        return {
            'has_table_data': False,
            'has_numeric_data': False,
            'text_only': True,
            'detected_types': [DataType.TEXT],
            'details': {
                'table_count': 0,
                'numeric_patterns_found': [],
                'confidence_scores': {
                    'table': 0.0,
                    'numeric': 0.0,
                    'text': 0.0
                }
            }
        }
    
    async def detect_calculation_needs(self, content: str) -> bool:
        """
        Detect if content requires calculations.
        
        Indicators:
        - "calculate", "compute", "determine"
        - Comparison words: "vs", "compare", "growth from X to Y"
        - Multiple numeric values
        
        Args:
            content: Content to analyze
            
        Returns:
            True if calculation needed
        """
        calculation_keywords = [
            'calculate', 'compute', 'determine', 'growth', 'change',
            'increase', 'decrease', 'compare', 'ratio', 'percent',
            'average', 'total', 'sum', 'difference'
        ]
        
        content_lower = content.lower()
        for keyword in calculation_keywords:
            if keyword in content_lower:
                # Verify there are numbers nearby
                if re.search(rf'{keyword}.*\d', content_lower):
                    self.logger.debug(f"Detected calculation need: {keyword}")
                    return True
        
        return False

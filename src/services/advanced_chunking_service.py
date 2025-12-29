"""
Advanced Chunking Service - Implements two-level chunking strategy for financial documents

Two-Level Chunking Strategy:
1. STRUCTURAL CHUNKS: Preserve document structure (500 tokens, 50 overlap)
2. METRIC-CENTRIC CHUNKS: Aggregate all information about specific metrics across document

This service provides:
- Metric extraction and identification
- Cross-document aggregation of metric information
- Chunk validation and reconciliation
- Bidirectional linking between structural and metric chunks
- Comprehensive logging using universal project standards
"""

import logging
import re
import uuid
import json
from typing import List, Dict, Optional, Tuple, Set
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Financial metrics categories for classification"""
    REVENUE = "revenue"
    PROFITABILITY = "profitability"  # Margin, profit, net income
    CASH_FLOW = "cash_flow"
    LIQUIDITY = "liquidity"  # Current ratio, quick ratio
    LEVERAGE = "leverage"  # Debt ratio, equity ratio
    EFFICIENCY = "efficiency"  # ROA, ROE, turnover
    VALUATION = "valuation"  # P/E, P/B, PEG
    GROWTH = "growth"
    DIVIDEND = "dividend"
    SEGMENT = "segment"
    GUIDANCE = "guidance"
    OTHER = "other"


class ChunkType(str, Enum):
    """Types of chunks in the system"""
    STRUCTURAL = "structural"  # Original document structure
    METRIC_CENTRIC = "metric_centric"  # Aggregated by metric
    SUMMARY = "summary"  # High-level summary chunks


@dataclass
class MetricOccurrence:
    """Reference to a metric mention in a structural chunk"""
    metric_name: str
    metric_type: MetricType
    value: Optional[str] = None  # e.g., "12.1%", "$500M"
    period: Optional[str] = None  # e.g., "Q3 2024", "FY2023"
    structural_chunk_id: Optional[int] = None  # Point ID of structural chunk
    page_ref: Optional[int] = None
    confidence: float = 0.9  # Confidence score for metric extraction
    relevance: float = 1.0  # Relevance score (0.0-1.0) from LLM assessment
    

@dataclass
class MetricChunk:
    """Metric-centric chunk combining information from multiple structural chunks"""
    metric_name: str
    metric_type: MetricType
    chunk_id: str  # UUID for this metric chunk
    text: str  # Synthesized comprehensive text about this metric
    occurrences: List[MetricOccurrence]  # All mentions of this metric
    source_chunk_ids: List[int]  # Point IDs of structural chunks it references
    is_from_table: bool = False  # True if metric extracted from table data
    period: Optional[str] = None
    confidence: float = 0.8
    validation_notes: Optional[str] = None
    file_id: Optional[str] = None  # Reference to source file
    chunk_type: ChunkType = ChunkType.METRIC_CENTRIC
    timestamp: Optional[str] = None
    sources: List[int] = None  # Source chunk IDs (references to structural chunks)
    

@dataclass
class StructuralChunk:
    """Structural chunk preserving document structure"""
    point_id: Optional[int]  # Qdrant point ID (assigned later)
    text: str
    chunk_index: int
    file_id: str
    metrics_found: List[MetricOccurrence]  # Metrics mentioned in this chunk
    page_ref: Optional[int] = None
    chunk_type: ChunkType = ChunkType.STRUCTURAL
    timestamp: Optional[str] = None


# Comprehensive metric keywords for financial documents
METRIC_KEYWORDS = {
    MetricType.REVENUE: {
        "keywords": ["revenue", "sales", "net sales", "total revenue", "turnover", "gross revenue"],
        "patterns": [r"revenue[:\s]+\$?[\d,.]+", r"sales[:\s]+\$?[\d,.]+"]
    },
    MetricType.PROFITABILITY: {
        "keywords": ["margin", "profit", "earnings", "net income", "ebit", "ebitda", "gross profit",
                    "operating profit", "pretax income", "net profit", "bottom line"],
        "patterns": [r"margin[:\s]+[\d.]+%", r"net income[:\s]+\$?[\d,.]+"]
    },
    MetricType.CASH_FLOW: {
        "keywords": ["cash flow", "fcf", "operating cash flow", "free cash flow", "cash from operations",
                    "cffo", "capex", "capital expenditure"],
        "patterns": [r"cash flow[:\s]+\$?[\d,.]+", r"fcf[:\s]+\$?[\d,.]+"]
    },
    MetricType.LIQUIDITY: {
        "keywords": ["liquidity", "current ratio", "quick ratio", "working capital", "dso", "inventory turnover"],
        "patterns": [r"current ratio[:\s]+[\d.]+", r"working capital[:\s]+\$?[\d,.]+"]
    },
    MetricType.LEVERAGE: {
        "keywords": ["debt", "leverage", "debt-to-equity", "debt ratio", "equity ratio", "interest coverage"],
        "patterns": [r"debt[:\s]+\$?[\d,.]+", r"leverage ratio[:\s]+[\d.]+"]
    },
    MetricType.EFFICIENCY: {
        "keywords": ["roa", "roe", "return on assets", "return on equity", "turnover", "productivity"],
        "patterns": [r"roe[:\s]+[\d.]+%", r"roa[:\s]+[\d.]+%"]
    },
    MetricType.VALUATION: {
        "keywords": ["pe ratio", "price-to-earnings", "pb ratio", "price-to-book", "peg", "valuation"],
        "patterns": [r"p/e[:\s]+[\d.]+", r"pe ratio[:\s]+[\d.]+"]
    },
    MetricType.GROWTH: {
        "keywords": ["growth", "increase", "expand", "acceleration", "decline", "decrease"],
        "patterns": [r"growth[:\s]+[\d.]+%", r"increased[:\s]+[\d.]+%"]
    },
    MetricType.DIVIDEND: {
        "keywords": ["dividend", "payout", "distribution", "dividend yield", "dividend per share"],
        "patterns": [r"dividend[:\s]+\$?[\d.]+", r"dividend yield[:\s]+[\d.]+%"]
    },
    MetricType.SEGMENT: {
        "keywords": ["segment", "business unit", "geographic", "product line", "division"],
        "patterns": [r"segment[:\s]+", r"geographic[:\s]+"]
    },
    MetricType.GUIDANCE: {
        "keywords": ["guidance", "outlook", "forecast", "expect", "anticipate", "project"],
        "patterns": [r"guidance[:\s]+", r"outlook[:\s]+"]
    }
}


# HARDCODED PREDEFINED METRICS LIST
# This list is used by the LLM to evaluate and score metrics
# The LLM will decide which metrics are important (score >= 6) based on the document content
PREDEFINED_FINANCIAL_METRICS = {
    "Revenue Metrics": [
        ("Total Revenue", MetricType.REVENUE),
        ("Revenue Growth Rate (YoY %)", MetricType.GROWTH),
        ("Revenue by Segment/Type", MetricType.SEGMENT),
        ("Recurring Revenue", MetricType.REVENUE),
        ("Revenue Concentration Index", MetricType.GROWTH),
    ],
    "Profitability Metrics": [
        ("Gross Margin %", MetricType.PROFITABILITY),
        ("Operating Margin %", MetricType.PROFITABILITY),
        ("Net Profit Margin %", MetricType.PROFITABILITY),
        ("EBIT", MetricType.PROFITABILITY),
        ("EBITDA", MetricType.PROFITABILITY),
        ("Net Income", MetricType.PROFITABILITY),
    ],
    "Cash Flow Metrics": [
        ("Operating Cash Flow", MetricType.CASH_FLOW),
        ("Free Cash Flow", MetricType.CASH_FLOW),
        ("Cash from Operations", MetricType.CASH_FLOW),
        ("Capital Expenditure", MetricType.CASH_FLOW),
        ("Cash Position / Cash Balance", MetricType.LIQUIDITY),
    ],
    "Balance Sheet Metrics": [
        ("Total Assets", MetricType.EFFICIENCY),
        ("Total Liabilities", MetricType.LEVERAGE),
        ("Total Equity", MetricType.LEVERAGE),
        ("Current Assets", MetricType.LIQUIDITY),
        ("Current Liabilities", MetricType.LIQUIDITY),
        ("Accounts Receivable", MetricType.EFFICIENCY),
        ("Inventory", MetricType.EFFICIENCY),
    ],
    "Efficiency & Effectiveness": [
        ("Asset Turnover Ratio", MetricType.EFFICIENCY),
        ("Return on Assets (ROA)", MetricType.EFFICIENCY),
        ("Return on Equity (ROE)", MetricType.EFFICIENCY),
        ("Days Sales Outstanding (DSO)", MetricType.EFFICIENCY),
        ("Inventory Turnover", MetricType.EFFICIENCY),
    ],
    "Liquidity & Solvency": [
        ("Current Ratio", MetricType.LIQUIDITY),
        ("Quick Ratio", MetricType.LIQUIDITY),
        ("Working Capital", MetricType.LIQUIDITY),
        ("Debt-to-Equity Ratio", MetricType.LEVERAGE),
        ("Interest Coverage Ratio", MetricType.LEVERAGE),
        ("Debt Ratio", MetricType.LEVERAGE),
    ],
    "Valuation Metrics": [
        ("Price-to-Earnings (P/E)", MetricType.VALUATION),
        ("Price-to-Book (P/B)", MetricType.VALUATION),
        ("Enterprise Value / EBITDA", MetricType.VALUATION),
        ("PEG Ratio", MetricType.VALUATION),
    ],
    "Dividend & Shareholder Metrics": [
        ("Dividend Per Share", MetricType.DIVIDEND),
        ("Dividend Payout Ratio", MetricType.DIVIDEND),
        ("Dividend Yield", MetricType.DIVIDEND),
        ("Earnings Per Share (EPS)", MetricType.REVENUE),
    ],
    "Guidance & Forward Indicators": [
        ("Revenue Guidance / Forecast", MetricType.GUIDANCE),
        ("Earnings Guidance", MetricType.GUIDANCE),
        ("Capital Expenditure Guidance", MetricType.GUIDANCE),
        ("Market Outlook / Expectations", MetricType.GUIDANCE),
    ],
}


class AdvancedChunkingService:
    """
    Service for two-level document chunking and metric extraction
    
    Workflow:
    1. Create STRUCTURAL chunks from original document (preserve structure)
    2. Extract metrics from all structural chunks
    3. Create METRIC-CENTRIC chunks by aggregating all info about each metric
    4. Validate and reconcile metric chunks
    5. Store with bidirectional links
    
    LLM Integration:
    - Extraction: Rule-based (pattern matching)
    - Synthesis: LLM-based (if available) or rule-based fallback
    - Validation: Rule-based (numeric verification)
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50, llm=None):
        """
        Initialize advanced chunking service
        
        Args:
            chunk_size: Token size for structural chunks
            chunk_overlap: Overlap tokens between structural chunks
            llm: Optional LLM instance for metric synthesis
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.llm = llm  # LLM for metric synthesis
        logger.info(f"[CHUNKING] Advanced chunking service initialized: "
                   f"chunk_size={chunk_size}, overlap={chunk_overlap}, "
                   f"llm_synthesis={'enabled' if llm else 'disabled'}")
    
    def create_structural_chunks(self, 
                                 text: str, 
                                 file_id: str) -> List[StructuralChunk]:
        """
        Step 1: Create structural chunks preserving document flow
        Uses sentence-based chunking (500 tokens, 50 overlap)
        
        Args:
            text: Full document text
            file_id: Source file identifier
            
        Returns:
            List of structural chunks
        """
        logger.info(f"[CHUNKING:STRUCTURAL] Starting structural chunking for file {file_id}")
        
        # Split by sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            words = sentence.split()
            sentence_length = len(words)
            
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                
                # Create overlap
                overlap_sentences = []
                overlap_length = 0
                for prev_sentence in reversed(current_chunk):
                    prev_length = len(prev_sentence.split())
                    if overlap_length + prev_length <= self.chunk_overlap:
                        overlap_sentences.insert(0, prev_sentence)
                        overlap_length += prev_length
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_length = overlap_length
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)
        
        # Create StructuralChunk objects
        structural_chunks = []
        chunk_index = 0
        
        for chunk_text in chunks:
            if chunk_text.strip():  # Skip empty chunks
                metrics_found = self._extract_metrics_from_text(chunk_text)
                structural_chunk = StructuralChunk(
                    point_id=None,
                    text=chunk_text,
                    chunk_index=chunk_index,
                    file_id=file_id,
                    metrics_found=metrics_found,
                    page_ref=None,
                    timestamp=datetime.now().isoformat()
                )
                structural_chunks.append(structural_chunk)
                logger.debug(f"[CHUNKING:STRUCTURAL] Chunk {chunk_index}: "
                            f"{len(chunk_text.split())} words, "
                            f"{len(metrics_found)} metrics found")
                chunk_index += 1
        
        logger.info(f"[CHUNKING:STRUCTURAL] ✓ Created {len(structural_chunks)} structural chunks "
                   f"from file {file_id}")
        return structural_chunks
    
    def _extract_metrics_from_text(self, text: str) -> List[MetricOccurrence]:
        """
        Extract metric mentions from text
        Handles both prose text and markdown tables (for Excel files)
        
        Args:
            text: Text to extract metrics from
            
        Returns:
            List of MetricOccurrence objects
        """
        occurrences = []
        text_lower = text.lower()
        
        # Try LLM-based extraction first if available
        if self.llm:
            llm_occurrences = self._extract_metrics_with_llm(text)
            if llm_occurrences:
                return llm_occurrences
        
        # Fallback to rule-based extraction
        return self._extract_metrics_rule_based(text)
    
    def _extract_metrics_with_llm(self, text: str) -> List[MetricOccurrence]:
        """
        Use LLM to intelligently identify and extract relevant metrics from document.
        
        Args:
            text: Document text
            
        Returns:
            List of MetricOccurrence objects identified by LLM
        """
        try:
            logger.debug(f"[CHUNKING:METRIC] Using LLM-based metric extraction")
            
            # Prepare extraction prompt
            extraction_prompt = f"""You are a financial analysis expert. Extract all relevant financial metrics from the following document excerpt.

For each metric found, provide:
1. Metric name (e.g., "revenue", "net income", "cash flow")
2. Metric type (e.g., "revenue", "profitability", "liquidity", "valuation", "growth")
3. Value (if available, with unit like $M, %, etc.)
4. Period (if available, like "2023", "Q3 2024")
5. Confidence (0.6-1.0 based on clarity and relevance)

Document excerpt:
{text[:2000]}

Return metrics in this format:
metric_name|metric_type|value|period|confidence

Example:
revenue|revenue|932M|2023|0.95
net income|profitability|150M|2023|0.90

Only include metrics that are clearly present in the document. Stop when done."""
            
            response = self.llm.invoke(extraction_prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse LLM response
            occurrences = []
            for line in response_text.strip().split('\n'):
                if '|' in line and not line.startswith('metric'):
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) >= 5:
                        try:
                            metric_name = parts[0]
                            metric_type_str = parts[1].lower()
                            value = parts[2] if parts[2] and parts[2] != 'N/A' else None
                            period = parts[3] if parts[3] and parts[3] != 'N/A' else None
                            confidence = float(parts[4])
                            
                            # Map type string to MetricType
                            metric_type = self._get_metric_type(metric_type_str)
                            
                            occurrence = MetricOccurrence(
                                metric_name=metric_name,
                                metric_type=metric_type,
                                value=value,
                                period=period,
                                confidence=confidence
                            )
                            occurrences.append(occurrence)
                        except (ValueError, IndexError) as e:
                            logger.debug(f"[CHUNKING:METRIC] Skipping malformed metric line: {line}")
                            continue
            
            if occurrences:
                logger.debug(f"[CHUNKING:METRIC] LLM identified {len(occurrences)} metrics")
                return occurrences
            else:
                logger.debug(f"[CHUNKING:METRIC] LLM extraction found no metrics, falling back to rule-based")
                return []
                
        except Exception as e:
            logger.warning(f"[CHUNKING:METRIC] LLM-based extraction failed: {e}, using rule-based fallback")
            return []
    
    def _get_metric_type(self, type_str: str) -> MetricType:
        """Map type string to MetricType enum"""
        type_str = type_str.lower().strip()
        
        mapping = {
            'revenue': MetricType.REVENUE,
            'sales': MetricType.REVENUE,
            'profitability': MetricType.PROFITABILITY,
            'profit': MetricType.PROFITABILITY,
            'earnings': MetricType.PROFITABILITY,
            'cash flow': MetricType.CASH_FLOW,
            'cash_flow': MetricType.CASH_FLOW,
            'liquidity': MetricType.LIQUIDITY,
            'leverage': MetricType.LEVERAGE,
            'debt': MetricType.LEVERAGE,
            'efficiency': MetricType.EFFICIENCY,
            'valuation': MetricType.VALUATION,
            'growth': MetricType.GROWTH,
            'dividend': MetricType.DIVIDEND,
            'segment': MetricType.SEGMENT,
            'guidance': MetricType.GUIDANCE,
        }
        
        for key, metric_type in mapping.items():
            if key in type_str:
                return metric_type
        
        return MetricType.OTHER
    
    def _extract_metrics_rule_based(self, text: str) -> List[MetricOccurrence]:
        """
        Rule-based metric extraction (fallback when LLM not available)
        
        Args:
            text: Text to extract metrics from
            
        Returns:
            List of MetricOccurrence objects
        """
        occurrences = []
        text_lower = text.lower()
        
        # Check if this looks like a table (markdown table or structured data)
        is_table_content = "|" in text and ("bảng" in text_lower or "sheet" in text_lower or "table" in text_lower)
        
        # For table content, extract metrics directly from table structure
        if is_table_content or "|" in text:
            table_metrics = self._extract_metrics_from_table(text)
            occurrences.extend(table_metrics)
        
        # Also extract metrics from keywords (for prose text)
        for metric_type, metric_info in METRIC_KEYWORDS.items():
            keywords = metric_info["keywords"]
            
            # Check for keyword presence
            for keyword in keywords:
                if keyword in text_lower:
                    # Try to extract value and period
                    value = self._extract_value_after_keyword(text, keyword)
                    period = self._extract_period_from_text(text)
                    
                    occurrence = MetricOccurrence(
                        metric_name=keyword,
                        metric_type=metric_type,
                        value=value,
                        period=period,
                        confidence=0.9 if value else 0.6
                    )
                    # Only add if not already from table
                    if not any(o.metric_name == keyword and o.value for o in occurrences):
                        occurrences.append(occurrence)
        
        # Remove duplicates by metric type (keep highest confidence)
        unique_occurrences = {}
        for occ in occurrences:
            key = (occ.metric_name, occ.metric_type)
            if key not in unique_occurrences or occ.confidence > unique_occurrences[key].confidence:
                unique_occurrences[key] = occ
        
        return list(unique_occurrences.values())
    
    def _extract_metrics_from_table(self, text: str) -> List[MetricOccurrence]:
        """
        Extract metrics from markdown table headers and content.
        Utility for Excel/CSV files converted to markdown. Also extracts actual values from table rows.
        
        Args:
            text: Markdown table text
            
        Returns:
            List of MetricOccurrence objects from table with extracted values
        """
        occurrences = []
        text_lower = text.lower()
        
        # Parse table structure to extract values
        lines = text.split('\n')
        table_rows = []
        for line in lines:
            if '|' in line and '-' not in line:
                cells = [cell.strip() for cell in line.split('|') if cell.strip()]
                table_rows.append(cells)
        
        # Common column names in financial tables (Vietnamese + English)
        table_metric_map = {
            # Revenue variations
            ("revenue", "doanh thu"): (MetricType.REVENUE, "revenue"),
            ("sales", "bán hàng", "bán"): (MetricType.REVENUE, "sales"),
            
            # Profit variations
            ("profit", "lợi nhuận", "lợi nhuậnn"): (MetricType.PROFITABILITY, "profit"),
            ("margin", "biên lợi"): (MetricType.PROFITABILITY, "margin"),
            ("earnings", "thu nhập"): (MetricType.PROFITABILITY, "earnings"),
            ("ebit", "ebitda"): (MetricType.PROFITABILITY, "ebit"),
            
            # Cash flow
            ("cash flow", "dòng tiền", "lưu chuyển tiền"): (MetricType.CASH_FLOW, "cash flow"),
            ("fcf", "cffo"): (MetricType.CASH_FLOW, "fcf"),
            
            # Growth
            ("growth", "tăng trưởng", "tăng"): (MetricType.GROWTH, "growth"),
            ("increase", "tăng", "tăng lên"): (MetricType.GROWTH, "increase"),
            
            # Returns
            ("roe", "roa"): (MetricType.EFFICIENCY, "roa/roe"),
            ("return", "lợi suất"): (MetricType.EFFICIENCY, "return"),
            
            # Debt/Leverage
            ("debt", "nợ", "tín dụng"): (MetricType.LEVERAGE, "debt"),
            ("leverage", "đòn bẩy"): (MetricType.LEVERAGE, "leverage"),
            
            # Ratios
            ("ratio", "tỷ suất", "tỷ lệ"): (MetricType.VALUATION, "ratio"),
            ("pe", "p/e", "price-to-earnings"): (MetricType.VALUATION, "pe"),
        }
        
        # Look for metrics in table headers and content
        for keywords_tuple, (metric_type, metric_name) in table_metric_map.items():
            for keyword in keywords_tuple:
                if keyword in text_lower:
                    # Found a metric in table
                    # Try to extract numeric values from nearby content
                    values_and_periods = self._extract_values_from_table_rows(text, keyword)
                    
                    if values_and_periods:
                        # Create occurrence for each value found
                        for value, period in values_and_periods:
                            occurrence = MetricOccurrence(
                                metric_name=metric_name,
                                metric_type=metric_type,
                                value=value,
                                period=period,
                                confidence=0.95  # High confidence for table data
                            )
                            occurrences.append(occurrence)
                    else:
                        # No values found, still create occurrence
                        period = self._extract_period_from_text(text)
                        occurrence = MetricOccurrence(
                            metric_name=metric_name,
                            metric_type=metric_type,
                            value=None,
                            period=period,
                            confidence=0.7
                        )
                        occurrences.append(occurrence)
                    break  # Only count once per metric type
        
        return occurrences
    
    def _extract_value_from_table_context(self, text: str, keyword: str) -> Optional[str]:
        """Extract numeric value from table context around keyword"""
        # Look for table rows containing the keyword
        lines = text.split('\n')
        for line in lines:
            if keyword.lower() in line.lower() and '|' in line:
                # Extract numeric values from the line
                pattern = r'\$?[\d,.]+(M|B|K|%)?'
                matches = re.findall(pattern, line)
                
                if matches:
                    # Return the first numeric value in that row (after the label)
                    return matches[0] if matches else None
        
        # Fallback: search in surrounding context
        lower_text = text.lower()
        pos = lower_text.find(keyword)
        
        if pos >= 0:
            context_start = max(0, pos - 50)
            context_end = min(len(text), pos + 100)
            context = text[context_start:context_end]
            
            pattern = r'\$?[\d,.]+(M|B|K|%)?'
            matches = re.findall(pattern, context)
            
            if matches:
                return matches[-1] if matches else None
        
        return None
    
    def _extract_values_from_table_rows(self, text: str, keyword: str) -> List[Tuple[str, Optional[str]]]:
        """
        Extract all numeric values from table rows containing keyword.
        Returns list of (value, period) tuples.
        
        Args:
            text: Table text containing keyword
            keyword: Metric keyword to search for
            
        Returns:
            List of (value, period) tuples extracted from rows
        """
        results = []
        lines = text.split('\n')
        
        # Find header line to get column structure
        headers = []
        header_idx = -1
        for idx, line in enumerate(lines):
            if '|' in line and '--' not in line:
                cells = [c.strip() for c in line.split('|') if c.strip()]
                # Check if this looks like headers (short text, likely years)
                if all(len(c) < 30 for c in cells):
                    headers = cells
                    header_idx = idx
                    break
        
        # Find data rows containing the keyword
        for line in lines[header_idx + 1:] if header_idx >= 0 else lines:
            if keyword.lower() in line.lower() and '|' in line:
                # Extract values from cells
                cells = [c.strip() for c in line.split('|') if c.strip()]
                
                if len(cells) > 1:
                    # First cell is usually the row label, remaining cells are values
                    values = cells[1:]
                    
                    # Try to pair with column headers (years/periods)
                    for i, value in enumerate(values):
                        # Extract numeric part using search instead of findall
                        pattern = r'\$?[\d,]+(?:\.[\d]+)?(?:M|B|K|%)?'
                        match = re.search(pattern, value)
                        
                        if match:
                            extracted_value = match.group(0)
                            # Try to get period from headers
                            period = None
                            if header_idx >= 0 and i < len(headers) - 1:
                                header_cell = headers[i + 1] if i + 1 < len(headers) else headers[-1]
                                # Check if it looks like a year
                                if re.search(r'\d{4}|\d{2}', header_cell):
                                    period = header_cell
                            
                            results.append((extracted_value, period))
        
        return results
    
    
    def _extract_value_after_keyword(self, text: str, keyword: str) -> Optional[str]:
        """Extract numeric value following a keyword"""
        # First try pattern matching for explicit values
        escaped_keyword = re.escape(keyword)
        pattern = f"{escaped_keyword}[:\\s]+([\\$]?[\\d,.]+%?)"
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[0]
        
        # Then try table extraction for structured data
        return self._extract_value_from_table_context(text, keyword)
    
    def _extract_period_from_text(self, text: str) -> Optional[str]:
        """Extract time period (Q1 2024, FY2023, etc) from text"""
        patterns = [
            r"[QqFf][1-4Y]?\s*(?:20\d{2}|FY\d{2,4})",
            r"(?:20\d{2}|FY\d{2,4})",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        return None
    
    def _select_metric_values(self,
                             metric_name: str,
                             occurrences: List[MetricOccurrence],
                             source_texts: List[str]) -> Dict[str, str]:
        """
        Select the best/most relevant values for a metric.
        
        Returns:
            Dict with keys:
            - latest_value: Most recent or primary value
            - period: Period for latest value
            - all_values: List of all unique values found
        
        Args:
            metric_name: Name of metric
            occurrences: All metric occurrences
            source_texts: Source texts for additional extraction
            
        Returns:
            Dict with selected metric values
        """
        result = {}
        
        # Collect all unique values and periods
        all_values = []
        periods = []
        values_with_periods = {}
        
        for occ in occurrences:
            if occ.value:
                all_values.append(occ.value)
                if occ.period:
                    periods.append(occ.period)
                    values_with_periods[occ.period] = occ.value
        
        # Also try to extract values directly from source texts
        for text in source_texts:
            # Look for metric name followed by value
            escaped_metric = re.escape(metric_name)
            pattern = f"{escaped_metric}[:\\s]+([\\$]?[\\d,.]+(?:M|B|K|%)?)"
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if match not in all_values:
                    all_values.append(match)
        
        # Remove duplicates while preserving order
        unique_values = []
        seen = set()
        for val in all_values:
            if val not in seen:
                unique_values.append(val)
                seen.add(val)
        
        # Select latest value (prefer most recent period)
        if values_with_periods:
            # Sort periods and get most recent
            sorted_periods = sorted(values_with_periods.keys(), reverse=True)
            if sorted_periods:
                latest_period = sorted_periods[0]
                result['latest_value'] = values_with_periods[latest_period]
                result['period'] = latest_period
        elif unique_values:
            # If no period available, use first value
            result['latest_value'] = unique_values[0]
        
        # Store all values
        if unique_values:
            result['all_values'] = unique_values
        
        return result
    
    def aggregate_metric_chunks(self,
                               structural_chunks: List[StructuralChunk],
                               file_id: str) -> List[MetricChunk]:
        """
        Step 2: LLM-DRIVEN METRIC SELECTION AND CHUNK CREATION
        
        Workflow (from financial_agent_prompt_ref.md):
        1. Prepare structural chunks for LLM evaluation
        2. Send hardcoded metric list + structural chunks to LLM
        3. LLM scores each metric by materiality/volatility/anomaly
        4. LLM returns selected metrics (score >= 6)
        5. Create metric-centric chunks ONLY for selected metrics
        
        Args:
            structural_chunks: List of structural chunks
            file_id: Source file ID
            
        Returns:
            List of metric-centric chunks (only for LLM-selected metrics with score >= 6)
        """
        logger.info(f"[CHUNKING:METRIC] Starting LLM-driven metric selection for file {file_id}")
        logger.info(f"[CHUNKING:METRIC] Step 1: Prepare structural chunks for LLM evaluation")
        
        # Prepare document context from structural chunks
        document_excerpts = []
        for chunk in structural_chunks[:10]:  # Use first 10 chunks for context
            document_excerpts.append(f"[Chunk {chunk.chunk_index}]\n{chunk.text[:300]}")
        document_context = "\n\n".join(document_excerpts)
        
        # Step 2: Use LLM to evaluate and select metrics from hardcoded list
        logger.info(f"[CHUNKING:METRIC] Step 2: Evaluating hardcoded metrics with LLM")
        selected_metrics = self._llm_evaluate_and_select_metrics(
            document_context,
            structural_chunks,
            file_id
        )
        
        if not selected_metrics:
            logger.warning(f"[CHUNKING:METRIC] No metrics selected by LLM, returning empty metric chunks")
            return []
        
        logger.info(f"[CHUNKING:METRIC] LLM selected {len(selected_metrics)} metrics (score >= 6.0)")
        for metric_name, score in selected_metrics.items():
            logger.info(f"[CHUNKING:METRIC]   ✓ {metric_name}: score={score:.1f}/10.0")
        
        # Step 3: Extract and aggregate data for SELECTED metrics only
        logger.info(f"[CHUNKING:METRIC] Step 3: Extract data for selected metrics from chunks")
        
        # Create a map of chunks indexed by metric name
        metric_chunk_data: Dict[str, Dict] = {}
        
        for struct_chunk in structural_chunks:
            # Extract ALL metrics from this chunk
            chunk_text = struct_chunk.text
            all_metrics = self._extract_all_metric_mentions(chunk_text, struct_chunk)
            
            # Keep only metrics that were selected by LLM
            for metric in all_metrics:
                metric_key = metric.metric_name.lower()
                
                # Check if this metric was selected
                selected_metric_name = None
                for sel_name in selected_metrics.keys():
                    if sel_name.lower() == metric_key or self._metrics_match(sel_name, metric.metric_name):
                        selected_metric_name = sel_name
                        break
                
                if selected_metric_name:
                    if selected_metric_name not in metric_chunk_data:
                        metric_chunk_data[selected_metric_name] = {
                            "metric_type": metric.metric_type,
                            "occurrences": [],
                            "source_chunks": [],
                            "texts": [],
                            "score": selected_metrics[selected_metric_name],
                            "is_from_table": self._is_table_chunk(chunk_text)
                        }
                    
                    metric.structural_chunk_id = struct_chunk.point_id
                    metric_chunk_data[selected_metric_name]["occurrences"].append(metric)
                    metric_chunk_data[selected_metric_name]["source_chunks"].append(struct_chunk.point_id)
                    metric_chunk_data[selected_metric_name]["texts"].append(chunk_text)
        
        # Step 4: Create metric-centric chunks for selected metrics
        logger.info(f"[CHUNKING:METRIC] Step 4: Creating metric-centric chunks for selected metrics")
        metric_chunks = []
        
        for metric_name, metric_data in metric_chunk_data.items():
            if not metric_data["occurrences"]:
                logger.debug(f"[CHUNKING:METRIC] Metric '{metric_name}' selected but no data found in chunks, skipping")
                continue
            
            # Extract best values
            best_values = self._select_metric_values(
                metric_name,
                metric_data["occurrences"],
                metric_data["texts"]
            )
            
            # Synthesize comprehensive text
            if metric_data["is_from_table"]:
                aggregated_text = self._synthesize_table_metric_text(
                    metric_name,
                    metric_data["texts"][0],
                    metric_data["occurrences"],
                    best_values
                )
            else:
                aggregated_text = self._synthesize_metric_text(
                    metric_name,
                    metric_data["texts"],
                    metric_data["occurrences"],
                    best_values
                )
            
            source_chunks = [cid for cid in metric_data["source_chunks"] if cid is not None]
            metric_chunk = MetricChunk(
                metric_name=metric_name,
                metric_type=metric_data["metric_type"],
                chunk_id=str(uuid.uuid4()),
                text=aggregated_text,
                occurrences=metric_data["occurrences"],
                source_chunk_ids=source_chunks,
                is_from_table=metric_data["is_from_table"],
                file_id=file_id,
                timestamp=datetime.now().isoformat(),
                sources=source_chunks,
                confidence=metric_data["score"] / 10.0  # Normalize score to 0-1
            )
            
            metric_chunks.append(metric_chunk)
            logger.info(f"[CHUNKING:METRIC]   ✓ Created chunk for '{metric_name}' (score={metric_data['score']:.1f}/10.0)")
        
        logger.info(f"[CHUNKING:METRIC] ✓ Created {len(metric_chunks)} metric-centric chunks "
                   f"from {len(selected_metrics)} LLM-selected metrics")
        return metric_chunks
    
    def _llm_evaluate_and_select_metrics(self,
                                        document_context: str,
                                        structural_chunks: List[StructuralChunk],
                                        file_id: str) -> Dict[str, float]:
        """
        Use LLM to evaluate hardcoded metrics against document and select important ones.
        
        This is the core LLM-driven selection mechanism:
        - LLM reads document context
        - LLM evaluates each predefined metric for materiality/volatility/anomaly
        - LLM returns scores for metrics (1-10 scale)
        - Service filters to keep only metrics with score >= 6
        
        Returns:
            Dict mapping metric_name -> score (1-10 scale) for selected metrics only
        """
        if not self.llm:
            logger.warning(f"[CHUNKING:METRIC] No LLM available, cannot perform metric selection")
            return {}
        
        try:
            logger.info(f"[CHUNKING:METRIC] LLM evaluating hardcoded metrics (threshold=6.0)")
            
            # Flatten predefined metrics into simple list
            all_metrics_list = []
            for category, metrics in PREDEFINED_FINANCIAL_METRICS.items():
                for metric_name, metric_type in metrics:
                    all_metrics_list.append(f"{metric_name} ({metric_type.value})")
            
            metrics_text = "\n".join([f"{i+1}. {m}" for i, m in enumerate(all_metrics_list)])
            
            # Create LLM prompt for metric evaluation
            evaluation_prompt = f"""You are a financial analyst evaluating which metrics are most important in this financial document.

DOCUMENT CONTENT (excerpts):
{document_context[:2000]}

HARDCODED METRICS TO EVALUATE:
{metrics_text}

For EACH metric, determine if it appears or is relevant in the document, and score it 1-10 based on:

MATERIALITY (size relative to category total):
- 9-10: Metric >25% of category total
- 7-8: Metric 15-25% of total
- 5-6: Metric 5-15% of total
- 1-4: Metric <5% or not clearly present

VOLATILITY (change across periods):
- +5 if YoY change >15%
- +3 if YoY change 5-15%
- +1 if YoY change <5%

STRUCTURAL IMPORTANCE:
- +3 for foundational metrics (Total Revenue, Total Assets)
- +2 for category drivers (Revenue by Type, margin %)
- +1 for supporting metrics

ANOMALY/SIGNAL:
- +4 if unexpected behavior or trend reversal
- +2 if contradicts larger trend

RESPOND WITH ONLY:
1. Metric Name: X
2. Metric Name: X
...etc

Where X is 1-10 score. NO explanations."""
            
            logger.debug(f"[CHUNKING:METRIC] Sending {len(all_metrics_list)} metrics to LLM for evaluation")
            response = self.llm.invoke(evaluation_prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse response: "N. Metric Name: X" format
            scores = {}
            lines = response_text.split('\n')
            
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                
                # Try to parse "N. Metric Name: X" or similar formats
                # Look for pattern: number period optional space then metric name then colon then number
                match = re.match(r'^[\d]+\.\s*(.+?):\s*(\d+(?:\.\d+)?)', line)
                if match:
                    metric_name_from_response = match.group(1).strip()
                    score_str = match.group(2)
                    
                    try:
                        score = float(score_str)
                        score = max(1.0, min(10.0, score))  # Clamp to 1-10
                        
                        # Try to match with predefined metrics
                        matched_metric = None
                        for metric in all_metrics_list:
                            # Extract just the metric name without type
                            metric_name_only = metric.split(' (')[0] if ' (' in metric else metric
                            if metric_name_only.lower() in metric_name_from_response.lower() or \
                               metric_name_from_response.lower() in metric_name_only.lower():
                                matched_metric = metric_name_only
                                break
                        
                        if matched_metric:
                            scores[matched_metric] = score
                            logger.debug(f"[CHUNKING:METRIC] Parsed: {matched_metric} -> {score:.1f}")
                    except ValueError:
                        logger.debug(f"[CHUNKING:METRIC] Could not parse score from: {line}")
            
            # Filter to keep only metrics with score >= 6.0
            SELECTION_THRESHOLD = 6.0
            selected = {name: score for name, score in scores.items() if score >= SELECTION_THRESHOLD}
            
            logger.info(f"[CHUNKING:METRIC] LLM evaluation complete: {len(selected)}/{len(scores)} metrics selected (score >= {SELECTION_THRESHOLD})")
            return selected
            
        except Exception as e:
            logger.error(f"[CHUNKING:METRIC] LLM metric selection failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return {}
    
    def _extract_all_metric_mentions(self, text: str, struct_chunk: StructuralChunk) -> List[MetricOccurrence]:
        """
        Extract all metric mentions from text using both rule-based and LLM methods.
        Used by metric selection process to find all potential metrics.
        """
        # Use existing extraction logic
        metrics = self._extract_metrics_from_text(text)
        return metrics
    
    def _metrics_match(self, metric1: str, metric2: str) -> bool:
        """Check if two metric names refer to the same metric"""
        # Simple matching - can be improved
        m1_lower = metric1.lower()
        m2_lower = metric2.lower()
        
        # Exact match after normalization
        if m1_lower == m2_lower:
            return True
        
        # Check if one is substring of other
        if m1_lower in m2_lower or m2_lower in m1_lower:
            return True
        
        return False
    
    
    def _is_table_chunk(self, text: str) -> bool:
        """Check if text appears to be a markdown table"""
        return "|" in text and text.strip().startswith("|")
    
    def _extract_table_column_metrics(self, table_text: str) -> List[MetricOccurrence]:
        """
        Extract metrics from table column headers and data.
        Looks for financial keywords in markdown table headers.
        
        Args:
            table_text: Markdown table text
            
        Returns:
            List of MetricOccurrence objects from table
        """
        occurrences = []
        table_lower = table_text.lower()
        
        # Extract column headers from markdown table (first | delimited line)
        lines = table_text.split('\n')
        headers = []
        for line in lines:
            if '|' in line and '-' not in line:  # Skip separator lines
                # Extract column names
                cols = line.split('|')
                headers.extend([col.strip() for col in cols if col.strip()])
                break
        
        # Match headers against metric keywords
        for metric_type, metric_info in METRIC_KEYWORDS.items():
            keywords = metric_info["keywords"]
            
            for keyword in keywords:
                # Check headers for keyword match
                for header in headers:
                    if keyword.lower() in header.lower():
                        occurrence = MetricOccurrence(
                            metric_name=keyword,
                            metric_type=metric_type,
                            value=None,  # Values in table rows, not headers
                            period=None,
                            confidence=0.95  # High confidence for explicit column headers
                        )
                        occurrences.append(occurrence)
                        break
        
        # Remove duplicates
        seen = set()
        unique = []
        for occ in occurrences:
            key = (occ.metric_name.lower(), occ.metric_type)
            if key not in seen:
                seen.add(key)
                unique.append(occ)
        
        return unique
    
    def _synthesize_table_metric_text(self,
                                     metric_name: str,
                                     table_text: str,
                                     occurrences: List[MetricOccurrence],
                                     best_values: Dict[str, str] = None) -> str:
        """
        Synthesize metric-centric text from table data with context and summary.
        
        Creates a comprehensive summary around the specific metric including:
        - Metric values and periods
        - Relevant table rows
        - Historical trends when available
        - Summary statistics
        
        Args:
            metric_name: Name of metric
            table_text: Markdown table text
            occurrences: Metric occurrences in table
            best_values: Selected metric values dict (optional)
            
        Returns:
            Metric-centric text with context and summary
        """
        synthesis_parts = []
        
        # Header with metric name
        synthesis_parts.append(f"## {metric_name.upper()}")
        synthesis_parts.append("")
        
        # Add best values and metrics summary
        if best_values:
            synthesis_parts.append("### Values")
            if best_values.get('latest_value'):
                latest_line = f"**Latest:** {best_values['latest_value']}"
                if best_values.get('period'):
                    latest_line += f" ({best_values['period']})"
                synthesis_parts.append(latest_line)
            
            if best_values.get('all_values') and len(best_values['all_values']) > 1:
                synthesis_parts.append(f"**Historical:** {', '.join(best_values['all_values'][:5])}")
            
            synthesis_parts.append("")
        
        # Extract relevant rows with context
        lines = table_text.split('\n')
        relevant_rows = []
        
        for i, line in enumerate(lines):
            if metric_name.lower() in line.lower():
                # Include surrounding context (header rows)
                if i > 0 and '|' in lines[i-1]:
                    if lines[i-1] not in relevant_rows:
                        relevant_rows.append(lines[i-1])
                relevant_rows.append(line)
        
        if relevant_rows:
            synthesis_parts.append("### Table Data")
            synthesis_parts.append("")
            for row in relevant_rows[:15]:
                if row.strip():
                    synthesis_parts.append(row)
            synthesis_parts.append("")
        
        # Add occurrence summary
        synthesis_parts.append("### Summary")
        synthesis_parts.append(f"- **Type:** {occurrences[0].metric_type if occurrences else 'Unknown'}")
        synthesis_parts.append(f"- **Occurrences:** {len(occurrences)}")
        
        if occurrences:
            avg_confidence = sum(o.confidence for o in occurrences) / len(occurrences)
            synthesis_parts.append(f"- **Confidence:** {avg_confidence:.0%}")
        
        synthesis_parts.append(f"- **Source:** Table Data")
        
        return "\n".join(synthesis_parts)
    
    def _synthesize_metric_text(self,
                               metric_name: str,
                               source_texts: List[str],
                               occurrences: List[MetricOccurrence],
                               best_values: Dict[str, str] = None) -> str:
        """
        Synthesize metric-centric comprehensive text from narrative sources.
        
        Creates a summary paragraph that:
        - Defines the metric in context
        - States key values and periods
        - Explains drivers and context
        - Highlights trends
        - Includes scoring rationale (materiality, volatility, importance)
        
        Args:
            metric_name: Name of metric
            source_texts: Texts mentioning this metric
            occurrences: Metric occurrences with relevance scores
            best_values: Selected metric values dict (optional)
            
        Returns:
            Metric-centric synthesis text with importance reasoning
        """
        # Extract relevant sentences containing the metric
        relevant_sentences = []
        for text in source_texts:
            sentences = re.split(r'(?<=[.!?])\s+', text)
            for sentence in sentences:
                if metric_name.lower() in sentence.lower():
                    relevant_sentences.append(sentence.strip())
        
        # Remove duplicates while preserving order
        seen = set()
        unique_sentences = []
        for sentence in relevant_sentences:
            if sentence not in seen:
                unique_sentences.append(sentence)
                seen.add(sentence)
        
        # Try LLM synthesis if available
        if self.llm and unique_sentences:
            try:
                logger.debug(f"[CHUNKING:SYNTHESIS] Using LLM to synthesize metric: {metric_name}")
                
                # Build value summary if available
                value_context = ""
                if best_values:
                    if best_values.get('latest_value'):
                        value_context = f"\nKEY VALUE: {best_values['latest_value']}"
                        if best_values.get('period'):
                            value_context += f" ({best_values['period']})"
                    if best_values.get('all_values'):
                        value_context += f"\nVALUES ACROSS PERIODS: {', '.join(best_values['all_values'])}"
                
                # Get average relevance score to include in synthesis
                avg_relevance = sum(o.relevance for o in occurrences) / len(occurrences) if occurrences else 0.0
                importance_1_to_10 = (avg_relevance * 9.0) + 1.0  # Convert from 0-1 back to 1-10
                
                # Prepare context for LLM
                context = "\n".join(unique_sentences)
                
                # Create synthesis prompt with comprehensive requirements and scoring rationale
                synthesis_prompt = f"""You are a financial analysis expert. Create a comprehensive metric-centric summary for "{metric_name}".

{value_context}

DOCUMENT MENTIONS:
{context}

This metric has been scored as IMPORTANT (importance rating: {importance_1_to_10:.1f}/10.0) due to:
- Materiality to overall financial position
- Volatility and year-over-year changes
- Structural importance to business operations
- Anomalies or significant trends

Synthesize this into a cohesive metric summary that:
1. Clearly explains what {metric_name} represents
2. Provides the actual values: {best_values.get('latest_value', 'N/A')} in {best_values.get('period', 'N/A')}
3. Shows historical context: {', '.join(best_values.get('all_values', [])) if best_values and best_values.get('all_values') else 'N/A'}
4. Explains business drivers and impacts
5. Highlights why this metric is important (materiality, volatility, or unusual trends)
6. Connects to other business factors mentioned

Format as a single comprehensive paragraph (150-200 words). Be precise with numbers and dates."""
                
                # Call LLM for synthesis
                response = self.llm.invoke(synthesis_prompt)
                synthesized = response.content if hasattr(response, 'content') else str(response)
                
                logger.debug(f"[CHUNKING:SYNTHESIS] ✓ LLM synthesis successful for {metric_name}")
                return synthesized
            except Exception as e:
                logger.warning(f"[CHUNKING:SYNTHESIS] LLM synthesis failed for {metric_name}: {e}, using fallback")
                # Fall through to rule-based synthesis
        
        # Rule-based synthesis (fallback or when no LLM)
        logger.debug(f"[CHUNKING:SYNTHESIS] Using rule-based synthesis for {metric_name}")
        
        synthesis_parts = []
        
        # Header with metric
        synthesis_parts.append(f"### {metric_name.upper()}")
        synthesis_parts.append("")
        
        # Add best values if available
        if best_values:
            synthesis_parts.append("**Metric Values:**")
            if best_values.get('latest_value'):
                value_line = f"- Latest: {best_values['latest_value']}"
                if best_values.get('period'):
                    value_line += f" ({best_values['period']})"
                synthesis_parts.append(value_line)
            if best_values.get('all_values') and len(best_values['all_values']) > 1:
                synthesis_parts.append(f"- Historical: {', '.join(best_values['all_values'])}")
            synthesis_parts.append("")
        
        # Add relevant context
        if unique_sentences:
            synthesis_parts.append("**Context from Document:**")
            synthesis_parts.append("")
            for sentence in unique_sentences[:5]:
                synthesis_parts.append(f"- {sentence}")
            synthesis_parts.append("")
        
        # Add metric metadata with importance
        synthesis_parts.append("**Metric Metadata:**")
        if occurrences:
            synthesis_parts.append(f"- Type: {occurrences[0].metric_type}")
            synthesis_parts.append(f"- Found: {len(occurrences)} time(s)")
            avg_confidence = sum(o.confidence for o in occurrences) / len(occurrences)
            synthesis_parts.append(f"- Confidence: {avg_confidence:.0%}")
            avg_relevance = sum(o.relevance for o in occurrences) / len(occurrences)
            importance_1_to_10 = (avg_relevance * 9.0) + 1.0
            synthesis_parts.append(f"- Importance Rating: {importance_1_to_10:.1f}/10.0")
        
        return "\n".join(synthesis_parts)
        for occ in occurrences:
            if occ.period:
                periods.add(occ.period)
            if occ.value:
                values.append(occ.value)
        
        if periods:
            synthesis += f"\nPERIODS: {', '.join(sorted(periods))}\n"
        
        synthesis += f"\nCONTEXT:\n" + "\n".join(unique_sentences)
        
        return synthesis
    
    def score_all_metrics_relevance(self,
                                   metric_chunks: List[MetricChunk],
                                   document_context: str) -> Dict[str, float]:
        """
        Score ALL metrics using LLM-driven materiality/volatility/anomaly framework.
        
        Based on financial_agent_prompt_ref.md, this implements:
        - Materiality (size relative to category)
        - Volatility (YoY changes)
        - Structural importance (foundational vs supporting)
        - Anomaly/signal (unexpected behavior)
        
        Args:
            metric_chunks: All metric chunks to score
            document_context: Document text for context
            
        Returns:
            Dict mapping metric_name -> importance_score (0.0-10.0)
        """
        if not self.llm or not metric_chunks:
            logger.debug(f"[CHUNKING:RELEVANCE] No LLM or no metrics, assuming default scores")
            return {mc.metric_name: 6.0 for mc in metric_chunks}
        
        try:
            logger.info(f"[CHUNKING:RELEVANCE] Starting LLM-driven importance scoring for {len(metric_chunks)} metrics")
            
            # Prepare metric list for LLM evaluation
            metric_list = "\n".join([
                f"{i+1}. {mc.metric_name} ({mc.metric_type.value})"
                for i, mc in enumerate(metric_chunks)
            ])
            
            # Create LLM prompt with financial framework
            scoring_prompt = f"""You are a financial analyst evaluating the importance and materiality of financial metrics.

DOCUMENT CONTEXT:
{document_context[:1500]}

METRICS TO EVALUATE:
{metric_list}

For each metric, score on a 1-10 scale based on:

MATERIALITY (size relative to category total):
- Score 9-10: Metric is >25% of its category total
- Score 7-8: Metric is 15-25% of total
- Score 5-6: Metric is 5-15% of total
- Score 1-4: Metric is <5% of total

VOLATILITY (change across periods):
- +5 points if YoY change is >15%
- +3 points if YoY change is 5-15%
- +1 point if YoY change is <5%

STRUCTURAL IMPORTANCE:
- +3 points for foundational metrics (Total Revenue, Total Assets)
- +2 points for category drivers (Revenue by Type, Expense breakdown)
- +1 point for supporting metrics (derived ratios)

ANOMALY/SIGNAL:
- +4 points if metric shows unexpected behavior or trend reversal
- +2 points if metric contradicts larger trend

RELEVANCE:
Consider if metric reveals something meaningful about operations.

RESPOND WITH ONLY these lines (one per metric, in order):
Metric 1: 8
Metric 2: 6
Metric 3: 7
...etc

Use format "Metric N: X" where X is 1-10 score. NO explanations."""
            
            # Call LLM for scoring
            response = self.llm.invoke(scoring_prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            logger.debug(f"[CHUNKING:RELEVANCE] LLM response: {response_text[:500]}")
            
            # Parse scores from response - extract "Metric N: X" format
            relevance_scores = {}
            
            # Try primary parsing: "Metric N: X" format
            pattern = r'Metric\s+\d+:\s*(\d+(?:\.\d+)?)'
            matches = re.findall(pattern, response_text)
            
            if matches and len(matches) >= len(metric_chunks) * 0.5:
                # Good match with primary pattern
                logger.debug(f"[CHUNKING:RELEVANCE] Found {len(matches)} scores with 'Metric N: X' pattern")
                for i, metric_chunk in enumerate(metric_chunks):
                    if i < len(matches):
                        try:
                            score = float(matches[i])
                            score = max(1.0, min(10.0, score))  # Clamp to [1, 10]
                            relevance_scores[metric_chunk.metric_name] = score
                        except ValueError:
                            relevance_scores[metric_chunk.metric_name] = 6.0
                    else:
                        relevance_scores[metric_chunk.metric_name] = 6.0
            else:
                # Fallback: extract any numbers from lines
                logger.debug(f"[CHUNKING:RELEVANCE] Primary pattern matched {len(matches)} scores, trying fallback")
                lines = response_text.split('\n')
                score_values = []
                
                for line in lines:
                    line = line.strip()
                    if not line or len(line) > 100:
                        continue
                    
                    # Extract number from line (handle "8", "8.5", ": 8", etc.)
                    numbers = re.findall(r'(\d+(?:\.\d+)?)', line)
                    if numbers:
                        try:
                            score = float(numbers[0])
                            score = max(1.0, min(10.0, score))
                            score_values.append(score)
                        except ValueError:
                            pass
                
                # Assign extracted scores to metrics
                for i, metric_chunk in enumerate(metric_chunks):
                    if i < len(score_values):
                        relevance_scores[metric_chunk.metric_name] = score_values[i]
                    else:
                        # Default to medium importance if score missing
                        relevance_scores[metric_chunk.metric_name] = 6.0
            
            # Log results
            scores_found = len([s for s in relevance_scores.values()])
            logger.info(f"[CHUNKING:RELEVANCE] ✓ Scored {scores_found}/{len(metric_chunks)} metrics with materiality/volatility framework")
            
            for metric_name, score in sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True)[:10]:
                logger.debug(f"[CHUNKING:RELEVANCE] '{metric_name}': {score:.1f}/10.0")
            
            return relevance_scores
        
        except Exception as e:
            logger.warning(f"[CHUNKING:RELEVANCE] LLM scoring failed: {e}, using default scores")
            import traceback
            logger.debug(traceback.format_exc())
            return {mc.metric_name: 6.0 for mc in metric_chunks}
    
    def score_metric_relevance(self,
                              metric_name: str,
                              metric_type: MetricType,
                              context_text: str,
                              document_snippet: str = "") -> float:
        """
        Score metric relevance using LLM (0.0-1.0).
        
        Evaluates if a metric is actually relevant to the document context.
        Uses LLM if available, otherwise returns 1.0 (assume relevant).
        
        NOTE: For efficiency, prefer score_all_metrics_relevance() for batch scoring.
        This method is for individual metric scoring when needed.
        
        Args:
            metric_name: Name of the metric (e.g., "revenue", "dividend")
            metric_type: Type category (e.g., REVENUE, DIVIDEND)
            context_text: Document context/excerpt mentioning the metric
            document_snippet: Additional document context for relevance assessment
            
        Returns:
            Relevance score 0.0-1.0 (1.0 = highly relevant, 0.0 = not relevant)
        """
        if not self.llm:
            logger.debug(f"[CHUNKING:RELEVANCE] No LLM available, assuming metric '{metric_name}' is relevant")
            return 1.0  # Default to relevant if no LLM
        
        try:
            logger.debug(f"[CHUNKING:RELEVANCE] Scoring relevance for metric: {metric_name}")
            
            # Create relevance assessment prompt
            relevance_prompt = f"""You are a financial analysis expert. Assess the relevance of the metric "{metric_name}" to this document context.

METRIC TYPE: {metric_type.value}
METRIC NAME: {metric_name}

DOCUMENT CONTEXT:
{context_text[:500]}

{"ADDITIONAL CONTEXT:\n" + document_snippet[:300] if document_snippet else ""}

Determine if this metric is:
- HIGHLY RELEVANT (1.0): Core to document's main narrative (financials, earnings, growth, etc.)
- RELEVANT (0.7): Related but secondary (supporting metrics, ratios)
- SOMEWHAT RELEVANT (0.4): Mentioned but not focus (background information)
- NOT RELEVANT (0.0): Unrelated or irrelevant to document (e.g., dividend metrics in a pre-IPO startup report)

Respond with ONLY a single decimal number between 0.0 and 1.0 (e.g., 0.85)"""
            
            # Call LLM
            response = self.llm.invoke(relevance_prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse relevance score from response
            try:
                # Extract first decimal number found in response
                import re as regex_module
                match = regex_module.search(r'0\.\d+|1\.0', response_text)
                if match:
                    score = float(match.group(0))
                    score = max(0.0, min(1.0, score))  # Clamp to [0.0, 1.0]
                    logger.debug(f"[CHUNKING:RELEVANCE] Metric '{metric_name}' scored: {score:.2f}")
                    return score
                else:
                    logger.warning(f"[CHUNKING:RELEVANCE] Could not parse score for '{metric_name}', defaulting to 0.5")
                    return 0.5
            except (ValueError, AttributeError) as e:
                logger.warning(f"[CHUNKING:RELEVANCE] Error parsing score: {e}, defaulting to 0.5")
                return 0.5
        
        except Exception as e:
            logger.warning(f"[CHUNKING:RELEVANCE] LLM relevance scoring failed for '{metric_name}': {e}, assuming relevant")
            return 1.0  # Fail open - assume relevant if LLM fails
    
    def filter_metrics_by_relevance(self,
                                   metric_chunks: List[MetricChunk],
                                   relevance_threshold: float = 0.5) -> List[MetricChunk]:
        """
        Filter metric chunks based on relevance scores.
        
        Removes metrics with relevance score below threshold.
        Useful for removing spurious metric extractions.
        
        Args:
            metric_chunks: Metric chunks to filter
            relevance_threshold: Minimum relevance score (0.0-1.0) to keep metric
            
        Returns:
            Filtered list of metric chunks with relevance >= threshold
        """
        logger.info(f"[CHUNKING:FILTER] Filtering {len(metric_chunks)} metrics by relevance "
                   f"(threshold={relevance_threshold:.2f})")
        
        filtered_chunks = []
        removed_count = 0
        
        for metric_chunk in metric_chunks:
            # Calculate average relevance from occurrences
            if metric_chunk.occurrences:
                avg_relevance = sum(occ.relevance for occ in metric_chunk.occurrences) / len(metric_chunk.occurrences)
            else:
                avg_relevance = 1.0
            
            if avg_relevance >= relevance_threshold:
                filtered_chunks.append(metric_chunk)
                logger.debug(f"[CHUNKING:FILTER] ✓ Kept '{metric_chunk.metric_name}': "
                           f"relevance={avg_relevance:.2f}")
            else:
                removed_count += 1
                logger.debug(f"[CHUNKING:FILTER] ✗ Removed '{metric_chunk.metric_name}': "
                           f"relevance={avg_relevance:.2f} (below threshold)")
        
        logger.info(f"[CHUNKING:FILTER] ✓ Filtered from {len(metric_chunks)} to {len(filtered_chunks)} metrics "
                   f"({removed_count} removed as irrelevant)")
        return filtered_chunks
    
    def validate_metric_chunks(self,
                              metric_chunks: List[MetricChunk],
                              llm_validator=None) -> List[MetricChunk]:
        """
        Step 3: Validate metric chunks for quality and consistency
        
        Args:
            metric_chunks: Metric chunks to validate
            llm_validator: Optional LLM function for semantic validation
            
        Returns:
            Validated metric chunks with validation notes
        """
        logger.info(f"[CHUNKING:VALIDATE] Starting validation of {len(metric_chunks)} metric chunks")
        
        for metric_chunk in metric_chunks:
            validation_notes = []
            
            # Check 1: Multiple sources (good signal)
            if len(metric_chunk.source_chunk_ids) >= 2:
                validation_notes.append("Multiple sources consolidation ✓")
            elif len(metric_chunk.source_chunk_ids) == 1:
                validation_notes.append("Single source ⚠")
            else:
                validation_notes.append("No sources found ✗")
            
            # Check 2: Values consistency (if multiple values exist)
            if metric_chunk.occurrences:
                values = [occ.value for occ in metric_chunk.occurrences if occ.value]
                if len(set(values)) > 1:
                    validation_notes.append(f"Multiple values found: {set(values)} ⚠")
                elif values:
                    validation_notes.append(f"Consistent value: {values[0]} ✓")
            
            # Check 3: Confidence score
            avg_confidence = sum(occ.confidence for occ in metric_chunk.occurrences) / len(metric_chunk.occurrences) if metric_chunk.occurrences else 0
            validation_notes.append(f"Confidence: {avg_confidence:.1%}")
            
            # Check 4: Relevance score (from LLM assessment)
            if metric_chunk.occurrences:
                avg_relevance = sum(occ.relevance for occ in metric_chunk.occurrences) / len(metric_chunk.occurrences)
                validation_notes.append(f"Relevance: {avg_relevance:.1%}")
            
            metric_chunk.validation_notes = " | ".join(validation_notes)
            metric_chunk.confidence = avg_confidence
        
        logger.info(f"[CHUNKING:VALIDATE] ✓ Validation complete: "
                   f"{len([m for m in metric_chunks if m.confidence >= 0.8])} "
                   f"high-confidence chunks")
        return metric_chunks
    
    def prepare_for_storage(self,
                           structural_chunks: List[StructuralChunk],
                           metric_chunks: List[MetricChunk],
                           user_id: str,
                           chat_session_id: str) -> Tuple[List[Dict], List[Dict]]:
        """
        Prepare chunks for storage in Qdrant with metadata and sources tracking.
        
        Metadata structure:
        - Structural chunks: sources = [] (no sources), is_table = True/False
        - Narrative metric chunks: sources = [structural_chunk_ids], is_from_table = False
        - Table metric chunks: sources = [structural_chunk_ids], is_from_table = True
        
        Args:
            structural_chunks: Structural chunks with point IDs assigned
            metric_chunks: Validated metric chunks
            user_id: User ID
            chat_session_id: Chat session ID
            
        Returns:
            (structural_payloads, metric_payloads) for Qdrant storage
        """
        logger.info(f"[CHUNKING:STORAGE] Preparing {len(structural_chunks)} structural + "
                   f"{len(metric_chunks)} metric chunks for storage")
        
        structural_payloads = []
        for struct_chunk in structural_chunks:
            payload = {
                'user_id': user_id,
                'chat_session_id': chat_session_id,
                'file_id': struct_chunk.file_id,
                'chunk_index': struct_chunk.chunk_index,
                'text': struct_chunk.text,
                'chunk_type': ChunkType.STRUCTURAL.value,
                'content_type': 'financial',
                'timestamp': struct_chunk.timestamp,
                'metrics_mentioned': [occ.metric_name for occ in struct_chunk.metrics_found]
            }
            structural_payloads.append(payload)
        
        metric_payloads = []
        for metric_chunk in metric_chunks:
            payload = {
                'user_id': user_id,
                'chat_session_id': chat_session_id,
                'file_id': metric_chunk.file_id,
                'metric_name': metric_chunk.metric_name,
                'metric_type': metric_chunk.metric_type.value,
                'chunk_id': metric_chunk.chunk_id,
                'text': metric_chunk.text,
                'chunk_type': ChunkType.METRIC_CENTRIC.value,
                'content_type': 'financial',
                'is_from_table': metric_chunk.is_from_table,  # True for table metrics
                'timestamp': metric_chunk.timestamp,
                'source_chunk_ids': metric_chunk.source_chunk_ids,  # Links to structural chunks
                'sources': metric_chunk.sources if metric_chunk.sources else metric_chunk.source_chunk_ids,  # Sources metadata
                'confidence': metric_chunk.confidence,
                'relevance': sum(occ.relevance for occ in metric_chunk.occurrences) / len(metric_chunk.occurrences) if metric_chunk.occurrences else 1.0,  # Average relevance score
                'validation_notes': metric_chunk.validation_notes,
                'periods': [occ.period for occ in metric_chunk.occurrences if occ.period]
            }
            metric_payloads.append(payload)
        
        logger.info(f"[CHUNKING:STORAGE] ✓ Prepared payloads for storage "
                   f"({len(structural_payloads)} structural, "
                   f"{len([m for m in metric_payloads if not m['is_from_table']])} narrative metric + "
                   f"{len([m for m in metric_payloads if m['is_from_table']])} table metric chunks)")
        return structural_payloads, metric_payloads
    
    def process_document(self,
                        text: str,
                        file_id: str,
                        user_id: str,
                        chat_session_id: str) -> Tuple[List[Dict], List[Dict]]:
        """
        Full pipeline: Structural -> Metric -> Validation -> Storage preparation
        
        Args:
            text: Full document text
            file_id: File identifier
            user_id: User ID
            chat_session_id: Chat session ID
            
        Returns:
            (structural_payloads, metric_payloads) ready for Qdrant
        """
        logger.info(f"[CHUNKING:PROCESS] Starting full pipeline for file {file_id}")
        
        # Step 1: Create structural chunks
        structural_chunks = self.create_structural_chunks(text, file_id)
        
        # Step 2: Aggregate into metric chunks
        metric_chunks = self.aggregate_metric_chunks(structural_chunks, file_id)
        
        # Step 3: Validate
        metric_chunks = self.validate_metric_chunks(metric_chunks)
        
        # Step 4: Prepare for storage
        struct_payloads, metric_payloads = self.prepare_for_storage(
            structural_chunks, metric_chunks, user_id, chat_session_id
        )
        
        logger.info(f"[CHUNKING:PROCESS] ✓ Full pipeline complete: "
                   f"{len(struct_payloads)} structural, "
                   f"{len(metric_payloads)} metric chunks")
        return struct_payloads, metric_payloads

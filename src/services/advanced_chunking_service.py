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
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Financial metrics categories for classification"""
    REVENUE = "revenue"
    PROFITABILITY = "profitability"  # Margin, profit, net income
    NET_INCOME = "net_income"  # Specific profit measure
    OPERATING_INCOME = "operating_income"  # Operating profit
    CASH_FLOW = "cash_flow"
    OPERATING_CASH_FLOW = "operating_cash_flow"  # Operating cash
    FREE_CASH_FLOW = "free_cash_flow"  # Free cash flow
    LIQUIDITY = "liquidity"  # Current ratio, quick ratio
    CURRENT_ASSETS = "current_assets"
    CURRENT_LIABILITIES = "current_liabilities"
    LEVERAGE = "leverage"  # Debt ratio, equity ratio
    TOTAL_ASSETS = "total_assets"
    TOTAL_LIABILITIES = "total_liabilities"
    EQUITY = "equity"
    DEBT = "debt"
    EFFICIENCY = "efficiency"  # ROA, ROE, turnover
    ROE = "roe"  # Return on equity
    ROA = "roa"  # Return on assets
    ASSET_TURNOVER = "asset_turnover"
    VALUATION = "valuation"  # P/E, P/B, PEG
    PE_RATIO = "pe_ratio"
    PB_RATIO = "pb_ratio"
    GROWTH = "growth"
    REVENUE_GROWTH = "revenue_growth"
    EARNINGS_GROWTH = "earnings_growth"
    DIVIDEND = "dividend"
    DIVIDEND_YIELD = "dividend_yield"
    SEGMENT = "segment"
    GUIDANCE = "guidance"
    EXPENSES = "expenses"
    OPERATING_EXPENSES = "operating_expenses"
    COST_OF_GOODS = "cost_of_goods"
    SG_AND_A = "sg_and_a"  # Selling, General & Administrative
    EBITDA = "ebitda"
    MARGINS = "margins"
    GROSS_MARGIN = "gross_margin"
    OPERATING_MARGIN = "operating_margin"
    NET_MARGIN = "net_margin"
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
    is_custom_metric: bool = False  # True if metric was created by LLM (not in predefined list)
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
            
            # Prepare extraction prompt - improved to capture history and allow synthesis
            extraction_prompt = f"""You are a financial analysis expert. Extract ALL relevant financial metrics from the following document excerpt, including:
- Primary metrics explicitly stated
- Historical values (with years/periods) 
- Derived metrics (e.g., calculate margins from gross/net income)
- Additional metrics you synthesize from context (up to 3 new ones)

For each metric, provide:
1. Metric name (clear name, e.g., "revenue", "total assets", "operating expenses")
2. Metric type (specific type: revenue, net_income, operating_income, total_assets, operating_expenses, gross_margin, operating_margin, net_margin, roe, roa, cash_flow, debt, equity, growth, ebitda, etc.)
3. Value (current value with unit like 932M, 15.5%, etc.)
4. Period (exact period: "2024", "Q3 2024", "FY2023", etc.)
5. Confidence (0.7-1.0 for found data, 0.6-0.8 for synthesized/derived)

CRITICAL: For history, include period info. Example if 2024=1107M, 2023=283M: format as "historical|2024:1107M|2023:283M"

Document excerpt:
{text[:5000]}

Return metrics in this format (one per line):
metric_name|metric_type|value|period|confidence

Examples:
Total Assets|total_assets|1107M|2024|0.95
Total Assets|total_assets|historical|2024:1107M|2023:283M|0.85
Revenue|revenue|932M|2023|0.92
Operating Expenses|operating_expenses|120M|2023|0.88
Gross Margin|gross_margin|45.5%|2023|0.87

INSTRUCTIONS:
- Extract ALL metrics you find, both explicit and derived
- Always include period/year with values
- If showing history, include years with each value (e.g., "2024:1107M, 2023:283M")
- You may add up to 3 synthesized metrics based on context
- Never hallucinate numbers - only use what's in the document
- Be specific with metric types (not just "other")
- Stop when done"""
            
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
                            
                            # Skip if value is "historical" - that's handled separately
                            if value and value.lower() == 'historical':
                                # Don't create a metric for historical marker itself
                                continue
                            
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
        
        # Extended mapping with more specific types
        mapping = {
            # Revenue
            'revenue': MetricType.REVENUE,
            'sales': MetricType.REVENUE,
            'turnover': MetricType.REVENUE,
            'net sales': MetricType.REVENUE,
            
            # Profitability & Income
            'net income': MetricType.NET_INCOME,
            'net profit': MetricType.NET_INCOME,
            'operating income': MetricType.OPERATING_INCOME,
            'operating profit': MetricType.OPERATING_INCOME,
            'ebitda': MetricType.EBITDA,
            'earnings': MetricType.PROFITABILITY,
            'profit': MetricType.PROFITABILITY,
            'profitability': MetricType.PROFITABILITY,
            
            # Cash Flow
            'operating cash': MetricType.OPERATING_CASH_FLOW,
            'operating cash flow': MetricType.OPERATING_CASH_FLOW,
            'free cash': MetricType.FREE_CASH_FLOW,
            'free cash flow': MetricType.FREE_CASH_FLOW,
            'fcf': MetricType.FREE_CASH_FLOW,
            'cash flow': MetricType.CASH_FLOW,
            'cash_flow': MetricType.CASH_FLOW,
            
            # Assets & Liabilities
            'total assets': MetricType.TOTAL_ASSETS,
            'assets': MetricType.TOTAL_ASSETS,
            'current assets': MetricType.CURRENT_ASSETS,
            'current liabilities': MetricType.CURRENT_LIABILITIES,
            'total liabilities': MetricType.TOTAL_LIABILITIES,
            'liabilities': MetricType.TOTAL_LIABILITIES,
            
            # Equity & Debt
            'equity': MetricType.EQUITY,
            'shareholders equity': MetricType.EQUITY,
            'debt': MetricType.DEBT,
            'long-term debt': MetricType.DEBT,
            'leverage': MetricType.LEVERAGE,
            
            # Liquidity
            'liquidity': MetricType.LIQUIDITY,
            'current ratio': MetricType.LIQUIDITY,
            'quick ratio': MetricType.LIQUIDITY,
            
            # Efficiency & Returns
            'roe': MetricType.ROE,
            'return on equity': MetricType.ROE,
            'roa': MetricType.ROA,
            'return on assets': MetricType.ROA,
            'asset turnover': MetricType.ASSET_TURNOVER,
            'efficiency': MetricType.EFFICIENCY,
            'turnover': MetricType.ASSET_TURNOVER,
            
            # Margins
            'gross margin': MetricType.GROSS_MARGIN,
            'operating margin': MetricType.OPERATING_MARGIN,
            'net margin': MetricType.NET_MARGIN,
            'margin': MetricType.MARGINS,
            'profitability margin': MetricType.MARGINS,
            
            # Expenses
            'operating expenses': MetricType.OPERATING_EXPENSES,
            'opex': MetricType.OPERATING_EXPENSES,
            'cost of goods': MetricType.COST_OF_GOODS,
            'cogs': MetricType.COST_OF_GOODS,
            'sg&a': MetricType.SG_AND_A,
            'selling general administrative': MetricType.SG_AND_A,
            'expenses': MetricType.EXPENSES,
            
            # Valuation
            'pe ratio': MetricType.PE_RATIO,
            'p/e': MetricType.PE_RATIO,
            'pb ratio': MetricType.PB_RATIO,
            'p/b': MetricType.PB_RATIO,
            'valuation': MetricType.VALUATION,
            
            # Growth
            'revenue growth': MetricType.REVENUE_GROWTH,
            'earnings growth': MetricType.EARNINGS_GROWTH,
            'growth': MetricType.GROWTH,
            
            # Dividend
            'dividend': MetricType.DIVIDEND,
            'dividend yield': MetricType.DIVIDEND_YIELD,
            'yield': MetricType.DIVIDEND_YIELD,
            
            # Other
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
        keyword_lower = keyword.lower()
        lines = text.split('\n')
        
        # FIRST: Look for markdown table rows containing the keyword
        for line in lines:
            if keyword_lower in line.lower() and '|' in line:
                # Extract numeric values from the markdown table line
                pattern = r'[\d,]+(?:\.\d+)?(?:M|B|K)?(?:%)?'
                matches = re.findall(pattern, line)
                
                if matches:
                    # Return the first numeric value (most recent year typically)
                    if '-' not in line or any(char.isdigit() for char in line):
                        # Filter out matches that are just commas
                        for match in matches:
                            if any(c.isdigit() for c in match):
                                return match
        
        # SECOND: Look for plain text table rows (space-separated values)
        # Pattern: keyword on a line, followed by numeric values
        for i, line in enumerate(lines):
            if keyword_lower in line.lower():
                # Extract all numbers from this line
                pattern = r'[\d,]+(?:\.\d+)?(?:M|B|K)?(?:%)?'
                matches = re.findall(pattern, line)
                
                # If we found numbers on the keyword line itself, use them
                if matches:
                    # Return the first valid number (skip comma-only matches)
                    for match in matches:
                        if any(c.isdigit() for c in match):
                            return match
                
                # Otherwise check the next few lines for related data
                # (in case the keyword is a header and values are below)
                for j in range(i+1, min(i+3, len(lines))):
                    next_line = lines[j]
                    next_matches = re.findall(pattern, next_line)
                    if next_matches and not any(word in next_line.lower() for word in ['revenue', 'expense', 'asset', 'liability']):
                        for match in next_matches:
                            if any(c.isdigit() for c in match):
                                return match
        
        # THIRD: Fallback to immediate context around keyword
        lower_text = text.lower()
        pos = lower_text.find(keyword)
        
        if pos >= 0:
            context_start = max(0, pos - 50)
            context_end = min(len(text), pos + 150)
            context = text[context_start:context_end]
            
            pattern = r'[\d,]+(?:\.\d+)?(?:M|B|K)?(?:%)?'
            matches = re.findall(pattern, context)
            
            if matches:
                # Return the value closest to keyword (usually the last one in context)
                for match in reversed(matches):
                    if any(c.isdigit() for c in match):
                        return match
        
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
        """Extract numeric value following a keyword, with table-aware context"""
        # FIRST: Try table extraction for structured data (tables are more reliable)
        table_value = self._extract_value_from_table_context(text, keyword)
        if table_value:
            return table_value
        
        # SECOND: Try pattern matching for explicit values in prose
        escaped_keyword = re.escape(keyword)
        # Match keyword followed by : or whitespace, then capture number
        # But be careful not to match partial words
        pattern = f"\\b{escaped_keyword}[:\\s]+([\\$]?[\\d,.]+(?:M|B|K)?%?)"
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[0]
        
        return None
    
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
    
    def _assess_document_capability(self, 
                                   structural_chunks: List[StructuralChunk],
                                   file_id: str) -> Dict[str, any]:
        """
        NEW: Assess what types of metrics this document can support.
        
        Analyzes document content to determine which metrics are:
        - Directly available (explicitly stated)
        - Calculable (data present to compute)
        - Partially available (incomplete data)
        - Not available (missing required data)
        
        This prevents selecting metrics that sound important but can't be extracted.
        """
        logger.info(f"[CHUNKING:ASSESSMENT] Starting document capability assessment")
        
        document_text = " ".join([c.text.lower() for c in structural_chunks])
        
        capability_map = {
            "has_revenue": any(word in document_text for word in ["revenue", "sales", "income"]),
            "has_profitability": any(word in document_text for word in ["profit", "margin", "ebitda", "earnings"]),
            "has_cash_flow": any(word in document_text for word in ["cash flow", "operating cash"]),
            "has_balance_sheet": any(word in document_text for word in ["assets", "liabilities", "equity", "balance sheet"]),
            "has_liabilities": any(word in document_text for word in ["liabilities", "debt", "payable"]),
            "has_income_statement": any(word in document_text for word in ["net income", "net profit", "ebit", "operating income"]),
            "has_yoy_comparison": len(structural_chunks) > 2 or "2023" in document_text and "2024" in document_text,
            "is_saas": any(word in document_text for word in ["subscription", "saas", "recurring revenue", "mrr", "arr"]),
            "has_inventory": any(word in document_text for word in ["inventory", "inventory turnover"]),
        }
        
        # Count available sections
        available_sections = sum(1 for v in capability_map.values() if v)
        
        logger.info(f"[CHUNKING:ASSESSMENT] Document capability summary:")
        logger.info(f"[CHUNKING:ASSESSMENT]   Available sections: {available_sections}/9")
        logger.info(f"[CHUNKING:ASSESSMENT]   Has Revenue data: {capability_map['has_revenue']}")
        logger.info(f"[CHUNKING:ASSESSMENT]   Has Profitability data: {capability_map['has_profitability']}")
        logger.info(f"[CHUNKING:ASSESSMENT]   Has Balance Sheet data: {capability_map['has_balance_sheet']}")
        logger.info(f"[CHUNKING:ASSESSMENT]   Has Income Statement: {capability_map['has_income_statement']}")
        logger.info(f"[CHUNKING:ASSESSMENT]   Has YoY Comparison: {capability_map['has_yoy_comparison']}")
        logger.info(f"[CHUNKING:ASSESSMENT]   Is SaaS: {capability_map['is_saas']}")
        
        return capability_map
    
    def _check_metric_feasibility(self,
                                 metric_name: str,
                                 metric_type: MetricType,
                                 document_capability: Dict) -> Dict[str, any]:
        """
        NEW: Check if a metric can be extracted from this document.
        
        Returns feasibility assessment with reasons.
        """
        metric_lower = metric_name.lower()
        
        # Define what data is required for each metric
        requirements = {
            "revenue": {"required": ["has_revenue"], "note": "Direct data"},
            "growth": {"required": ["has_revenue", "has_yoy_comparison"], "note": "Need YoY comparison"},
            "margin": {"required": ["has_profitability"], "note": "Need profit data"},
            "ebitda": {"required": ["has_profitability", "has_income_statement"], "note": "Need complete income statement"},
            "roa": {"required": ["has_income_statement", "has_balance_sheet"], "note": "Need net income"},
            "roe": {"required": ["has_income_statement", "has_balance_sheet"], "note": "Need net income"},
            "ratio": {"required": ["has_balance_sheet"], "note": "Balance sheet metrics"},
            "current ratio": {"required": ["has_liabilities"], "note": "Need current liabilities"},
            "quick ratio": {"required": ["has_liabilities"], "note": "Need current liabilities"},
            "inventory": {"required": ["has_inventory"], "note": "Need inventory data"},
            "turnover": {"required": ["has_balance_sheet", "has_revenue"], "note": "Need assets and revenue"},
        }
        
        # Check what's required
        feasibility = "DIRECTLY_AVAILABLE"
        confidence = 1.0
        reason = ""
        
        for keyword, requirement in requirements.items():
            if keyword in metric_lower:
                required_caps = requirement["required"]
                available_caps = [cap for cap in required_caps if document_capability.get(cap, False)]
                
                if len(available_caps) == len(required_caps):
                    # All requirements met
                    feasibility = "DIRECTLY_AVAILABLE" if len(required_caps) <= 1 else "CALCULABLE"
                    reason = requirement["note"]
                elif available_caps:
                    # Partial data
                    feasibility = "PARTIAL_DATA"
                    confidence = len(available_caps) / len(required_caps)
                    reason = f"Have {len(available_caps)}/{len(required_caps)} required data types"
                else:
                    # No data
                    feasibility = "NOT_AVAILABLE"
                    confidence = 0.0
                    reason = f"Missing: {', '.join(required_caps)}"
                break
        
        # Special case: SaaS companies shouldn't have inventory metrics
        if document_capability.get("is_saas") and "inventory" in metric_lower:
            feasibility = "NOT_APPLICABLE"
            confidence = 0.0
            reason = "SaaS company (no inventory)"
        
        return {
            "metric": metric_name,
            "feasibility": feasibility,
            "confidence": confidence,
            "reason": reason,
            "should_attempt": feasibility in ["DIRECTLY_AVAILABLE", "CALCULABLE", "PARTIAL_DATA"]
        }

    def aggregate_metric_chunks(self,
                               structural_chunks: List[StructuralChunk],
                               file_id: str) -> List[MetricChunk]:
        """
        Step 2: LLM-DRIVEN METRIC SELECTION AND CHUNK CREATION (OPTIMIZED)
        
        IMPROVEMENTS:
        1. Document capability assessment (NEW)
        2. Metric feasibility checking (NEW)
        3. Detailed extraction logging (NEW)
        4. Gap analysis (NEW)
        5. Final summary (NEW)
        
        Args:
            structural_chunks: List of structural chunks
            file_id: Source file ID
            
        Returns:
            List of metric-centric chunks (only for LLM-selected metrics with score >= 6)
        """
        logger.info(f"[CHUNKING:METRIC] Starting OPTIMIZED LLM-driven metric selection for file {file_id}")
        logger.info(f"[CHUNKING:METRIC] Step 1: Assess document capability")
        
        # NEW: Assess what metrics the document can support
        doc_capability = self._assess_document_capability(structural_chunks, file_id)
        
        # Prepare document context from structural chunks
        document_excerpts = []
        for chunk in structural_chunks[:10]:  # Use first 10 chunks for context
            document_excerpts.append(f"[Chunk {chunk.chunk_index}]\n{chunk.text[:300]}")
        document_context = "\n\n".join(document_excerpts)
        
        logger.info(f"[CHUNKING:METRIC] Step 2: Evaluating metric groups with LLM (batch mode)")
        
        # Step 2: Use LLM to evaluate and select metrics from hardcoded list (OPTIMIZED BATCH)
        logger.info(f"[CHUNKING:METRIC] Step 2: Evaluating metric groups with LLM (batch mode)")
        selected_metrics = self._llm_evaluate_and_select_metrics(
            document_context,
            structural_chunks,
            file_id
        )
        
        # Identify which metrics are custom (not in predefined list)
        predefined_metric_names = set()
        for category, metrics in PREDEFINED_FINANCIAL_METRICS.items():
            for metric_name, metric_type in metrics:
                predefined_metric_names.add(metric_name.lower())
        
        custom_metric_names = set()
        for metric_name in selected_metrics.keys():
            if metric_name.lower() not in predefined_metric_names:
                custom_metric_names.add(metric_name)
                logger.info(f"[CHUNKING:METRIC] Identified custom metric: {metric_name}")
        
        if not selected_metrics:
            logger.warning(f"[CHUNKING:METRIC] No metrics selected by LLM, returning empty metric chunks")
            return []
        
        logger.info(f"[CHUNKING:METRIC] LLM selected {len(selected_metrics)} metrics (score >= 6.0)")
        for metric_name, score in list(selected_metrics.items())[:20]:  # Show first 20
            logger.info(f"[CHUNKING:METRIC]   ✓ {metric_name}: score={score:.1f}/10.0")
        
        # Step 3: Extract and aggregate data for SELECTED metrics only
        logger.info(f"[CHUNKING:METRIC] Step 3: Extract data for selected metrics from chunks")
        
        # NEW: Check feasibility of each selected metric BEFORE extraction
        logger.info(f"[CHUNKING:METRIC:FEASIBILITY] Analyzing extractability of {len(selected_metrics)} selected metrics")
        # doc_capability already computed at line 1080, reusing it (OPTIMIZATION: removed duplicate call)
        feasibility_map = {}
        feasible_metrics = []
        unfeasible_metrics = []
        
        for metric_name, score in selected_metrics.items():
            feasibility = self._check_metric_feasibility(metric_name, None, doc_capability)
            feasibility_map[metric_name] = feasibility
            
            if feasibility["should_attempt"]:
                feasible_metrics.append((metric_name, score, feasibility))
                logger.info(f"[CHUNKING:METRIC:FEASIBILITY]   ✓ {metric_name}: {feasibility['feasibility']} ({feasibility['reason']})")
            else:
                unfeasible_metrics.append((metric_name, score, feasibility))
                logger.info(f"[CHUNKING:METRIC:FEASIBILITY]   ⚠ {metric_name}: {feasibility['feasibility']} ({feasibility['reason']})")
        
        logger.info(f"[CHUNKING:METRIC:FEASIBILITY] Proceeding with {len(feasible_metrics)}/{len(selected_metrics)} feasible metrics")
        
        # Create a map of chunks indexed by metric name
        metric_chunk_data: Dict[str, Dict] = {}
        extraction_log = {}  # Track extraction results for each metric
        
        # For debugging: track which metrics we're looking for
        logger.debug(f"[CHUNKING:METRIC] Looking for these {len(selected_metrics)} selected metrics:")
        for metric_name in list(selected_metrics.keys())[:20]:
            logger.debug(f"[CHUNKING:METRIC]   - {metric_name}")
        
        # OPTIMIZED: Extract metrics from ALL chunks in ONE LLM call instead of per-chunk calls
        logger.info(f"[CHUNKING:METRIC:EXTRACTION:BATCH] OPTIMIZED: Extracting metrics from {len(structural_chunks)} chunks in ONE LLM call")
        all_metrics_all_chunks = self._extract_metrics_batch(structural_chunks, selected_metrics)
        
        # Build metric_chunk_data from batched extraction results
        for struct_chunk, all_metrics in zip(structural_chunks, all_metrics_all_chunks):
            if all_metrics:
                logger.debug(f"[CHUNKING:METRIC] Chunk {struct_chunk.chunk_index}: Found {len(all_metrics)} metrics")
                for m in all_metrics:
                    logger.debug(f"[CHUNKING:METRIC]   Found: {m.metric_name}")
            
            # Keep only metrics that were selected by LLM
            for metric in all_metrics:
                metric_key = metric.metric_name.lower()
                
                # Check if this metric was selected
                selected_metric_name = None
                for sel_name in selected_metrics.keys():
                    if sel_name.lower() == metric_key or self._metrics_match(sel_name, metric.metric_name):
                        selected_metric_name = sel_name
                        logger.debug(f"[CHUNKING:METRIC] Matched '{metric.metric_name}' to selected '{sel_name}'")
                        break
                
                if selected_metric_name:
                    if selected_metric_name not in metric_chunk_data:
                        metric_chunk_data[selected_metric_name] = {
                            "metric_type": metric.metric_type,
                            "occurrences": [],
                            "source_chunks": [],
                            "texts": [],
                            "score": selected_metrics[selected_metric_name],
                            "is_from_table": self._is_table_chunk(struct_chunk.text)
                        }
                        extraction_log[selected_metric_name] = {
                            "score": selected_metrics[selected_metric_name],
                            "status": "EXTRACTING",
                            "chunks_found": [],
                            "values_found": 0,
                            "confidence": 0.0
                        }
                    
                    metric.structural_chunk_id = struct_chunk.point_id
                    metric_chunk_data[selected_metric_name]["occurrences"].append(metric)
                    extraction_log[selected_metric_name]["chunks_found"].append(struct_chunk.chunk_index)
                    extraction_log[selected_metric_name]["values_found"] += 1
        
        # NEW: Log extraction results and identify gaps
        logger.info(f"[CHUNKING:METRIC:EXTRACTION] Summary of extraction attempts:")
        extracted_count = 0
        partial_count = 0
        failed_count = 0
        
        for metric_name, score in selected_metrics.items():
            if metric_name in metric_chunk_data:
                occurrences = metric_chunk_data[metric_name]["occurrences"]
                if occurrences:
                    extracted_count += 1
                    logger.info(f"[CHUNKING:METRIC:EXTRACTION]   ✓ {metric_name} (score={score:.1f}): {len(occurrences)} occurrences found")
                else:
                    partial_count += 1
                    logger.info(f"[CHUNKING:METRIC:EXTRACTION]   ⚠ {metric_name} (score={score:.1f}): Data structure present but no values extracted")
            else:
                failed_count += 1
                feasibility = feasibility_map.get(metric_name, {})
                reason = feasibility.get("reason", "Unknown reason")
                logger.info(f"[CHUNKING:METRIC:EXTRACTION]   ✗ {metric_name} (score={score:.1f}): Not found ({reason})")
        
        logger.info(f"[CHUNKING:METRIC:EXTRACTION:SUMMARY] Extraction results:")
        logger.info(f"[CHUNKING:METRIC:EXTRACTION:SUMMARY]   Successfully extracted: {extracted_count}/{len(selected_metrics)}")
        logger.info(f"[CHUNKING:METRIC:EXTRACTION:SUMMARY]   Partial/incomplete: {partial_count}/{len(selected_metrics)}")
        logger.info(f"[CHUNKING:METRIC:EXTRACTION:SUMMARY]   Not found: {failed_count}/{len(selected_metrics)}")
        logger.info(f"[CHUNKING:METRIC:EXTRACTION:SUMMARY]   Success rate: {extracted_count/max(len(selected_metrics),1)*100:.1f}%")
        
        # Log summary of extraction
        logger.info(f"[CHUNKING:METRIC] Step 3 complete: Found data for {len(metric_chunk_data)} of {len(selected_metrics)} selected metrics")
        metrics_without_data = set(selected_metrics.keys()) - set(metric_chunk_data.keys())
        if metrics_without_data:
            logger.debug(f"[CHUNKING:METRIC] Metrics selected by LLM but not found in chunks ({len(metrics_without_data)}):")
            for metric_name in list(metrics_without_data)[:10]:  # Show first 10
                logger.debug(f"[CHUNKING:METRIC]   - {metric_name}")

        
        # Step 4: Create metric-centric chunks for selected metrics (OPTIMIZED - BATCHED SYNTHESIS)
        logger.info(f"[CHUNKING:METRIC] Step 4: Creating metric-centric chunks (batched synthesis)")
        
        # Pre-compute best values for all metrics
        metrics_to_synthesize = {}
        for metric_name, metric_data in metric_chunk_data.items():
            if metric_data["occurrences"]:
                best_values = self._select_metric_values(
                    metric_name,
                    metric_data["occurrences"],
                    metric_data["texts"]
                )
                metrics_to_synthesize[metric_name] = {
                    "data": metric_data,
                    "best_values": best_values
                }
        
        # OPTIMIZED: Single LLM call for all metric syntheses instead of per-metric calls
        batch_synthesis_results = {}
        if self.llm and metrics_to_synthesize:
            batch_synthesis_results = self._batch_synthesize_metrics(
                metrics_to_synthesize,
                document_context
            )
        
        # Create chunks using synthesized text
        metric_chunks = []
        for metric_name, metric_info in metrics_to_synthesize.items():
            metric_data = metric_info["data"]
            best_values = metric_info["best_values"]
            
            # Use batched synthesis result, fallback to template
            aggregated_text = batch_synthesis_results.get(metric_name)
            if not aggregated_text:
                # Fallback: Template-based synthesis (no LLM)
                aggregated_text = self._template_metric_synthesis(
                    metric_name,
                    metric_data,
                    best_values
                )
            
            # Check if this is a custom metric
            is_custom = metric_name in custom_metric_names
            
            source_chunks = [cid for cid in metric_data["source_chunks"] if cid is not None]
            metric_chunk = MetricChunk(
                metric_name=metric_name,
                metric_type=metric_data["metric_type"],
                chunk_id=str(uuid.uuid4()),
                text=aggregated_text,
                occurrences=metric_data["occurrences"],
                source_chunk_ids=source_chunks,
                is_from_table=metric_data["is_from_table"],
                is_custom_metric=is_custom,
                file_id=file_id,
                timestamp=datetime.now().isoformat(),
                sources=source_chunks,
                confidence=metric_data["score"] / 10.0  # Normalize score to 0-1
            )
            
            metric_chunks.append(metric_chunk)
            logger.info(f"[CHUNKING:METRIC]   ✓ Created chunk for '{metric_name}'")
        
        # Step 4B: SYNTHESIS OF MISSING CUSTOM METRICS (OPTIMIZED - BATCHED)
        logger.info(f"[CHUNKING:METRIC] Step 4B: Synthesizing missing custom metrics (batched)")
        
        missing_metrics = set(selected_metrics.keys()) - set(m.metric_name for m in metric_chunks)
        custom_missing = [m for m in missing_metrics if m in custom_metric_names]
        
        if custom_missing:
            logger.info(f"[CHUNKING:METRIC:SYNTHESIS] Found {len(custom_missing)} custom metrics to synthesize")
            
            # OPTIMIZED: Synthesize all custom metrics in ONE LLM call
            synthesized_chunks_dict = self._batch_synthesize_custom_metrics(
                custom_missing,
                selected_metrics,
                document_context
            )
            
            # Add all synthesized chunks
            for metric_name, synthesized_chunk in synthesized_chunks_dict.items():
                if synthesized_chunk:
                    synthesized_chunk.file_id = file_id
                    metric_chunks.append(synthesized_chunk)
                    logger.info(f"[CHUNKING:METRIC:SYNTHESIS]   ✓ Synthesized: '{metric_name}'")
                else:
                    logger.warning(f"[CHUNKING:METRIC:SYNTHESIS]   ✗ Failed: '{metric_name}'")
        else:
            logger.debug(f"[CHUNKING:METRIC:SYNTHESIS] No custom metrics to synthesize")
        
        # Final comprehensive summary with gap analysis
        logger.info(f"[CHUNKING:METRIC:FINAL_SUMMARY] Metric chunk creation completed")
        logger.info(f"[CHUNKING:METRIC:FINAL_SUMMARY] {'═' * 80}")
        logger.info(f"[CHUNKING:METRIC:FINAL_SUMMARY] Requested: {len(selected_metrics)} metrics selected by LLM")
        logger.info(f"[CHUNKING:METRIC:FINAL_SUMMARY] Extracted: {extracted_count} metrics with data found")
        logger.info(f"[CHUNKING:METRIC:FINAL_SUMMARY] Partial: {partial_count} metrics with incomplete data")
        logger.info(f"[CHUNKING:METRIC:FINAL_SUMMARY] Failed: {failed_count} metrics not found")
        logger.info(f"[CHUNKING:METRIC:FINAL_SUMMARY] Synthesized: {len(custom_missing)} custom metrics synthesized")
        logger.info(f"[CHUNKING:METRIC:FINAL_SUMMARY] Created: {len(metric_chunks)} metric chunks total")
        logger.info(f"[CHUNKING:METRIC:FINAL_SUMMARY] Success rate: {len(metric_chunks)/max(len(selected_metrics),1)*100:.1f}%")
        
        custom_count = sum(1 for m in metric_chunks if m.is_custom_metric)
        if custom_count > 0:
            logger.info(f"[CHUNKING:METRIC:FINAL_SUMMARY] Composition:")
            logger.info(f"[CHUNKING:METRIC:FINAL_SUMMARY]   ├─ Predefined metrics: {len(metric_chunks) - custom_count}")
            logger.info(f"[CHUNKING:METRIC:FINAL_SUMMARY]   └─ Custom/synthesized metrics: {custom_count}")
        
        # Log metrics that couldn't be created
        still_missing = set(selected_metrics.keys()) - set(m.metric_name for m in metric_chunks)
        if still_missing:
            logger.info(f"[CHUNKING:METRIC:FINAL_SUMMARY] Metrics not created ({len(still_missing)}):")
            for metric_name in sorted(still_missing):
                feasibility = feasibility_map.get(metric_name, {})
                reason = feasibility.get("reason", "Unknown")
                score = selected_metrics[metric_name]
                logger.info(f"[CHUNKING:METRIC:FINAL_SUMMARY]   ✗ {metric_name} (score={score:.1f}): {reason}")
        
        logger.info(f"[CHUNKING:METRIC:FINAL_SUMMARY] {'═' * 80}")
        
        return metric_chunks
    
    def _llm_evaluate_and_select_metrics(self,
                                        document_context: str,
                                        structural_chunks: List[StructuralChunk],
                                        file_id: str) -> Dict[str, float]:
        """
        OPTIMIZED: Evaluate hardcoded metrics and group them by category.
        
        This is the core LLM-driven selection mechanism (OPTIMIZED):
        - LLM reads document context
        - LLM groups metrics by category (Revenue, Profitability, Cash Flow, etc.)
        - LLM evaluates metric groups (5-8 metrics per group) for importance
        - LLM returns scores for metric groups (1-10 scale)
        - Service filters to keep only groups with score >= 6
        - Creates individual metric chunks from group selections
        
        OPTIMIZATIONS:
        1. Batch evaluation: Groups of related metrics instead of individual metrics
        2. Structured JSON output: LLM returns JSON for efficient parsing
        3. No intermediate reasoning: Eliminates token-heavy explanations
        4. Grouped batch generation: Fewer LLM calls, more metrics per call
        
        Returns:
            Dict mapping metric_name -> score (1-10 scale) for selected metrics only
        """
        if not self.llm:
            logger.warning(f"[CHUNKING:METRIC] No LLM available, cannot perform metric selection")
            return {}
        
        try:
            logger.info(f"[CHUNKING:METRIC:OPT] Starting OPTIMIZED batch metric evaluation (grouping by category)")
            logger.info(f"[CHUNKING:METRIC:OPT] Evaluating {len(PREDEFINED_FINANCIAL_METRICS)} metric categories in groups")
            
            # Step 1: Group metrics by category
            metric_groups = {}
            for category, metrics in PREDEFINED_FINANCIAL_METRICS.items():
                metric_names = [m[0] for m in metrics]
                metric_groups[category] = metric_names
            
            # Step 2: Create batch evaluation prompt for ALL metric groups at once
            # Prepare groups as JSON for structured response
            groups_json = {}
            for category, metric_names in metric_groups.items():
                groups_json[category] = metric_names[:8]  # Limit to 8 metrics per group for prompt efficiency
            
            groups_json_str = json.dumps(groups_json, indent=2)
            
            # Create OPTIMIZED LLM prompt:
            # - Structured JSON output (not prose explanations)
            # - Batch evaluation (all groups at once)
            # - No intermediate reasoning (direct scores)
            # - Materiality-focused: Prioritize importance over extractability
            # - Allow synthesis: LLM can create derived/aggregated metrics
            # - Table awareness: Recognize table structures and create summary metrics
            evaluation_prompt = f"""FINANCIAL ANALYST METRIC SELECTION - MATERIALITY-FOCUSED

DOCUMENT CONTENT (excerpts):
{document_context[:2000]}

METRIC GROUPS TO EVALUATE (JSON format):
{groups_json_str}

TASK: Select metrics for financial summarization based on MATERIALITY and IMPORTANCE, not just data availability.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CRITICAL PRINCIPLE:
Metrics should serve SUMMARIZATION purposes, not just data extraction.
- SELECT metrics that matter for understanding financial health
- DO NOT SELECT individual line items just because they exist in data
- SYNTHESIZE missing but important metrics (e.g., calculate totals, trends, ratios)
- IDENTIFY tables and create aggregate/comparative metrics from the whole table

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SCORING RULES (based on MATERIALITY for summarization):

1. MATERIALITY ASSESSMENT (Primary - 0-10 scale):
   9-10 points: Core financial statement items (Total Assets, Total Liabilities, Net Income, Revenue)
   8-9 points: Major category aggregates (Gross Profit, Operating Income, Total Expenses)
   7-8 points: Important subcategories or ratios (Current Assets, Debt Ratio, ROE)
   6-7 points: Operational metrics (Operating Margin, Asset Turnover, Growth Rates)
   5-6 points: Supporting metrics (Individual expense categories, segment data if relevant)
   1-4 points: Detail line items NOT aggregated (skip unless required for aggregate)

2. SYNTHESIS BONUS (+2 to +3):
   Grant bonus if the metric:
   - Can be calculated/derived from available data (e.g., Total = Sum of components)
   - Represents a trend (YoY change, quarterly progression) even if not explicitly stated
   - Aggregates a table (e.g., "Production Cost Summary" from a detailed breakdown)

3. TABLE STRUCTURE DETECTION (+1 to +3):
   Identify table types and create appropriate summary metrics:
   - BREAKDOWN table (e.g., Cost components): Create "Total Cost" aggregate metric
   - TIME SERIES table (e.g., quarterly data): Create "Trend Analysis" or "YoY Growth" metric
   - COMPARISON table (e.g., 2022 vs 2023): Create "Growth Rate" or "Change Analysis" metric
   - MATRIX table (e.g., product × region): Create "By Segment Summary" metric

4. EXAMPLES:
   ✓ SELECT "Total Production Cost" (aggregate multiple line items) - GOOD for summarization
   ✗ AVOID selecting "Material Cost" alone (just one component) - NOT good for summarization
   ✓ SELECT "Revenue Growth Rate (YoY %)" (synthesized trend metric)
   ✗ AVOID selecting individual revenue items unless aggregating them

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OUTPUT FORMAT (JSON only, no explanations):
{{
  "Revenue Metrics": {{"score": 9, "selected": ["Total Revenue", "Revenue Growth Rate (YoY %)"]}},
  "Profitability Metrics": {{"score": 8, "selected": ["Gross Profit Total", "Operating Margin %"]}},
  "Balance Sheet": {{"score": 9, "selected": ["Total Assets", "Total Liabilities", "Equity Aggregate"]}},
  ...
}}

Rules for output:
- Include score (1-10) based on materiality assessment above
- For "selected" array: Include BOTH extracted AND synthesized metrics
- Only include groups with score >= 6
- Prefer aggregates and summarization metrics over detail line items
- If table detected, include table-level summary metrics (not individual rows)
- You MAY create 1-2 custom metrics if they improve summarization (e.g., "Total Operating Costs")"""
            
            logger.debug(f"[CHUNKING:METRIC:OPT] Sending materiality-focused metric selection prompt")
            logger.debug(f"[CHUNKING:METRIC:OPT] Reminder: LLM will select based on importance for summarization, not just data presence")
            
            # Call LLM ONCE for all groups (batch mode)
            response = self.llm.invoke(evaluation_prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            logger.debug(f"[CHUNKING:METRIC:OPT] LLM batch response received: {len(response_text)} chars")
            
            # Step 3: Parse JSON response - OPTIMIZED parsing (structured format)
            scores = {}
            custom_metrics = []
            
            try:
                # Extract JSON from response (handle potential markdown code blocks)
                json_match = re.search(r'\{[\s\S]*\}', response_text)
                if json_match:
                    json_str = json_match.group(0)
                    response_json = json.loads(json_str)
                    
                    # Parse each group's results
                    for group_name, group_data in response_json.items():
                        if isinstance(group_data, dict) and 'selected' in group_data:
                            selected_metrics = group_data.get('selected', [])
                            group_score = group_data.get('score', 5)
                            
                            # Convert score to float and clamp
                            try:
                                group_score = float(group_score)
                                group_score = max(1.0, min(10.0, group_score))
                            except (ValueError, TypeError):
                                group_score = 5.0
                            
                            # Only include selected metrics from high-scoring groups
                            if group_score >= 6.0:
                                logger.debug(f"[CHUNKING:METRIC:OPT] Group '{group_name}': score={group_score:.1f}, metrics={len(selected_metrics)}")
                                
                                for metric_name in selected_metrics:
                                    if metric_name and isinstance(metric_name, str):
                                        # Use group score as baseline, adjusted for position in selection
                                        metric_score = group_score
                                        scores[metric_name.strip()] = metric_score
                                        logger.debug(f"[CHUNKING:METRIC:OPT]   ✓ {metric_name.strip()}: {metric_score:.1f}")
                            else:
                                logger.debug(f"[CHUNKING:METRIC:OPT] Group '{group_name}': score={group_score:.1f} (below threshold, skipped)")
                    
                else:
                    logger.warning(f"[CHUNKING:METRIC:OPT] Could not find JSON in response, attempting fallback parsing")
                    # Fallback to line-by-line parsing for unexpected formats
                    lines = response_text.split('\n')
                    for line in lines:
                        line = line.strip()
                        if not line or len(line) > 150:
                            continue
                        
                        # Try to extract metric: score pattern
                        match = re.match(r'^-?\s*["\']?(.+?)["\']?\s*:\s*(\d+(?:\.\d+)?)', line)
                        if match:
                            metric_name = match.group(1).strip()
                            try:
                                score = float(match.group(2))
                                score = max(1.0, min(10.0, score))
                                if score >= 6.0:
                                    scores[metric_name] = score
                            except ValueError:
                                pass
            
            except json.JSONDecodeError as e:
                logger.warning(f"[CHUNKING:METRIC:OPT] JSON parsing failed: {e}, using fallback")
                # Fallback: parse line by line
                lines = response_text.split('\n')
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Extract "metric: score" or "metric: score" patterns
                    match = re.search(r'^-?\s*["\']?(.+?)["\']?\s*:\s*(\d+(?:\.\d+)?)', line)
                    if match:
                        metric_name = match.group(1).strip()
                        try:
                            score = float(match.group(2))
                            score = max(1.0, min(10.0, score))
                            if score >= 6.0 and metric_name not in scores:
                                scores[metric_name] = score
                        except ValueError:
                            pass
            
            # Step 4: Convert scores dict to selected metrics with score >= 6.0
            SELECTION_THRESHOLD = 6.0
            selected = {name: score for name, score in scores.items() if score >= SELECTION_THRESHOLD}
            
            # Log summary
            logger.info(f"[CHUNKING:METRIC:OPT] ✓ Batch evaluation complete: {len(selected)} metrics selected (score >= {SELECTION_THRESHOLD})")
            if selected:
                for metric_name, score in sorted(selected.items(), key=lambda x: x[1], reverse=True)[:15]:
                    logger.debug(f"[CHUNKING:METRIC:OPT]   ✓ {metric_name}: {score:.1f}/10.0")
            
            return selected
            
        except Exception as e:
            logger.error(f"[CHUNKING:METRIC:OPT] Batch metric evaluation failed: {e}")
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
        """
        Check if two metric names refer to the same metric
        Uses fuzzy matching to handle variations in naming
        """
        m1_lower = metric1.lower().strip()
        m2_lower = metric2.lower().strip()
        
        # Exact match after normalization
        if m1_lower == m2_lower:
            return True
        
        # Check if one is substring of other
        if m1_lower in m2_lower or m2_lower in m1_lower:
            return True
        
        # Extract key words for fuzzy matching
        # Remove common words and punctuation
        m1_words = set(w for w in m1_lower.split() if len(w) > 2)
        m2_words = set(w for w in m2_lower.split() if len(w) > 2)
        
        # If significant word overlap (>50%), consider it a match
        if m1_words and m2_words:
            overlap = len(m1_words & m2_words)
            total = len(m1_words | m2_words)
            if total > 0 and overlap / total >= 0.5:
                return True
        
        # Special handling for common metric abbreviations/variations
        variations = {
            'revenue': ['sales', 'turnover', 'net sales', 'gross revenue', 'total revenue'],
            'net income': ['profit', 'net profit', 'earnings', 'bottom line'],
            'cash flow': ['cffo', 'operating cash', 'fcf', 'free cash flow'],
            'margin': ['profitability', 'margin %'],
            'debt': ['liabilities', 'leverage'],
            'assets': ['asset', 'total assets'],
            'equity': ['stockholders equity', 'shareholders equity'],
        }
        
        for key, alts in variations.items():
            if (key in m1_lower and any(alt in m2_lower for alt in alts)) or \
               (key in m2_lower and any(alt in m1_lower for alt in alts)):
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
    
    def _extract_metrics_batch(self, structural_chunks: List[StructuralChunk], selected_metrics: Dict[str, float]) -> List[List[MetricOccurrence]]:
        """
        OPTIMIZED: Extract metrics from ALL chunks in a single LLM call instead of per-chunk calls.
        Significantly reduces LLM inference time by batching multiple extraction requests.
        
        Args:
            structural_chunks: List of structural chunks to extract metrics from
            selected_metrics: Dict of selected metric names and scores
            
        Returns:
            List of metric lists (one per structural chunk)
        """
        if not self.llm:
            # Fallback to rule-based extraction for each chunk
            return [self._extract_metrics_from_text(chunk.text) for chunk in structural_chunks]
        
        try:
            logger.debug(f"[CHUNKING:METRIC:EXTRACTION:BATCH] Batching extraction for {len(structural_chunks)} chunks")
            
            # Prepare chunk texts for batch extraction
            chunk_texts = [chunk.text[:5000] for chunk in structural_chunks]  # Limit to 5k chars per chunk
            selected_metric_names = list(selected_metrics.keys())
            
            # Create batch extraction prompt
            batch_prompt = f"""You are a financial analysis expert. Extract metrics from {len(structural_chunks)} document excerpts.

Looking for these metrics (or close matches):
{', '.join(selected_metric_names[:30])}

For each document excerpt, extract ALL relevant financial metrics found, including:
- Primary metrics explicitly stated
- Values with units and periods
- Any metrics matching the names above or similar

For each metric found, provide (pipe-separated):
metric_name|metric_type|value|period|confidence

Document excerpts to analyze:
{chr(10).join([f"--- EXCERPT {i+1} ---\n{text}\n" for i, text in enumerate(chunk_texts)])}

For each excerpt, list metrics found (format: metric_name|type|value|period|confidence). Separate excerpts with a blank line.

CRITICAL: 
- Match metrics to the list above as closely as possible
- Include period/year with all values
- Only report metrics actually found in the excerpts
- Be specific with metric types"""
            
            response = self.llm.invoke(batch_prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # DEBUG: Log raw response for debugging
            logger.info(f"[CHUNKING:METRIC:EXTRACTION:BATCH:DEBUG] Raw LLM response ({len(response_text)} chars):\n{response_text[:1500]}")
            
            # Parse response - try intelligent parsing with fallback
            all_chunk_metrics = []
            
            # Strategy 1: Look for lines with pipe separators (pipe-separated format)
            lines_with_pipes = [line for line in response_text.split('\n') 
                              if '|' in line and not any(x in line.lower() for x in ['metric_name', 'excerpt', '---', '==', 'looking', 'for'])]
            
            logger.debug(f"[CHUNKING:METRIC:EXTRACTION:BATCH:DEBUG] Found {len(lines_with_pipes)} lines with pipe separators")
            
            if lines_with_pipes:
                # Use pipe-separated parsing
                current_chunk_idx = 0
                current_metrics = []
                
                for line in response_text.split('\n'):
                    stripped = line.strip()
                    
                    # Check if this is an excerpt header
                    if 'EXCERPT' in line.upper():
                        if current_metrics and current_chunk_idx < len(structural_chunks):
                            all_chunk_metrics.append(current_metrics)
                            current_metrics = []
                        current_chunk_idx += 1
                        continue
                    
                    # Blank lines might separate sections
                    if not stripped:
                        continue
                    
                    # Skip headers and metadata
                    if any(x in line.lower() for x in ['metric_name', 'excerpt', '---', '==', 'looking', 'for', 'provide', 'format']):
                        continue
                    
                    # Try to parse as metric line
                    if '|' in line:
                        parts = [p.strip() for p in line.split('|')]
                        if len(parts) >= 4:  # At least: name|type|value|period (confidence optional)
                            try:
                                metric_name = parts[0]
                                metric_type_str = parts[1].lower()
                                value = parts[2] if parts[2] and parts[2].lower() != 'n/a' else None
                                period = parts[3] if len(parts) > 3 and parts[3] and parts[3].lower() != 'n/a' else None
                                confidence = 0.85  # Default confidence
                                
                                if len(parts) > 4:
                                    try:
                                        confidence = float(parts[4])
                                    except ValueError:
                                        confidence = 0.85
                                
                                # Skip if metric name is a header/placeholder
                                if metric_name.lower() in ['metric_name', 'name', 'excerpt', '---']:
                                    continue
                                    
                                # Skip historical markers
                                if value and value.lower() == 'historical':
                                    continue
                                
                                # Skip if no value and no period
                                if not value and not period:
                                    continue
                                
                                metric_type = self._get_metric_type(metric_type_str)
                                
                                occurrence = MetricOccurrence(
                                    metric_name=metric_name,
                                    metric_type=metric_type,
                                    value=value,
                                    period=period,
                                    confidence=confidence
                                )
                                current_metrics.append(occurrence)
                                logger.debug(f"[CHUNKING:METRIC:EXTRACTION:BATCH:DEBUG] Parsed: {metric_name} = {value} ({period})")
                                
                            except (ValueError, IndexError) as parse_err:
                                logger.debug(f"[CHUNKING:METRIC:EXTRACTION:BATCH:DEBUG] Failed to parse line: {line} ({parse_err})")
                                continue
                
                # Add any remaining metrics
                if current_metrics:
                    all_chunk_metrics.append(current_metrics)
            
            else:
                logger.warning(f"[CHUNKING:METRIC:EXTRACTION:BATCH] No pipe-separated metrics found in LLM response")
                logger.info(f"[CHUNKING:METRIC:EXTRACTION:BATCH] Response preview:\n{response_text[:800]}")
            
            # Pad with empty lists if we didn't get all chunks
            while len(all_chunk_metrics) < len(structural_chunks):
                all_chunk_metrics.append([])
            
            # Truncate if we got too many
            all_chunk_metrics = all_chunk_metrics[:len(structural_chunks)]
            
            # DEBUG: Log extraction results per chunk
            for i, metrics in enumerate(all_chunk_metrics):
                if metrics:
                    logger.debug(f"[CHUNKING:METRIC:EXTRACTION:BATCH:DEBUG] Chunk {i+1}: Found {len(metrics)} metrics")
                    for m in metrics[:3]:
                        logger.debug(f"  - {m.metric_name}: {m.value} ({m.period})")
                else:
                    logger.debug(f"[CHUNKING:METRIC:EXTRACTION:BATCH:DEBUG] Chunk {i+1}: No metrics extracted")
            
            total_metrics = sum(len(m) for m in all_chunk_metrics)
            logger.info(f"[CHUNKING:METRIC:EXTRACTION:BATCH] ✓ Batch extraction complete: {total_metrics} metrics from {len(structural_chunks)} chunks")
            
            # FALLBACK: If batch extraction returned 0 metrics, try per-chunk extraction
            if total_metrics == 0:
                logger.warning(f"[CHUNKING:METRIC:EXTRACTION:BATCH] Batch extraction returned 0 metrics - using fallback per-chunk extraction")
                fallback_results = [self._extract_all_metric_mentions(chunk.text, chunk) for chunk in structural_chunks]
                total_fallback = sum(len(m) for m in fallback_results)
                logger.info(f"[CHUNKING:METRIC:EXTRACTION:BATCH] ✓ Fallback extraction found {total_fallback} metrics")
                return fallback_results
            
            return all_chunk_metrics
            
        except Exception as e:
            logger.warning(f"[CHUNKING:METRIC:EXTRACTION:BATCH] Batch extraction failed: {e}, falling back to per-chunk extraction")
            # Fallback: extract per chunk (slower but reliable)
            return [self._extract_all_metric_mentions(chunk.text, chunk) for chunk in structural_chunks]
    
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
        OPTIMIZED: Synthesize metric-centric text with structured format (less verbose).
        
        Creates a compact, structured summary instead of narrative prose:
        - Uses JSON structure for efficient parsing
        - Removes explanatory reasoning (saves tokens)
        - Focuses on essential facts only
        - Optimized for LLM embedding and retrieval
        
        Args:
            metric_name: Name of metric
            source_texts: Texts mentioning this metric
            occurrences: Metric occurrences with relevance scores
            best_values: Selected metric values dict (optional)
            
        Returns:
            Structured metric text optimized for LLM processing
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
        
        # Use OPTIMIZED LLM synthesis if available - structured JSON output
        if self.llm and unique_sentences:
            try:
                logger.debug(f"[CHUNKING:SYNTHESIS:OPT] Using optimized LLM to synthesize metric: {metric_name}")
                
                # Build value summary if available
                value_context = ""
                if best_values:
                    if best_values.get('latest_value'):
                        value_context = f"Latest value: {best_values['latest_value']}"
                        if best_values.get('period'):
                            value_context += f" ({best_values['period']})"
                    if best_values.get('all_values'):
                        value_context += f" | Historical: {' -> '.join(best_values['all_values'][:3])}"
                
                # Prepare context for LLM
                context = "\n".join(unique_sentences[:5])  # Use only top 5 most relevant sentences
                
                # OPTIMIZED prompt: Structured JSON output, no explanations
                synthesis_prompt = f"""METRIC SYNTHESIS - STRUCTURED OUTPUT ONLY

Metric: {metric_name}
Values: {value_context or 'N/A'}

Document mentions:
{context}

Generate ONLY a JSON object with these exact fields (no markdown, no explanations):
{{
  "definition": "one-line definition of this metric",
  "latest_value": "{best_values.get('latest_value', 'N/A') if best_values else 'N/A'}",
  "period": "{best_values.get('period', 'N/A') if best_values else 'N/A'}",
  "trend": "increasing/decreasing/stable",
  "key_fact": "most important single fact about this metric",
  "context": "one sentence about business impact"
}}"""
                
                # Call LLM for synthesis
                response = self.llm.invoke(synthesis_prompt)
                response_text = response.content if hasattr(response, 'content') else str(response)
                
                # Try to extract JSON
                try:
                    json_match = re.search(r'\{[\s\S]*\}', response_text)
                    if json_match:
                        json_str = json_match.group(0)
                        metric_json = json.loads(json_str)
                        
                        # Build structured but readable output
                        synthesis_parts = [
                            f"METRIC: {metric_name}",
                            f"DEFINITION: {metric_json.get('definition', 'N/A')}",
                            f"VALUE: {metric_json.get('latest_value', 'N/A')} ({metric_json.get('period', 'N/A')})",
                            f"TREND: {metric_json.get('trend', 'N/A')}",
                            f"KEY_FACT: {metric_json.get('key_fact', 'N/A')}",
                            f"CONTEXT: {metric_json.get('context', 'N/A')}"
                        ]
                        
                        logger.debug(f"[CHUNKING:SYNTHESIS:OPT] ✓ Structured synthesis successful for {metric_name}")
                        return " | ".join(synthesis_parts)
                except (json.JSONDecodeError, AttributeError):
                    logger.debug(f"[CHUNKING:SYNTHESIS:OPT] JSON parsing failed, using response as-is")
                    return response_text[:500]  # Return first 500 chars of response
            
            except Exception as e:
                logger.warning(f"[CHUNKING:SYNTHESIS:OPT] LLM synthesis failed for {metric_name}: {e}, using fallback")
                # Fall through to rule-based synthesis
        
        # OPTIMIZED Rule-based synthesis (fallback or when no LLM)
        logger.debug(f"[CHUNKING:SYNTHESIS:OPT] Using optimized rule-based synthesis for {metric_name}")
        
        synthesis_parts = [f"METRIC: {metric_name}"]
        
        # Add best values if available
        if best_values:
            if best_values.get('latest_value'):
                value_line = f"VALUE: {best_values['latest_value']}"
                if best_values.get('period'):
                    value_line += f" ({best_values['period']})"
                synthesis_parts.append(value_line)
            if best_values.get('all_values') and len(best_values['all_values']) > 1:
                synthesis_parts.append(f"HISTORY: {' -> '.join(best_values['all_values'][:3])}")
        
        # Add key fact (first sentence only)
        if unique_sentences:
            synthesis_parts.append(f"FACT: {unique_sentences[0][:100]}")
        
        # Add metric metadata
        if occurrences:
            synthesis_parts.append(f"TYPE: {occurrences[0].metric_type.value}")
            synthesis_parts.append(f"CONFIDENCE: {sum(o.confidence for o in occurrences) / len(occurrences):.0%}")
        
        return " | ".join(synthesis_parts)
    
    def _batch_synthesize_metrics(self, metrics_to_synthesize: Dict, document_context: str) -> Dict[str, str]:
        """
        OPTIMIZED: Synthesize all metrics in ONE LLM call instead of per-metric.
        
        Args:
            metrics_to_synthesize: Dict of {metric_name: {data, best_values}}
            document_context: Document excerpt for context
            
        Returns:
            Dict mapping metric_name -> synthesized_text
        """
        if not self.llm or not metrics_to_synthesize:
            return {}
        
        try:
            logger.info(f"[CHUNKING:SYNTHESIS:BATCH] Synthesizing {len(metrics_to_synthesize)} metrics in ONE LLM call")
            
            # Build batch synthesis prompt
            metrics_list = []
            for metric_name, info in metrics_to_synthesize.items():
                best_values = info["best_values"]
                value_str = best_values.get('latest_value', 'N/A')
                period_str = best_values.get('period', 'N/A')
                metrics_list.append(f"- {metric_name} (value: {value_str}, period: {period_str})")
            
            synthesis_prompt = f"""BATCH METRIC TEXT SYNTHESIS

Document Context:
{document_context[:2000]}

Synthesize brief descriptions for these {len(metrics_to_synthesize)} metrics:

{chr(10).join(metrics_list)}

For each metric, provide a concise one-line summary focusing on:
1. What the metric measures
2. Current value/trend
3. Business significance

RESPOND WITH ONLY JSON (no markdown, no explanations):
{{
  "metric_name_1": "one-line description with value and context",
  "metric_name_2": "one-line description with value and context",
  ...
}}"""
            
            logger.debug(f"[CHUNKING:SYNTHESIS:BATCH] Sending batch synthesis prompt ({len(synthesis_prompt)} chars)")
            
            # SINGLE LLM CALL for all metrics
            response = self.llm.invoke(synthesis_prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse JSON response
            try:
                json_match = re.search(r'\{[\s\S]*\}', response_text)
                if json_match:
                    json_str = json_match.group(0)
                    synthesis_dict = json.loads(json_str)
                    
                    logger.info(f"[CHUNKING:SYNTHESIS:BATCH] ✓ Successfully synthesized {len(synthesis_dict)} metrics in 1 call")
                    return synthesis_dict
            except json.JSONDecodeError as e:
                logger.warning(f"[CHUNKING:SYNTHESIS:BATCH] JSON parsing failed: {e}")
            
            return {}
        
        except Exception as e:
            logger.error(f"[CHUNKING:SYNTHESIS:BATCH] Batch synthesis failed: {e}")
            return {}
    
    def _template_metric_synthesis(self, metric_name: str, metric_data: Dict, best_values: Dict) -> str:
        """
        OPTIMIZED: Fast template-based synthesis (NO LLM).
        
        Used as fallback when LLM not available or for predefined metrics.
        Provides deterministic, fast output without LLM overhead.
        
        Args:
            metric_name: Name of metric
            metric_data: Extracted metric data
            best_values: Selected values dict
            
        Returns:
            Synthesized metric text (no LLM)
        """
        parts = []
        
        # Core metric info
        parts.append(f"METRIC: {metric_name}")
        
        # Add value and period
        if best_values and best_values.get('latest_value'):
            value_str = best_values['latest_value']
            period_str = best_values.get('period', '')
            if period_str:
                parts.append(f"VALUE: {value_str} ({period_str})")
            else:
                parts.append(f"VALUE: {value_str}")
        
        # Add trend if multiple values available
        if best_values and best_values.get('all_values') and len(best_values['all_values']) > 1:
            parts.append(f"TREND: {' → '.join(best_values['all_values'][:3])}")
        
        # Add type and confidence
        if metric_data["occurrences"]:
            avg_confidence = sum(o.confidence for o in metric_data["occurrences"]) / len(metric_data["occurrences"])
            parts.append(f"TYPE: {metric_data['occurrences'][0].metric_type.value}")
            parts.append(f"CONFIDENCE: {avg_confidence:.0%}")
            parts.append(f"OCCURRENCES: {len(metric_data['occurrences'])}")
        
        # Source indicator
        if metric_data["is_from_table"]:
            parts.append("SOURCE: Table")
        else:
            parts.append("SOURCE: Text")
        
        return " | ".join(parts)
    
    def _batch_synthesize_custom_metrics(self, 
                                        custom_metric_names: List[str],
                                        selected_metrics: Dict[str, float],
                                        document_context: str) -> Dict[str, Optional[MetricChunk]]:
        """
        OPTIMIZED: Synthesize all custom metrics in ONE LLM call.
        
        Args:
            custom_metric_names: List of custom metric names to synthesize
            selected_metrics: Dict of all selected metrics with scores
            document_context: Document excerpt for context (should include actual metric values)
            
        Returns:
            Dict mapping metric_name -> MetricChunk (or None if synthesis failed)
        """
        if not self.llm or not custom_metric_names:
            return {}
        
        try:
            logger.info(f"[CHUNKING:SYNTHESIS:BATCH:CUSTOM] Synthesizing {len(custom_metric_names)} custom metrics in ONE call")
            
            # Build batch synthesis prompt with STRICT DATA GROUNDING
            metrics_list = ", ".join(custom_metric_names)
            
            synthesis_prompt = f"""SYNTHESIZE FINANCIAL METRICS - BATCH MODE

CRITICAL INSTRUCTIONS:
- ONLY use values and data EXPLICITLY found in the document below
- If you cannot find exact values, state "DATA_NOT_AVAILABLE"
- Do NOT hallucinate or make up any numbers
- For calculations: show the math (e.g., Revenue 872M - Cost 395M = Gross Profit 477M)
- Always include units (M, %, K, etc.) from the source data

Document Data:
{document_context[:4000]}

Your Task:
Synthesize these {len(custom_metric_names)} derived/custom metrics based ONLY on actual data above:
{metrics_list}

For each metric:
1. Type: "calculation" (if computed), "aggregate" (if summed), "data_not_available" (if missing)
2. Values: ONLY from document or calculated from document values
3. Calculation: exact formula used (e.g., "Total Revenue - Cost of Revenue")
4. Confidence: 0.85+ if using actual data, 0.0 if DATA_NOT_AVAILABLE

RESPOND WITH ONLY JSON (no markdown, no explanations):
{{
  "metric_name": {{
    "type": "calculation|aggregate|data_not_available",
    "values": ["value_with_unit"],
    "calculation": "exact formula or 'DATA_NOT_AVAILABLE'",
    "confidence": 0.85,
    "summary": "based on: [source data]"
  }}
}}

VALIDATION RULE: If you cannot find the required input data in the document, return type:"data_not_available" with confidence:0.0"""
            
            logger.debug(f"[CHUNKING:SYNTHESIS:BATCH:CUSTOM] Sending batch synthesis prompt with data grounding")
            
            # SINGLE LLM CALL for all custom metrics
            response = self.llm.invoke(synthesis_prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse and create chunks
            result_chunks = {}
            try:
                json_match = re.search(r'\{[\s\S]*\}', response_text)
                if json_match:
                    json_str = json_match.group(0)
                    synthesis_dict = json.loads(json_str)
                    
                    logger.info(f"[CHUNKING:SYNTHESIS:BATCH:CUSTOM] ✓ Synthesized {len(synthesis_dict)} metrics")
                    
                    # Convert synthesis results to MetricChunk objects
                    for metric_name, metric_data in synthesis_dict.items():
                        if metric_name in custom_metric_names and isinstance(metric_data, dict):
                            try:
                                # VALIDATION: Skip metrics marked as data_not_available
                                if metric_data.get('type') == 'data_not_available' or metric_data.get('confidence', 0) == 0:
                                    logger.debug(f"[CHUNKING:SYNTHESIS:BATCH:CUSTOM] Skipping '{metric_name}' - data not available in document")
                                    result_chunks[metric_name] = None
                                    continue
                                
                                # Build occurrences from synthesis
                                occurrences = []
                                for value in metric_data.get('values', []):
                                    occurrence = MetricOccurrence(
                                        metric_name=metric_name,
                                        metric_type=MetricType.OTHER,
                                        value=value,
                                        period="Calculated",
                                        confidence=float(metric_data.get('confidence', 0.7))
                                    )
                                    occurrences.append(occurrence)
                                
                                # Create metric chunk
                                synthesis_text = f"""SYNTHESIZED METRIC: {metric_name}

Type: {metric_data.get('type', 'Unknown')}
Values: {', '.join(str(v) for v in metric_data.get('values', ['N/A']))}
Calculation: {metric_data.get('calculation', 'N/A')}

Summary:
{metric_data.get('summary', 'Metric calculated from document data')}

Confidence: {float(metric_data.get('confidence', 0.7)):.0%}"""
                                
                                metric_chunk = MetricChunk(
                                    metric_name=metric_name,
                                    metric_type=MetricType.OTHER,
                                    chunk_id=str(uuid.uuid4()),
                                    text=synthesis_text,
                                    occurrences=occurrences if occurrences else [MetricOccurrence(
                                        metric_name=metric_name,
                                        metric_type=MetricType.OTHER,
                                        value=str(metric_data.get('values', ['Calculated'])[0]),
                                        period="N/A",
                                        confidence=float(metric_data.get('confidence', 0.7))
                                    )],
                                    source_chunk_ids=[],
                                    is_from_table=False,
                                    is_custom_metric=True,
                                    file_id="",  # Will be set by caller
                                    timestamp=datetime.now().isoformat(),
                                    sources=[],
                                    confidence=min(0.9, float(metric_data.get('confidence', 0.7)))

                                )
                                result_chunks[metric_name] = metric_chunk
                            except Exception as e:
                                logger.warning(f"[CHUNKING:SYNTHESIS:BATCH:CUSTOM] Failed to create chunk for {metric_name}: {e}")
                                result_chunks[metric_name] = None
            
            except json.JSONDecodeError as e:
                logger.warning(f"[CHUNKING:SYNTHESIS:BATCH:CUSTOM] JSON parsing failed: {e}")
            
            return result_chunks
        
        except Exception as e:
            logger.error(f"[CHUNKING:SYNTHESIS:BATCH:CUSTOM] Batch synthesis failed: {e}")
            return {metric_name: None for metric_name in custom_metric_names}
    
    def _synthesize_custom_metric_data(self,
                                       metric_name: str,
                                       document_context: str,
                                       score: float) -> Optional[MetricChunk]:
        """
        SYNTHESIS FOR CUSTOM METRICS: Create synthesized metric data when LLM selects 
        a metric that couldn't be directly extracted (e.g., aggregates, trends, synthetics).
        
        This is CRITICAL for materiality-focused metric selection:
        - LLM selects "Total Production Cost" (aggregate) even if not explicitly stated
        - We synthesize its data from component costs
        - LLM selects "Revenue Growth Rate" (trend) from multiple periods
        - We synthesize the calculation
        
        Args:
            metric_name: Name of custom/synthesized metric
            document_context: Document excerpt with relevant data
            score: Materiality score from LLM (1-10)
            
        Returns:
            MetricChunk with synthesized data, or None if synthesis fails
        """
        if not self.llm:
            logger.warning(f"[CHUNKING:SYNTHESIS:CUSTOM] No LLM available to synthesize '{metric_name}'")
            return None
        
        try:
            logger.info(f"[CHUNKING:SYNTHESIS:CUSTOM] Creating synthesized data for custom metric: {metric_name}")
            
            # Use LLM to synthesize the metric from document context
            synthesis_prompt = f"""SYNTHESIZE FINANCIAL METRIC DATA

Metric Name: {metric_name}

Document Context:
{document_context[:3000]}

TASK: Create synthesized metric data for "{metric_name}" based on the document.

This metric was selected by a financial analyst as important for summarization, 
but doesn't appear explicitly in the document. You must:

1. Identify if this is:
   - An AGGREGATE (e.g., sum of components)
   - A CALCULATION (e.g., ratio, percentage change)
   - A TREND (e.g., YoY comparison)
   - A DERIVED METRIC (e.g., calculated from other data)

2. Extract or calculate the values using document data
3. Provide the synthesized metric summary

RESPOND WITH ONLY JSON (no explanations):
{{
  "metric_name": "{metric_name}",
  "type": "aggregate|calculation|trend|derived",
  "values": ["value1", "value2"],  // synthesized values with periods
  "calculation": "How this was calculated/derived",
  "confidence": 0.7,  // 0.6-0.9 range for synthesized data
  "summary": "Key facts about this synthesized metric"
}}"""
            
            response = self.llm.invoke(synthesis_prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse JSON response
            try:
                json_match = re.search(r'\{[\s\S]*\}', response_text)
                if json_match:
                    json_str = json_match.group(0)
                    metric_data = json.loads(json_str)
                    
                    # Build synthesized occurrences
                    occurrences = []
                    if metric_data.get('values'):
                        for value in metric_data['values']:
                            # Parse "value (period)" format
                            match = re.match(r'(.+?)\s*\((.+?)\)', value)
                            if match:
                                val, period = match.groups()
                            else:
                                val, period = value, "N/A"
                            
                            occurrence = MetricOccurrence(
                                metric_name=metric_name,
                                metric_type=MetricType.OTHER,
                                value=val.strip(),
                                period=period.strip(),
                                confidence=float(metric_data.get('confidence', 0.7))
                            )
                            occurrences.append(occurrence)
                    
                    # Build comprehensive text
                    synthesis_text = f"""SYNTHESIZED METRIC: {metric_name}

Type: {metric_data.get('type', 'Unknown')}
Values: {', '.join(metric_data.get('values', ['N/A']))}
Calculation: {metric_data.get('calculation', 'N/A')}

Summary:
{metric_data.get('summary', 'Metric synthesized from document data')}

Note: This is a synthesized metric created for financial analysis by aggregating or 
deriving from document data. Confidence: {metric_data.get('confidence', 0.7):.0%}"""
                    
                    # Create metric chunk
                    metric_chunk = MetricChunk(
                        metric_name=metric_name,
                        metric_type=MetricType.OTHER,
                        chunk_id=str(uuid.uuid4()),
                        text=synthesis_text,
                        occurrences=occurrences if occurrences else [MetricOccurrence(
                            metric_name=metric_name,
                            metric_type=MetricType.OTHER,
                            value=metric_data.get('values', ['N/A'])[0] if metric_data.get('values') else 'Synthesized',
                            period="N/A",
                            confidence=float(metric_data.get('confidence', 0.7))
                        )],
                        source_chunk_ids=[],
                        is_from_table=metric_data.get('type') == 'aggregate',
                        is_custom_metric=True,
                        file_id="",  # Will be set by caller
                        timestamp=datetime.now().isoformat(),
                        sources=[],
                        confidence=min(0.9, float(metric_data.get('confidence', 0.7)))  # Cap at 0.9 for synthesized
                    )
                    
                    logger.info(f"[CHUNKING:SYNTHESIS:CUSTOM] ✓ Successfully synthesized '{metric_name}' "
                              f"(type={metric_data.get('type')}, confidence={metric_data.get('confidence'):.0%})")
                    return metric_chunk
                else:
                    logger.warning(f"[CHUNKING:SYNTHESIS:CUSTOM] No JSON in synthesis response")
                    return None
                    
            except json.JSONDecodeError as e:
                logger.warning(f"[CHUNKING:SYNTHESIS:CUSTOM] JSON parsing failed: {e}")
                return None
        
        except Exception as e:
            logger.warning(f"[CHUNKING:SYNTHESIS:CUSTOM] Failed to synthesize '{metric_name}': {e}")
            return None
    
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
    
    def _log_pdf_results_to_file(self, 
                                 text: str, 
                                 file_id: str,
                                 structural_chunks: List['StructuralChunk'],
                                 metric_chunks: List['MetricChunk'],
                                 struct_payloads: List[Dict],
                                 metric_payloads: List[Dict]):
        """Log PDF extraction results and metric chunk creation to pdf_result_log.txt"""
        try:
            log_file = Path(__file__).parent.parent.parent / "pdf_result_log.txt"
            
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*100}\n")
                f.write(f"PROCESSING RESULT - {datetime.now().isoformat()}\n")
                f.write(f"{'='*100}\n\n")
                
                # PDF Extraction Summary
                f.write(f"PDF EXTRACTION SUMMARY:\n")
                f.write(f"  File ID: {file_id}\n")
                f.write(f"  Text Length: {len(text)} characters\n")
                f.write(f"  Word Count: {len(text.split())} words\n")
                f.write(f"  Character Density: {len(text.split()) / max(len(text), 1) * 100:.2f}%\n\n")
                
                # Structural Chunks Summary
                f.write(f"STRUCTURAL CHUNKS CREATED:\n")
                f.write(f"  Total Chunks: {len(structural_chunks)}\n")
                if structural_chunks:
                    f.write(f"  Avg Chunk Size: {sum(len(c.text) for c in structural_chunks) / len(structural_chunks):.0f} chars\n")
                    f.write(f"  Min/Max Chunk: {min(len(c.text) for c in structural_chunks)} / {max(len(c.text) for c in structural_chunks)} chars\n")
                    f.write(f"  Sample Chunks:\n")
                    for i, chunk in enumerate(structural_chunks[:3]):
                        f.write(f"    [{i+1}] Index {chunk.chunk_index}: {len(chunk.text)} chars, "
                               f"Metrics: {len(chunk.metrics_found)}\n")
                        if chunk.metrics_found:
                            for metric in chunk.metrics_found[:3]:
                                f.write(f"         - {metric.metric_name}: {metric.value} ({metric.period})\n")
                f.write(f"\n")
                
                # Metric Chunks Summary
                f.write(f"METRIC-CENTRIC CHUNKS:\n")
                f.write(f"  Total Metric Chunks: {len(metric_chunks)}\n")
                f.write(f"  Custom/Synthesized: {sum(1 for m in metric_chunks if m.is_custom_metric)}\n")
                if metric_chunks:
                    f.write(f"  Metrics Created:\n")
                    for i, chunk in enumerate(metric_chunks):
                        f.write(f"    [{i+1}] {chunk.metric_name}\n")
                        f.write(f"        Type: {chunk.metric_type.value}\n")
                        f.write(f"        Source Chunks: {len(chunk.source_chunk_ids)}\n")
                        f.write(f"        Occurrences: {len(chunk.occurrences)}\n")
                        f.write(f"        Is Custom: {chunk.is_custom_metric}\n")
                        f.write(f"        Confidence: {chunk.confidence:.2f}\n")
                        f.write(f"        Text Preview: {chunk.text[:150]}...\n")
                else:
                    f.write(f"  ⚠ NO METRIC CHUNKS CREATED - See diagnostic below\n")
                f.write(f"\n")
                
                # Storage Payloads
                f.write(f"STORAGE PAYLOADS:\n")
                f.write(f"  Structural Payloads: {len(struct_payloads)}\n")
                f.write(f"  Metric Payloads: {len(metric_payloads)}\n\n")
                
                # Diagnostic Info
                f.write(f"DIAGNOSTIC INFORMATION:\n")
                if len(metric_chunks) == 0:
                    f.write(f"  ⚠ NO METRIC CHUNKS CREATED\n")
                    f.write(f"  Possible causes:\n")
                    f.write(f"    1. LLM failed to return selected metrics (check selected_metrics dict)\n")
                    f.write(f"    2. aggregate_metric_chunks() returned empty list\n")
                    f.write(f"    3. validate_metric_chunks() filtered all chunks\n")
                    f.write(f"    4. Structural chunks have no metrics found\n")
                    f.write(f"    5. LLM model not responding correctly\n")
                else:
                    f.write(f"  ✓ {len(metric_chunks)} metric chunks created successfully\n")
                    f.write(f"  ✓ {sum(m.is_custom_metric for m in metric_chunks)} custom/synthesized metrics\n")
                
                f.write(f"\n{'='*100}\n\n")
                
        except Exception as e:
            logger.error(f"Failed to log PDF results: {e}")

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
        
        # NEW: Log PDF results to file for debugging
        self._log_pdf_results_to_file(text, file_id, structural_chunks, metric_chunks, 
                                     struct_payloads, metric_payloads)
        
        logger.info(f"[CHUNKING:PROCESS] ✓ Full pipeline complete: "
                   f"{len(struct_payloads)} structural, "
                   f"{len(metric_payloads)} metric chunks")
        return struct_payloads, metric_payloads

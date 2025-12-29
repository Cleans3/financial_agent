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
        
        # Check if this looks like a table (markdown table or structured data)
        is_table_content = "|" in text and ("bảng" in text_lower or "sheet" in text_lower or "table" in text_lower)
        
        # Extract metrics from keywords (for prose text)
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
                    occurrences.append(occurrence)
        
        # If this is table content, also extract metrics from column headers
        if is_table_content:
            table_metrics = self._extract_metrics_from_table(text)
            occurrences.extend(table_metrics)
        
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
        Useful for Excel/CSV files converted to markdown.
        
        Args:
            text: Markdown table text
            
        Returns:
            List of MetricOccurrence objects from table
        """
        occurrences = []
        text_lower = text.lower()
        
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
                    value = self._extract_value_from_table_context(text, keyword)
                    period = self._extract_period_from_text(text)
                    
                    occurrence = MetricOccurrence(
                        metric_name=metric_name,
                        metric_type=metric_type,
                        value=value,
                        period=period,
                        confidence=0.85 if value else 0.7  # Slightly lower confidence for table metrics
                    )
                    occurrences.append(occurrence)
                    break  # Only count once per metric type
        
        return occurrences
    
    def _extract_value_from_table_context(self, text: str, keyword: str) -> Optional[str]:
        """Extract numeric value from table context around keyword"""
        # Find the keyword position
        lower_text = text.lower()
        pos = lower_text.find(keyword)
        
        if pos >= 0:
            # Look for numbers in the surrounding context (within 100 chars)
            context_start = max(0, pos - 50)
            context_end = min(len(text), pos + 100)
            context = text[context_start:context_end]
            
            # Extract numeric values
            pattern = r'\$?[\d,.]+(M|B|K|%)?'
            matches = re.findall(pattern, context)
            
            if matches:
                # Return the last (most likely the value) match
                return matches[-1] if matches else None
        
        return None
    
    def _extract_value_after_keyword(self, text: str, keyword: str) -> Optional[str]:
        """Extract numeric value following a keyword"""
        pattern = f"{keyword}[:\\s]+([\\$]?[\\d,.]+%?)"
        matches = re.findall(pattern, text, re.IGNORECASE)
        return matches[0] if matches else None
    
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
    
    def aggregate_metric_chunks(self,
                               structural_chunks: List[StructuralChunk],
                               file_id: str) -> List[MetricChunk]:
        """
        Step 2: Aggregate metrics from structural chunks into metric-centric chunks
        
        Collects all unique metrics across document and synthesizes comprehensive 
        information. For table chunks, applies table-specific metric extraction logic.
        
        Args:
            structural_chunks: List of structural chunks
            file_id: Source file ID
            
        Returns:
            List of metric-centric chunks with source tracking
        """
        logger.info(f"[CHUNKING:METRIC] Starting metric aggregation for file {file_id}")
        
        # Collect all unique metrics across structural chunks
        metrics_map: Dict[str, Dict] = {}
        
        for struct_chunk in structural_chunks:
            # For table chunks, also extract table-specific metrics
            metrics_to_process = list(struct_chunk.metrics_found)
            
            if self._is_table_chunk(struct_chunk.text):
                # Add table-specific metrics (from column headers)
                table_metrics = self._extract_table_column_metrics(struct_chunk.text)
                metrics_to_process.extend(table_metrics)
            
            for occurrence in metrics_to_process:
                metric_key = occurrence.metric_name.lower()
                
                if metric_key not in metrics_map:
                    metrics_map[metric_key] = {
                        "type": occurrence.metric_type,
                        "occurrences": [],
                        "source_chunks": [],
                        "texts": [],
                        "is_from_table": False
                    }
                
                # Mark if metric came from table
                if self._is_table_chunk(struct_chunk.text):
                    metrics_map[metric_key]["is_from_table"] = True
                
                # Store occurrence with chunk reference
                occurrence.structural_chunk_id = struct_chunk.point_id
                metrics_map[metric_key]["occurrences"].append(occurrence)
                metrics_map[metric_key]["source_chunks"].append(struct_chunk.point_id)
                metrics_map[metric_key]["texts"].append(struct_chunk.text)
        
        # Create metric chunks with synthesis
        metric_chunks = []
        for metric_name, metric_data in metrics_map.items():
            # Use table-specific synthesis if metric is from table, otherwise use regular synthesis
            if metric_data["is_from_table"]:
                aggregated_text = self._synthesize_table_metric_text(
                    metric_name,
                    metric_data["texts"][0],  # Use first table text
                    metric_data["occurrences"]
                )
            else:
                aggregated_text = self._synthesize_metric_text(
                    metric_name,
                    metric_data["texts"],
                    metric_data["occurrences"]
                )
            
            source_chunks = [cid for cid in metric_data["source_chunks"] if cid is not None]
            metric_chunk = MetricChunk(
                metric_name=metric_name,
                metric_type=metric_data["type"],
                chunk_id=str(uuid.uuid4()),
                text=aggregated_text,
                occurrences=metric_data["occurrences"],
                source_chunk_ids=source_chunks,
                is_from_table=metric_data["is_from_table"],
                file_id=file_id,
                timestamp=datetime.now().isoformat(),
                sources=source_chunks
            )
            metric_chunks.append(metric_chunk)
        
        # Batch score all metrics at once (single LLM call for all metrics)
        if metric_chunks and self.llm:
            logger.debug(f"[CHUNKING:METRIC] Starting batch relevance scoring for {len(metric_chunks)} metrics")
            document_context = " ".join([t[:200] for t in [tc["texts"][0] for tc in metrics_map.values() if tc["texts"]]])
            relevance_scores = self.score_all_metrics_relevance(metric_chunks, document_context)
            
            # Apply relevance scores to occurrences
            for metric_chunk in metric_chunks:
                score = relevance_scores.get(metric_chunk.metric_name, 1.0)
                for occurrence in metric_chunk.occurrences:
                    occurrence.relevance = score
                logger.debug(f"[CHUNKING:METRIC] Applied relevance={score:.2f} to '{metric_chunk.metric_name}'")
        
        logger.info(f"[CHUNKING:METRIC] ✓ Created {len(metric_chunks)} metric-centric chunks "
                   f"from {len(structural_chunks)} structural chunks")
        return metric_chunks
    
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
                                     occurrences: List[MetricOccurrence]) -> str:
        """
        Synthesize metric text from table data (structured, not narrative).
        
        Unlike narrative synthesis, table synthesis:
        - Preserves table structure/format
        - Doesn't use LLM (tables are already well-structured)
        - Extracts all rows mentioning the metric
        - Provides summary statistics
        
        Args:
            metric_name: Name of metric
            table_text: Markdown table text
            occurrences: Metric occurrences in table
            
        Returns:
            Structured text summarizing metric data from table
        """
        synthesis = f"TABLE METRIC: {metric_name.upper()}\n"
        synthesis += f"Source: Markdown table\n\n"
        
        # Extract relevant rows containing the metric
        lines = table_text.split('\n')
        relevant_rows = [line for line in lines if metric_name.lower() in line.lower()]
        
        if relevant_rows:
            synthesis += "RELEVANT TABLE DATA:\n"
            for row in relevant_rows[:10]:  # Limit to first 10 rows
                synthesis += f"{row}\n"
        
        # Add summary
        synthesis += f"\nFIELDS FOUND: {len(occurrences)}\n"
        synthesis += f"CONFIDENCE: {(sum(o.confidence for o in occurrences) / len(occurrences) if occurrences else 0):.1%}\n"
        
        return synthesis
    
    def _synthesize_metric_text(self,
                               metric_name: str,
                               source_texts: List[str],
                               occurrences: List[MetricOccurrence]) -> str:
        """
        Synthesize comprehensive metric text from multiple sources
        
        Process:
        1. If LLM available: Use LLM to create comprehensive narrative
        2. If no LLM: Use rule-based synthesis combining excerpts
        
        Args:
            metric_name: Name of metric
            source_texts: Texts mentioning this metric
            occurrences: Metric occurrences
            
        Returns:
            Synthesized text combining relevant excerpts
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
                
                # Prepare context for LLM
                context = "\n".join(unique_sentences)
                
                # Create synthesis prompt
                synthesis_prompt = f"""You are a financial analysis expert. Synthesize the following information about the metric "{metric_name}" into a comprehensive, cohesive summary.

EXTRACTED MENTIONS:
{context}

Create a single comprehensive paragraph that:
1. Clearly defines what {metric_name} means in this context
2. States the key values and periods mentioned
3. Explains the drivers and context
4. Highlights any notable changes or trends
5. Connects information from different parts of the document

Keep it concise (100-150 words) and professional."""
                
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
        
        synthesis = f"METRIC: {metric_name.upper()}\n\n"
        
        # Add values and periods found
        periods = set()
        values = set()
        for occ in occurrences:
            if occ.period:
                periods.add(occ.period)
            if occ.value:
                values.add(occ.value)
        
        if periods:
            synthesis += f"PERIODS: {', '.join(sorted(periods))}\n"
        if values:
            synthesis += f"VALUES: {', '.join(sorted(values))}\n"
        
        synthesis += f"\nCONTEXT:\n" + "\n".join(unique_sentences)
        
        return synthesis
    
    def score_all_metrics_relevance(self,
                                   metric_chunks: List[MetricChunk],
                                   document_context: str) -> Dict[str, float]:
        """
        Score relevance of ALL metrics at once using single LLM call (batch scoring).
        Much more efficient than scoring metrics individually.
        
        Args:
            metric_chunks: All metric chunks to score
            document_context: Full or representative document text for context
            
        Returns:
            Dict mapping metric_name -> relevance_score (0.0-1.0)
        """
        if not self.llm or not metric_chunks:
            logger.debug(f"[CHUNKING:RELEVANCE] No LLM or no metrics, assuming all relevant")
            return {mc.metric_name: 1.0 for mc in metric_chunks}
        
        try:
            logger.info(f"[CHUNKING:RELEVANCE] Batch scoring {len(metric_chunks)} metrics")
            
            # Prepare metric list for prompt
            metric_list = "\n".join([
                f"{i+1}. {mc.metric_name} ({mc.metric_type.value})"
                for i, mc in enumerate(metric_chunks)
            ])
            
            # Create batch scoring prompt
            batch_prompt = f"""You are a financial analysis expert. Score the relevance of each metric to this document context.

DOCUMENT CONTEXT (excerpt):
{document_context[:800]}

METRICS TO SCORE:
{metric_list}

For each metric, respond with a decimal score 0.0-1.0:
- 1.0 (Highly Relevant): Core to document's main narrative
- 0.7 (Relevant): Supporting metric, clearly applicable
- 0.4 (Somewhat Relevant): Mentioned but tangential
- 0.0 (Not Relevant): Unrelated or inapplicable

RESPOND WITH ONLY these lines (one per metric, in order):
0.95
0.85
0.15
...etc

DO NOT include explanations, metric names, or anything else."""
            
            # Call LLM once for all metrics
            response = self.llm.invoke(batch_prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse scores from response
            relevance_scores = {}
            lines = [line.strip() for line in response_text.split('\n') if line.strip()]
            
            scores_found = 0
            for i, metric_chunk in enumerate(metric_chunks):
                if i < len(lines):
                    try:
                        score = float(lines[i])
                        score = max(0.0, min(1.0, score))  # Clamp to [0.0, 1.0]
                        relevance_scores[metric_chunk.metric_name] = score
                        scores_found += 1
                        logger.debug(f"[CHUNKING:RELEVANCE] '{metric_chunk.metric_name}': {score:.2f}")
                    except (ValueError, IndexError):
                        logger.warning(f"[CHUNKING:RELEVANCE] Could not parse score for metric {i}, defaulting to 0.5")
                        relevance_scores[metric_chunk.metric_name] = 0.5
                else:
                    logger.warning(f"[CHUNKING:RELEVANCE] Missing score for metric {i}, defaulting to 0.5")
                    relevance_scores[metric_chunk.metric_name] = 0.5
            
            logger.info(f"[CHUNKING:RELEVANCE] ✓ Batch scored {scores_found}/{len(metric_chunks)} metrics")
            return relevance_scores
        
        except Exception as e:
            logger.warning(f"[CHUNKING:RELEVANCE] Batch scoring failed: {e}, assuming all relevant")
            return {mc.metric_name: 1.0 for mc in metric_chunks}
    
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

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
    

@dataclass
class MetricChunk:
    """Metric-centric chunk combining information from multiple structural chunks"""
    metric_name: str
    metric_type: MetricType
    chunk_id: str  # UUID for this metric chunk
    text: str  # Synthesized comprehensive text about this metric
    occurrences: List[MetricOccurrence]  # All mentions of this metric
    source_chunk_ids: List[int]  # Point IDs of structural chunks it references
    period: Optional[str] = None
    confidence: float = 0.8
    validation_notes: Optional[str] = None
    file_id: Optional[str] = None  # Reference to source file
    chunk_type: ChunkType = ChunkType.METRIC_CENTRIC
    timestamp: Optional[str] = None
    

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
        
        Args:
            text: Full document text
            file_id: Source file identifier
            
        Returns:
            List of structural chunks with metadata
        """
        logger.info(f"[CHUNKING:STRUCTURAL] Starting structural chunking for file {file_id}")
        
        # Split into sentences
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
        
        # Create StructuralChunk objects and extract metrics from each
        structural_chunks = []
        for idx, chunk_text in enumerate(chunks):
            metrics_found = self._extract_metrics_from_text(chunk_text)
            structural_chunk = StructuralChunk(
                point_id=None,  # Will be assigned by Qdrant
                text=chunk_text,
                chunk_index=idx,
                file_id=file_id,
                metrics_found=metrics_found,
                page_ref=None,  # Could be enhanced to extract page numbers
                timestamp=datetime.now().isoformat()
            )
            structural_chunks.append(structural_chunk)
            logger.debug(f"[CHUNKING:STRUCTURAL] Chunk {idx}: "
                        f"{len(chunk_text.split())} words, "
                        f"{len(metrics_found)} metrics found")
        
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
        
        Args:
            structural_chunks: List of structural chunks
            file_id: Source file ID
            
        Returns:
            List of metric-centric chunks
        """
        logger.info(f"[CHUNKING:METRIC] Starting metric aggregation for file {file_id}")
        
        # Collect all unique metrics across all structural chunks
        metrics_map: Dict[str, Dict] = {}  # metric_name -> {type, occurrences, source_chunks}
        
        for struct_chunk in structural_chunks:
            for occurrence in struct_chunk.metrics_found:
                metric_key = occurrence.metric_name.lower()
                
                if metric_key not in metrics_map:
                    metrics_map[metric_key] = {
                        "type": occurrence.metric_type,
                        "occurrences": [],
                        "source_chunks": [],
                        "texts": []
                    }
                
                # Store occurrence with chunk reference
                occurrence.structural_chunk_id = struct_chunk.point_id
                metrics_map[metric_key]["occurrences"].append(occurrence)
                metrics_map[metric_key]["source_chunks"].append(struct_chunk.point_id)
                metrics_map[metric_key]["texts"].append(struct_chunk.text)
        
        # Create metric chunks
        metric_chunks = []
        for metric_name, metric_data in metrics_map.items():
            # Aggregate text from all structural chunks mentioning this metric
            aggregated_text = self._synthesize_metric_text(
                metric_name,
                metric_data["texts"],
                metric_data["occurrences"]
            )
            
            metric_chunk = MetricChunk(
                metric_name=metric_name,
                metric_type=metric_data["type"],
                chunk_id=str(uuid.uuid4()),
                text=aggregated_text,
                occurrences=metric_data["occurrences"],
                source_chunk_ids=[cid for cid in metric_data["source_chunks"] if cid is not None],
                file_id=file_id,
                timestamp=datetime.now().isoformat()
            )
            metric_chunks.append(metric_chunk)
            
            logger.debug(f"[CHUNKING:METRIC] Created metric chunk for '{metric_name}': "
                        f"type={metric_data['type']}, "
                        f"sources={len(metric_data['source_chunks'])}")
        
        logger.info(f"[CHUNKING:METRIC] ✓ Created {len(metric_chunks)} metric-centric chunks "
                   f"from {len(structural_chunks)} structural chunks")
        return metric_chunks
    
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
        Prepare chunks for storage in Qdrant with metadata
        
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
                'timestamp': metric_chunk.timestamp,
                'source_chunk_ids': metric_chunk.source_chunk_ids,  # Links to structural chunks
                'confidence': metric_chunk.confidence,
                'validation_notes': metric_chunk.validation_notes,
                'periods': [occ.period for occ in metric_chunk.occurrences if occ.period]
            }
            metric_payloads.append(payload)
        
        logger.info(f"[CHUNKING:STORAGE] ✓ Prepared payloads for storage")
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

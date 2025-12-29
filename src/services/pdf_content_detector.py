"""
Detect PDF page content type to determine optimal extraction strategy.
Uses pymupdf for analysis - no external CV libraries needed.
"""

import pymupdf
from enum import Enum
from typing import Dict, List
from dataclasses import dataclass


class PageType(Enum):
    """PDF page content classification"""
    TEXT_HEAVY = "text_heavy"        # Prose, narratives (MD&A)
    TABLE_DOMINANT = "table_dominant" # Mostly tables (Balance Sheet, etc)
    MIXED = "mixed"                   # Both text and tables
    SCANNED = "scanned"              # Image-based, not native PDF
    EMPTY = "empty"


@dataclass
class PageMetrics:
    """Metrics for page analysis"""
    word_count: int
    text_block_count: int
    has_extractable_text: bool
    estimated_table_lines: int
    is_scanned: bool


class PDFContentDetector:
    """Detects page content type and recommends extraction strategy"""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
    
    def analyze_page(self, pdf_path: str, page_num: int) -> PageMetrics:
        """Analyze page and return metrics"""
        doc = pymupdf.open(pdf_path)
        page = doc[page_num]
        
        try:
            # Extract text
            text = page.get_text()
            word_count = len(text.split())
            has_text = word_count > 0
            
            # Count text blocks
            blocks = page.get_text("blocks")
            text_blocks = sum(1 for b in blocks if isinstance(b, tuple) and len(b) > 4)
            
            # Detect table structure by counting lines
            # Tables have organized horizontal and vertical structure
            table_lines = self._count_structural_lines(page)
            
            # Check if scanned
            is_scanned = self._is_scanned(page, text, word_count)
            
            if self.debug:
                print(f"Page {page_num}: {word_count} words, {text_blocks} text blocks, "
                      f"{table_lines} structural lines, scanned={is_scanned}")
            
            return PageMetrics(
                word_count=word_count,
                text_block_count=text_blocks,
                has_extractable_text=has_text,
                estimated_table_lines=table_lines,
                is_scanned=is_scanned
            )
        finally:
            doc.close()
    
    def _is_scanned(self, page, text: str, word_count: int) -> bool:
        """Check if page is scanned (image-based) vs native text"""
        # If very little text, likely scanned
        if word_count < 50:
            return True
        
        # Check text object density
        blocks = page.get_text("blocks")
        text_blocks = sum(1 for b in blocks if isinstance(b, tuple) and len(b) > 4)
        total_blocks = len(blocks)
        
        # If mostly non-text blocks, likely scanned
        if total_blocks > 0 and text_blocks / total_blocks < 0.2:
            return True
        
        return False
    
    def _count_structural_lines(self, page) -> int:
        """Estimate table structure by detecting lines (crude but fast)"""
        # Count paths that look like lines
        try:
            drawings = page.get_drawings()
            line_count = 0
            
            for drawing in drawings:
                # drawings are dicts in pymupdf
                if isinstance(drawing, dict):
                    # Count any drawing as a potential table line
                    line_count += 1
            
            return line_count
        except Exception:
            # If drawing detection fails, estimate from page complexity
            # More blocks = more likely to have tables
            return len(page.get_text("blocks"))
    
    def classify_page(self, pdf_path: str, page_num: int) -> PageType:
        """Classify page content type"""
        metrics = self.analyze_page(pdf_path, page_num)
        
        # Scanned first
        if metrics.is_scanned:
            return PageType.SCANNED
        
        # Empty
        if metrics.word_count < 50:
            return PageType.EMPTY
        
        # Table dominant: high line count, low word count
        if metrics.estimated_table_lines > 20 and metrics.word_count < 500:
            return PageType.TABLE_DOMINANT
        
        # Mixed: both lines and substantial text
        if metrics.estimated_table_lines > 10 and metrics.word_count > 200:
            return PageType.MIXED
        
        # Text heavy: lots of words, few lines
        return PageType.TEXT_HEAVY
    
    def get_extraction_strategy(self, page_type: PageType) -> Dict:
        """Return recommended extraction strategy"""
        strategies = {
            PageType.TEXT_HEAVY: {
                "primary": "pymupdf",
                "use_tables": False,
                "use_ocr": False,
                "reason": "Text-only content, use native text extraction"
            },
            PageType.TABLE_DOMINANT: {
                "primary": "camelot",
                "use_tables": True,
                "use_ocr": False,
                "reason": "Table-focused, use camelot for structure"
            },
            PageType.MIXED: {
                "primary": "hybrid",
                "use_tables": True,
                "use_ocr": False,
                "reason": "Both text and tables, use both tools"
            },
            PageType.SCANNED: {
                "primary": "ocr",
                "use_tables": False,
                "use_ocr": True,
                "reason": "Image-based, needs OCR"
            },
            PageType.EMPTY: {
                "primary": "skip",
                "use_tables": False,
                "use_ocr": False,
                "reason": "Page is empty"
            }
        }
        return strategies.get(page_type, strategies[PageType.TEXT_HEAVY])
    
    def analyze_document(self, pdf_path: str) -> Dict[int, str]:
        """Classify all pages in document"""
        doc = pymupdf.open(pdf_path)
        classification = {}
        
        try:
            for page_num in range(len(doc)):
                page_type = self.classify_page(pdf_path, page_num)
                classification[page_num] = page_type.value
                
                if self.debug:
                    print(f"Page {page_num + 1}: {page_type.value}")
        finally:
            doc.close()
        
        return classification

"""
PDF Processing Tools v2 - Hybrid extraction using pymupdf + camelot
Replaces pdfplumber with better content detection and table extraction.

Strategy:
1. Detect page content type (text, table, mixed, scanned, empty)
2. Extract text with pymupdf (native, accurate, fast)
3. Extract tables with camelot (structured, handles complex layouts)
4. Repair table structures (fix multi-row cells, validate columns)
5. Create chunks with rich metadata (not following 500 token rule for tables)
"""

import os
import re
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import asdict
from pydantic import BaseModel

import pymupdf
from pdf2image import convert_from_path
import pytesseract
from PIL import Image

from ..services.pdf_content_detector import PDFContentDetector, PageType
from ..services.table_extractor import TableExtractor, TableAnalyzer, TableRepairer
from ..services.table_chunking_service import TableChunkingService

logger = logging.getLogger(__name__)


class PDFExtractionResult(BaseModel):
    """Result from PDF extraction"""
    success: bool
    file_name: str
    total_pages: int
    extracted_text: str  # All narrative text
    tables: List[Dict]  # Table data with metadata
    page_classifications: Dict[int, str]  # Page type by page number
    has_text: bool
    processing_method: str  # "native", "ocr", "hybrid"
    message: str


def extract_text_from_pdf(pdf_path: str) -> PDFExtractionResult:
    """
    Extract text and tables from PDF using content-aware strategy.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        PDFExtractionResult with text, tables, and metadata
    """
    try:
        logger.info("="*80)
        logger.info(f"[PDF_TOOLS_V2] Starting PDF extraction v2")
        logger.info(f"[PDF_TOOLS_V2] File: {pdf_path}")
        logger.info("="*80)
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"File not found: {pdf_path}")
        
        file_name = Path(pdf_path).name
        file_size_mb = os.path.getsize(pdf_path) / (1024*1024)
        logger.info(f"[PDF_TOOLS_V2] File size: {file_size_mb:.2f} MB")
        
        # Step 1: Detect content type for each page
        logger.info(f"[PDF_TOOLS_V2] [Step 1/4] Classifying page content types...")
        detector = PDFContentDetector(debug=False)
        page_classifications = detector.analyze_document(pdf_path)
        
        # Log page classification summary
        classification_summary = {}
        for page_num, page_type in page_classifications.items():
            classification_summary[page_type] = classification_summary.get(page_type, 0) + 1
        logger.info(f"[PDF_TOOLS_V2] Page classification summary: {classification_summary}")
        for page_num, page_type in list(page_classifications.items())[:10]:
            logger.debug(f"[PDF_TOOLS_V2]   Page {page_num}: {page_type}")
        
        # Step 2: Extract text with pymupdf (skip table-heavy pages)
        logger.info(f"[PDF_TOOLS_V2] [Step 2/4] Extracting text with pymupdf (skipping table-heavy pages)...")
        extracted_text, page_count = _extract_text_pymupdf(pdf_path, page_classifications)
        
        # Clean remaining narrative text (in case mixed pages)
        extracted_text = _clean_narrative_text(extracted_text)
        
        text_length = len(extracted_text)
        word_count = len(extracted_text.split())
        has_text = word_count > 50
        logger.info(f"[PDF_TOOLS_V2] Text extraction complete:")
        logger.info(f"[PDF_TOOLS_V2]   - Total pages: {page_count}")
        logger.info(f"[PDF_TOOLS_V2]   - Text length: {text_length} chars")
        logger.info(f"[PDF_TOOLS_V2]   - Word count: {word_count}")
        logger.info(f"[PDF_TOOLS_V2]   - Has substantial text: {has_text}")
        
        # Step 3: Extract and repair tables
        logger.info(f"[PDF_TOOLS_V2] [Step 3/4] Extracting and repairing tables...")
        tables = _extract_and_repair_tables(pdf_path, page_classifications)
        logger.info(f"[PDF_TOOLS_V2] Table extraction complete:")
        logger.info(f"[PDF_TOOLS_V2]   - Tables found: {len(tables)}")
        for idx, table in enumerate(tables):
            logger.info(f"[PDF_TOOLS_V2]   - Table {idx+1}: {table.get('rows', 0)} rows x {table.get('cols', 0)} cols (Page {table.get('page', 1)})")
        
        # Step 4: Determine processing method
        logger.info(f"[PDF_TOOLS_V2] [Step 4/4] Determining processing method...")
        scanned_pages = [p for p, pt in page_classifications.items() if pt == PageType.SCANNED.value]
        if scanned_pages:
            processing_method = "hybrid_with_ocr"
            logger.info(f"[PDF_TOOLS_V2] Processing method: {processing_method} (scanned pages detected: {scanned_pages[:5]})")
        elif tables:
            processing_method = "hybrid"
            logger.info(f"[PDF_TOOLS_V2] Processing method: {processing_method} (tables detected)")
        else:
            processing_method = "native"
            logger.info(f"[PDF_TOOLS_V2] Processing method: {processing_method} (pure text PDF)")
        
        logger.info("="*80)
        logger.info(f"[PDF_TOOLS_V2] Extraction SUCCESSFUL")
        logger.info(f"[PDF_TOOLS_V2] Summary: {page_count} pages, {len(tables)} tables, method={processing_method}")
        logger.info("="*80)
        
        return PDFExtractionResult(
            success=True,
            file_name=file_name,
            total_pages=page_count,
            extracted_text=extracted_text,
            tables=tables,
            page_classifications=page_classifications,
            has_text=has_text,
            processing_method=processing_method,
            message="Successfully extracted using hybrid pymupdf + camelot"
        )
        
    except Exception as e:
        logger.error("="*80)
        logger.error(f"[PDF_TOOLS_V2] PDF extraction FAILED")
        logger.error(f"[PDF_TOOLS_V2] Error: {e}", exc_info=True)
        logger.error("="*80)
        return PDFExtractionResult(
            success=False,
            file_name=Path(pdf_path).name if os.path.exists(pdf_path) else "unknown",
            total_pages=0,
            extracted_text="",
            tables=[],
            page_classifications={},
            has_text=False,
            processing_method="failed",
            message=f"Extraction failed: {str(e)}"
        )


def _clean_narrative_text(text: str) -> str:
    """
    Remove table-like content from narrative text extracted by PyMuPDF.
    This prevents redundancy between extracted tables and narrative text.
    
    Removes lines that look like:
    - Pure numeric rows (financial data tables)
    - Header rows with years (e.g., "2023  2022  2021")
    - Alignment rows (dashes and pipes)
    - Rows with excessive whitespace/numbers
    - Lines with mixed pipes and numbers (corrupted table data)
    """
    logger.debug("[PDF_TOOLS_V2] Cleaning narrative text to remove table-like content...")
    
    lines = text.split('\n')
    cleaned_lines = []
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Skip empty lines (preserve structure)
        if not stripped:
            cleaned_lines.append(line)
            continue
        
        # Check if line looks like table content
        is_table_like = False
        
        # Pattern 1: Year header line (e.g., "2023  2022  2021" or "2023 2022 2021")
        if re.match(r'^\s*\d{4}[\s\d]{6,}$', stripped):
            is_table_like = True
            logger.debug(f"[PDF_TOOLS_V2] Removing year header: {stripped[:50]}")
        
        # Pattern 2: Pure numeric data row with alignment (e.g., "123    456    789")
        elif re.match(r'^[\s\d\.,\-\(\)]{10,}$', stripped) and len(stripped.split()) >= 2:
            # Must have at least 2 numeric groups
            numeric_groups = re.findall(r'\d+(?:\.\d+)?', stripped)
            if len(numeric_groups) >= 2:
                is_table_like = True
                logger.debug(f"[PDF_TOOLS_V2] Removing numeric row: {stripped[:50]}")
        
        # Pattern 3: Separator line (dashes and pipes)
        elif re.match(r'^[\|\-\s]+$', stripped):
            is_table_like = True
            logger.debug(f"[PDF_TOOLS_V2] Removing separator line")
        
        # Pattern 4: Row with many trailing numbers/currency amounts
        elif re.search(r'[\$€£₹]?\d+(?:\.\d+)?(?:\s+[\$€£₹]?\d+(?:\.\d+)?){2,}', stripped):
            # Check if this looks like a data row (not prose with numbers)
            words = stripped.split()
            numeric_count = sum(1 for w in words if re.match(r'^[\$€£₹]?\d+', w))
            if numeric_count / len(words) > 0.5 and len(words) >= 3:
                is_table_like = True
                logger.debug(f"[PDF_TOOLS_V2] Removing financial data row: {stripped[:50]}")
        
        # Pattern 5: Mixed corrupted table data (pipes with no proper structure)
        # Lines like "| Revenue | 2024 | 2023 ||---| or "...||---|---|---|---||"
        elif '||' in stripped and ('---' in stripped or re.search(r'\|\s*\d', stripped)):
            is_table_like = True
            logger.debug(f"[PDF_TOOLS_V2] Removing corrupted table line: {stripped[:50]}")
        
        # Pattern 6: Lines with excessive pipe characters (broken table formatting)
        elif stripped.count('|') > 5 and (stripped.count('---') > 0 or re.search(r'\|\s*\d.*\|', stripped)):
            is_table_like = True
            logger.debug(f"[PDF_TOOLS_V2] Removing excessive-pipe line: {stripped[:50]}")
        
        if not is_table_like:
            cleaned_lines.append(line)
    
    result = '\n'.join(cleaned_lines)
    removed_chars = len(text) - len(result)
    logger.info(f"[PDF_TOOLS_V2] Narrative cleaned: removed {removed_chars} characters of table-like content")
    return result


def _extract_text_pymupdf(pdf_path: str, page_classifications: Dict[int, str] = None) -> Tuple[str, int]:
    """
    Extract all narrative text from PDF using pymupdf.
    Skips table-heavy pages to prevent redundancy with Camelot table extraction.
    
    Args:
        pdf_path: Path to PDF
        page_classifications: Optional dict of page_number -> page_type to skip TABLE pages
    
    Returns:
        (extracted_text, page_count) tuple
    """
    logger.debug(f"[PDF_TOOLS_V2] Opening PDF with pymupdf: {pdf_path}")
    doc = pymupdf.open(pdf_path)
    all_text = []
    pages_with_text = 0
    pages_skipped = 0
    
    try:
        for page_num, page in enumerate(doc):
            # Skip pages classified as primarily tables to avoid redundancy
            if page_classifications:
                page_type = page_classifications.get(page_num + 1, PageType.TEXT_HEAVY.value)
                if page_type == PageType.TABLE_DOMINANT.value:
                    logger.debug(f"[PDF_TOOLS_V2]   Page {page_num + 1}: Skipped (TABLE page - will use Camelot extraction)")
                    pages_skipped += 1
                    continue
            
            # Extract text
            text = page.get_text()
            text_stripped = text.strip()
            
            if text_stripped:
                pages_with_text += 1
                text_length = len(text)
                logger.debug(f"[PDF_TOOLS_V2]   Page {page_num + 1}: {text_length} chars extracted")
                # Add page marker for reference
                all_text.append(f"--- PAGE {page_num + 1} ---\n{text}")
            else:
                logger.debug(f"[PDF_TOOLS_V2]   Page {page_num + 1}: Empty page")
        
        page_count = len(doc)
        logger.info(f"[PDF_TOOLS_V2] PyMuPDF extraction: {page_count} total pages, {pages_with_text} with text, {pages_skipped} skipped (table-heavy)")
    finally:
        doc.close()
    
    combined_text = "\n\n".join(all_text)
    logger.info(f"[PDF_TOOLS_V2] Total extracted text: {len(combined_text)} characters")
    return combined_text, page_count


def _extract_and_repair_tables(
    pdf_path: str,
    page_classifications: Dict[int, str]
) -> List[Dict]:
    """
    Extract tables using camelot and repair structural issues.
    
    Returns:
        List of table dicts with data and metadata
    """
    logger.info(f"[PDF_TOOLS_V2] Extracting tables from {Path(pdf_path).name}...")
    tables_raw = TableExtractor.extract_tables(pdf_path, use_camelot=True)
    
    if not tables_raw:
        logger.info(f"[PDF_TOOLS_V2] No tables found in document")
        return []
    
    logger.info(f"[PDF_TOOLS_V2] Raw tables extracted: {len(tables_raw)}")
    
    # Repair each table
    repaired_tables = []
    for table_idx, table_info in enumerate(tables_raw):
        table_data = table_info.get('data', [])
        page = table_info.get('page', 1)
        
        logger.debug(f"[PDF_TOOLS_V2] Repairing table {table_idx + 1} on page {page}...")
        
        # Apply repairs
        table_data_before = len(table_data)
        table_data = TableRepairer.repair_table(table_data)
        table_data_after = len(table_data)
        
        # Analyze structure
        table_info_dict = TableAnalyzer.get_table_info(table_data)
        rows = table_info_dict.get('rows', 0)
        cols = table_info_dict.get('cols', 0)
        
        logger.info(f"[PDF_TOOLS_V2]   Table {table_idx + 1}: {rows} rows × {cols} cols (page {page}, repaired {table_data_before}->{table_data_after} rows)")
        
        repaired_tables.append({
            'data': table_data,
            'page': page,
            'rows': rows,
            'cols': cols,
            'info': table_info_dict,
            'source': 'camelot'
        })
    
    logger.info(f"[PDF_TOOLS_V2] Table extraction complete: {len(repaired_tables)} tables extracted and repaired")
    return repaired_tables


def create_table_chunks(
    tables: List[Dict],
    document_metadata: Dict
) -> List[Dict]:
    """
    Create semantic chunks from extracted tables.
    Each table becomes one or more chunks depending on size.
    
    Args:
        tables: List of table dicts from extraction
        document_metadata: Document context (company, year, etc.)
    
    Returns:
        List of chunk dicts ready for embedding
    """
    logger.info(f"[PDF_TOOLS_V2] Creating semantic chunks from {len(tables)} tables...")
    chunker = TableChunkingService(max_rows_per_chunk=15)
    all_chunks = []
    
    for table_idx, table in enumerate(tables):
        table_data = table.get('data', [])
        page_number = table.get('page', 1)
        
        if not table_data:
            logger.debug(f"[PDF_TOOLS_V2]   Table {table_idx + 1}: Empty, skipping")
            continue
        
        # Create chunks from this table
        table_name = document_metadata.get('table_names', {}).get(
            f"page_{page_number}_table_{table_idx}",
            f"Table {table_idx + 1}"
        )
        
        logger.debug(f"[PDF_TOOLS_V2]   Chunking table {table_idx + 1}: {table_name} ({len(table_data)} rows)...")
        
        chunks = chunker.chunk_table(
            table_data=table_data,
            table_name=table_name,
            document_metadata=document_metadata,
            page_number=page_number
        )
        
        # Convert to dicts for storage
        for chunk in chunks:
            all_chunks.append(chunk.to_dict())
        
        logger.info(f"[PDF_TOOLS_V2]   Table {table_idx + 1}: Created {len(chunks)} chunks")
    
    logger.info(f"[PDF_TOOLS_V2] Table chunking complete: {len(all_chunks)} chunks from {len(tables)} tables")
    return all_chunks


# Fallback OCR function for scanned PDFs
def extract_text_from_scanned_pdf(pdf_path: str) -> str:
    """
    Extract text from scanned/image-based PDFs using OCR.
    Fallback when pymupdf text extraction fails.
    """
    try:
        logger.info(f"[PDF_TOOLS_V2] Starting OCR on scanned PDF: {Path(pdf_path).name}")
        
        # Convert PDF pages to images
        logger.debug(f"[PDF_TOOLS_V2] Converting PDF pages to images...")
        images = convert_from_path(pdf_path)
        logger.info(f"[PDF_TOOLS_V2] Converted {len(images)} pages to images")
        
        all_text = []
        successful_pages = 0
        
        for page_num, image in enumerate(images):
            # Preprocess image for better OCR
            logger.debug(f"[PDF_TOOLS_V2] Preprocessing image for page {page_num + 1}...")
            image = _preprocess_image_for_ocr(image)
            
            # OCR with pytesseract
            logger.debug(f"[PDF_TOOLS_V2] Running Tesseract OCR on page {page_num + 1}...")
            text = pytesseract.image_to_string(image, lang='eng+vie')
            
            if text.strip():
                successful_pages += 1
                text_length = len(text)
                logger.info(f"[PDF_TOOLS_V2]   Page {page_num + 1}: {text_length} chars extracted via OCR")
                all_text.append(f"--- PAGE {page_num + 1} (OCR) ---\n{text}")
            else:
                logger.warning(f"[PDF_TOOLS_V2]   Page {page_num + 1}: No text extracted via OCR")
        
        combined_text = "\n\n".join(all_text)
        logger.info(f"[PDF_TOOLS_V2] OCR extraction complete: {len(images)} pages, {successful_pages} with text, {len(combined_text)} chars total")
        
        return combined_text
        
    except Exception as e:
        logger.error(f"[PDF_TOOLS_V2] OCR extraction failed: {e}", exc_info=True)
        return ""


def _preprocess_image_for_ocr(image: Image.Image) -> Image.Image:
    """Preprocess image to improve OCR accuracy"""
    import cv2
    import numpy as np
    
    # Convert PIL to numpy/OpenCV format
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Thresholding
    _, thresh = cv2.threshold(enhanced, 150, 255, cv2.THRESH_BINARY)
    
    # Denoising
    denoised = cv2.bilateralFilter(thresh, 9, 75, 75)
    
    return Image.fromarray(denoised)


# ============================================================================
# Workflow Integration - Wrapper function for LangGraph workflow compatibility
# ============================================================================

class PDFAnalysisResult(BaseModel):
    """Result from PDF analysis including LLM processing"""
    success: bool
    file_name: str
    total_pages: int
    extracted_text: str
    tables_markdown: str
    analysis: str
    processing_method: str
    message: str


def convert_tables_to_markdown(tables: List[Dict[str, Any]]) -> str:
    """Convert extracted tables to markdown format with proper formatting."""
    logger.debug(f"[PDF_TOOLS_V2] Converting {len(tables)} tables to markdown...")
    markdown_output = []
    
    for idx, table in enumerate(tables):
        table_data = table.get('data', [])
        page = table.get('page', 1)
        
        if not table_data or len(table_data) == 0:
            continue
        
        # Add table header with metadata
        markdown_output.append(f"\n### Table {idx + 1} (Page {page})\n")
        
        # Convert to markdown table with proper formatting
        for row_idx, row in enumerate(table_data):
            if not row or all(str(cell).strip() == '' for cell in row):
                # Skip empty rows
                continue
            
            # Create row string
            row_str = "| " + " | ".join(str(cell).strip() for cell in row) + " |"
            markdown_output.append(row_str)
    
    result = "\n".join(markdown_output)
    logger.info(f"[PDF_TOOLS_V2] Converted to markdown: {len(result)} characters")
    return result


def analyze_pdf(
    pdf_path: str,
    question: str = "",
    gemini_api_key: Optional[str] = None
) -> PDFAnalysisResult:
    """
    Complete PDF analysis pipeline using pdf_tools_v2.
    
    Features:
    - Content-aware extraction (detects page types)
    - Hybrid text + table extraction
    - Table structure repair
    - OCR fallback for scanned PDFs
    - Detailed logging for debugging
    
    Args:
        pdf_path: Path to PDF file
        question: Optional question/context for LLM analysis
        gemini_api_key: Optional Gemini API key
        
    Returns:
        PDFAnalysisResult with extracted content and analysis
    """
    try:
        logger.info("="*80)
        logger.info("[PDF_TOOLS_V2] Starting PDF analysis pipeline")
        logger.info("="*80)
        logger.info(f"[PDF_TOOLS_V2] PDF File: {pdf_path}")
        if question:
            logger.info(f"[PDF_TOOLS_V2] Question: {question}")
        
        # Step 1: Extract text and tables
        logger.info("[PDF_TOOLS_V2] [Step 1/3] Extracting content...")
        extraction_result = extract_text_from_pdf(pdf_path)
        
        if not extraction_result.success:
            logger.error(f"[PDF_TOOLS_V2] Extraction failed: {extraction_result.message}")
            return PDFAnalysisResult(
                success=False,
                file_name=extraction_result.file_name,
                total_pages=extraction_result.total_pages,
                extracted_text="",
                tables_markdown="",
                analysis="",
                processing_method="failed",
                message=extraction_result.message
            )
        
        logger.info(f"[PDF_TOOLS_V2] Extraction successful: {extraction_result.total_pages} pages, {len(extraction_result.tables)} tables")
        logger.debug(f"[PDF_TOOLS_V2] Processing method: {extraction_result.processing_method}")
        logger.debug(f"[PDF_TOOLS_V2] Text length: {len(extraction_result.extracted_text)} chars")
        
        # Step 3: Convert tables to markdown
        logger.info("[PDF_TOOLS_V2] [Step 2/3] Converting tables to markdown...")
        tables_markdown = convert_tables_to_markdown(extraction_result.tables)
        logger.info(f"[PDF_TOOLS_V2] Tables converted: {len(extraction_result.tables)} tables, {len(tables_markdown)} chars markdown")
        
        # Step 4: Build interleaved content by page for context preservation
        logger.info("[PDF_TOOLS_V2] [Step 3/3] Building page-aware content with proper interleaving...")
        
        combined_text = extraction_result.extracted_text
        
        # If we have tables, insert them at appropriate page positions with proper line breaks
        if extraction_result.tables:
            # Create a map of tables by page  
            tables_by_page = {}
            for table in extraction_result.tables:
                page = table.get('page', 1)
                if page not in tables_by_page:
                    tables_by_page[page] = []
                tables_by_page[page].append(table)
            
            # Split text by page markers
            page_sections = combined_text.split('--- PAGE ')
            
            if len(page_sections) > 1:
                # Reconstruct with properly interleaved tables
                rebuilt_content = []
                
                for section_idx, section in enumerate(page_sections):
                    if section_idx == 0:
                        # First section before any page marker
                        rebuilt_content.append(section)
                    else:
                        # Extract page number from section start (format: "N ---\n content")
                        parts = section.split(' ---\n', 1)
                        if len(parts) == 2:
                            page_num_str = parts[0].strip()
                            page_content = parts[1]
                            
                            # Try to parse page number
                            try:
                                page_num = int(page_num_str)
                                
                                # Add page marker and content
                                rebuilt_content.append(f"\n--- PAGE {page_num_str} ---\n{page_content}")
                                
                                # Add any tables for this page with proper newlines
                                if page_num in tables_by_page:
                                    for table_idx, table in enumerate(tables_by_page[page_num]):
                                        table_data = table.get('data', [])
                                        if table_data and len(table_data) > 0:
                                            rebuilt_content.append(f"\n### Table {table_idx + 1} (Page {page_num})\n")
                                            
                                            for row_idx, row in enumerate(table_data):
                                                if not row or all(str(cell).strip() == '' for cell in row):
                                                    continue
                                                
                                                # Create row with proper pipe delimiters
                                                row_str = "| " + " | ".join(str(cell).strip() for cell in row) + " |"
                                                rebuilt_content.append(row_str)
                                            
                                            rebuilt_content.append("")  # Blank line after table
                            except ValueError:
                                # Page number parsing failed, just append as-is
                                rebuilt_content.append(f"\n--- PAGE {section}")
                        else:
                            rebuilt_content.append(f"\n--- PAGE {section}")
                
                combined_text = "\n".join(rebuilt_content)
            else:
                # No page markers, append tables at end
                if tables_markdown:
                    combined_text += f"\n\n## All Tables\n\n{tables_markdown}"
        
        logger.info(f"[PDF_TOOLS_V2] Combined content: {len(combined_text)} chars total")
        logger.debug(f"[PDF_TOOLS_V2] RAW RESULT:\n{combined_text[:1000]}...")  # First 1000 chars for debugging
        
        
        logger.info("="*80)
        logger.info("[PDF_TOOLS_V2] PDF analysis SUCCESSFUL")
        logger.info("="*80)
        
        return PDFAnalysisResult(
            success=True,
            file_name=extraction_result.file_name,
            total_pages=extraction_result.total_pages,
            extracted_text=combined_text,
            tables_markdown=tables_markdown if tables_markdown else "",
            analysis="",
            processing_method=extraction_result.processing_method,
            message="Successfully analyzed with pdf_tools_v2 (hybrid pymupdf + camelot)"
        )
    
    except Exception as e:
        logger.error("="*80)
        logger.error("[PDF_TOOLS_V2] PDF analysis FAILED")
        logger.error(f"[PDF_TOOLS_V2] Error: {e}", exc_info=True)
        logger.error("="*80)
        return PDFAnalysisResult(
            success=False,
            file_name=Path(pdf_path).name if os.path.exists(pdf_path) else "unknown",
            total_pages=0,
            extracted_text="",
            tables_markdown="",
            analysis="",
            processing_method="failed",
            message=f"Analysis failed: {str(e)}"
        )

"""
PDF Processing Tools - Xử lý file PDF báo cáo tài chính
Sử dụng pdfplumber để trích xuất text và bảng từ PDF
Với fallback OCR cho PDF scanned
"""

import os
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class PDFExtractionResult(BaseModel):
    """Model cho kết quả trích xuất từ PDF"""
    success: bool
    file_name: str
    total_pages: int
    extracted_text: str
    tables: List[Dict[str, Any]]  # Danh sách bảng trích xuất được
    has_text: bool  # True nếu PDF chứa text, False nếu là scanned
    message: str


class PDFAnalysisResult(BaseModel):
    """Model cho kết quả phân tích PDF"""
    success: bool
    file_name: str
    total_pages: int
    extracted_text: str
    tables_markdown: str
    analysis: str
    processing_method: str  # "native" hoặc "ocr"
    message: str


def extract_text_from_pdf(pdf_path: str) -> PDFExtractionResult:
    """
    Trích xuất text và bảng từ file PDF
    Thử dùng pdfplumber trước (nếu PDF chứa text)
    Nếu thất bại, fallback sang OCR với pytesseract
    
    Args:
        pdf_path: Đường dẫn đến file PDF
        
    Returns:
        PDFExtractionResult chứa text và tables
    """
    try:
        logger.info(f"Starting PDF extraction: {pdf_path}")
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Không tìm thấy file: {pdf_path}")
        
        file_name = Path(pdf_path).name
        extracted_text = ""
        tables = []
        has_text = False
        
        # Bước 1: Cố gắng trích xuất text native từ PDF
        try:
            logger.info("Attempting native PDF text extraction with pdfplumber...")
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                logger.info(f"PDF has {total_pages} pages")
                
                # Trích xuất text từ tất cả các trang
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            extracted_text += f"\n--- TRANG {page_num} ---\n{page_text}\n"
                            has_text = True
                        
                        # Trích xuất bảng
                        page_tables = page.extract_tables()
                        if page_tables:
                            for table_idx, table in enumerate(page_tables):
                                tables.append({
                                    "page": page_num,
                                    "table_index": table_idx,
                                    "data": table
                                })
                                logger.info(f"Found table on page {page_num}")
                    
                    except Exception as e:
                        logger.warning(f"Error extracting from page {page_num}: {e}")
                        continue
        
        except Exception as e:
            logger.warning(f"Native extraction failed: {e}")
            has_text = False
        
        # Bước 2: Nếu PDF không chứa text (scanned), dùng OCR
        if not has_text:
            logger.info("PDF appears to be scanned/image-based. Falling back to OCR...")
            try:
                extracted_text = _extract_text_via_ocr(pdf_path)
                logger.info(f"OCR extraction successful. Extracted {len(extracted_text)} characters")
            except Exception as e:
                logger.error(f"OCR extraction failed: {e}")
                raise
        
        logger.info(f"PDF extraction completed. Text length: {len(extracted_text)} chars, Tables found: {len(tables)}")
        
        return PDFExtractionResult(
            success=True,
            file_name=file_name,
            total_pages=total_pages,
            extracted_text=extracted_text,
            tables=tables,
            has_text=has_text,
            message=f"Trích xuất thành công. Method: {'Native' if has_text else 'OCR'}"
        )
    
    except Exception as e:
        logger.error(f"Error extracting PDF: {e}")
        return PDFExtractionResult(
            success=False,
            file_name=Path(pdf_path).name if os.path.exists(pdf_path) else "unknown",
            total_pages=0,
            extracted_text="",
            tables=[],
            has_text=False,
            message=f"Lỗi trích xuất PDF: {str(e)}"
        )


def _extract_text_via_ocr(pdf_path: str) -> str:
    """
    Fallback OCR extraction cho PDF scanned
    
    Args:
        pdf_path: Đường dẫn đến file PDF
        
    Returns:
        Text trích xuất được từ OCR
    """
    logger.info("Converting PDF pages to images for OCR...")
    
    try:
        # Chuyển PDF sang ảnh
        images = convert_from_path(pdf_path)
        extracted_text = ""
        
        for page_num, image in enumerate(images, 1):
            logger.info(f"Running OCR on page {page_num}/{len(images)}...")
            
            try:
                # OCR tiếng Việt + tiếng Anh
                text = pytesseract.image_to_string(image, lang='vie+eng')
                if text.strip():
                    extracted_text += f"\n--- TRANG {page_num} ---\n{text}\n"
            except Exception as e:
                logger.warning(f"OCR failed for page {page_num}: {e}")
                continue
        
        if not extracted_text.strip():
            raise ValueError("OCR không trích xuất được text nào từ PDF")
        
        return extracted_text
    
    except Exception as e:
        logger.error(f"OCR extraction failed: {e}")
        raise


def convert_tables_to_markdown(tables: List[Dict[str, Any]]) -> str:
    """
    Chuyển đổi danh sách bảng thành Markdown format
    
    Args:
        tables: Danh sách bảng từ pdfplumber
        
    Returns:
        Markdown string của các bảng
    """
    if not tables:
        return ""
    
    markdown = ""
    
    for table_info in tables:
        page = table_info["page"]
        table_idx = table_info["table_index"]
        table_data = table_info["data"]
        
        markdown += f"\n### Bảng {table_idx + 1} (Trang {page})\n\n"
        
        if not table_data:
            markdown += "*Bảng trống*\n\n"
            continue
        
        # Tạo header
        if table_data:
            headers = table_data[0]
            markdown += "| " + " | ".join(str(h) for h in headers) + " |\n"
            markdown += "|" + "|".join(["---"] * len(headers)) + "|\n"
            
            # Tạo rows
            for row in table_data[1:]:
                markdown += "| " + " | ".join(str(cell) if cell is not None else "" for cell in row) + " |\n"
        
        markdown += "\n"
    
    return markdown


def analyze_pdf(
    pdf_path: str,
    question: str = ""
) -> PDFAnalysisResult:
    """
    Phân tích file PDF báo cáo tài chính
    
    Args:
        pdf_path: Đường dẫn đến file PDF
        question: Câu hỏi hoặc context thêm từ người dùng (optional)
        
    Returns:
        PDFAnalysisResult chứa text, tables, và analysis
    """
    try:
        logger.info(f"Analyzing PDF: {pdf_path}")
        
        # Bước 1: Trích xuất text và bảng
        extraction_result = extract_text_from_pdf(pdf_path)
        
        if not extraction_result.success:
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
        
        # Bước 2: Chuyển bảng sang Markdown
        tables_markdown = convert_tables_to_markdown(extraction_result.tables)
        
        # Bước 3: Kết hợp tất cả dữ liệu
        combined_text = extraction_result.extracted_text
        if tables_markdown:
            combined_text += f"\n\n## Các bảng trong PDF\n\n{tables_markdown}"
        
        logger.info("PDF analysis completed")
        
        return PDFAnalysisResult(
            success=True,
            file_name=extraction_result.file_name,
            total_pages=extraction_result.total_pages,
            extracted_text=extraction_result.extracted_text,
            tables_markdown=tables_markdown,
            analysis=combined_text,  # Sẽ được xử lý thêm bởi LLM
            processing_method="native" if extraction_result.has_text else "ocr",
            message=extraction_result.message
        )
    
    except Exception as e:
        logger.error(f"Error analyzing PDF: {e}")
        return PDFAnalysisResult(
            success=False,
            file_name=Path(pdf_path).name if os.path.exists(pdf_path) else "unknown",
            total_pages=0,
            extracted_text="",
            tables_markdown="",
            analysis="",
            processing_method="failed",
            message=f"Lỗi phân tích PDF: {str(e)}"
        )


# Export tools
def get_pdf_tools() -> list:
    """
    Trả về danh sách tools cho LangGraph
    """
    from langchain_core.tools import tool
    
    @tool
    def analyze_pdf_file(pdf_path: str, question: str = "") -> str:
        """
        Phân tích file PDF báo cáo tài chính.
        Tự động trích xuất text native (nếu có) hoặc dùng OCR (nếu scanned).
        Cũng trích xuất các bảng dữ liệu.
        
        Args:
            pdf_path: Đường dẫn đến file PDF báo cáo tài chính
            question: Câu hỏi hoặc context thêm (optional)
            
        Returns:
            JSON string với kết quả phân tích (text, tables, method)
        """
        try:
            result = analyze_pdf(pdf_path, question)
            return json.dumps({
                "success": result.success,
                "file_name": result.file_name,
                "total_pages": result.total_pages,
                "extracted_text": result.extracted_text[:1000] + "..." if len(result.extracted_text) > 1000 else result.extracted_text,
                "tables_markdown": result.tables_markdown,
                "analysis": result.analysis,
                "processing_method": result.processing_method,
                "message": result.message
            }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({
                "success": False,
                "message": f"Lỗi phân tích PDF: {str(e)}"
            }, ensure_ascii=False)
    
    return [analyze_pdf_file]


if __name__ == "__main__":
    # Test
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pdf_tools.py <pdf_path> [question]")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    question = sys.argv[2] if len(sys.argv) > 2 else ""
    
    result = analyze_pdf(pdf_path, question)
    
    print("\n" + "="*80)
    print(f"Success: {result.success}")
    print(f"File: {result.file_name}")
    print(f"Pages: {result.total_pages}")
    print(f"Processing Method: {result.processing_method}")
    print(f"Message: {result.message}")
    print("\n" + "-"*80)
    print("EXTRACTED TEXT (First 500 chars):")
    print(result.extracted_text[:500])
    print("\n" + "-"*80)
    print("TABLES MARKDOWN:")
    print(result.tables_markdown if result.tables_markdown else "(No tables found)")

"""
Financial Report Tools - Xử lý báo cáo tài chính từ hình ảnh
Sử dụng OCR (pytesseract) + Gemini để phân tích báo cáo
"""

import os
import base64
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional
import cv2
import numpy as np
from PIL import Image
import pytesseract
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class FinancialReportResult(BaseModel):
    """Model cho kết quả phân tích báo cáo tài chính"""
    success: bool
    report_type: str  # BCDN, KQKD, Dòng tiền, Chỉ số
    company: str
    period: str
    extracted_text: str
    markdown_table: str
    analysis: str
    message: str


def extract_text_from_image(image_path: str) -> str:
    """
    Trích xuất text từ hình ảnh báo cáo bằng OCR
    
    Args:
        image_path: Đường dẫn đến file hình ảnh
        
    Returns:
        Text đã trích xuất
    """
    try:
        logger.info(f"Starting OCR for image: {image_path}")
        
        # Đọc hình ảnh
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Không tìm thấy file: {image_path}")
        
        # Tiền xử lý hình ảnh để cải thiện OCR
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Tăng độ tương phản
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Thresholding
        _, thresh = cv2.threshold(enhanced, 150, 255, cv2.THRESH_BINARY)
        
        # Denoising
        denoised = cv2.bilateralFilter(thresh, 9, 75, 75)
        
        # OCR với pytesseract
        logger.info("Running pytesseract OCR...")
        text = pytesseract.image_to_string(denoised, lang='vie+eng')
        
        if not text.strip():
            raise ValueError("OCR không trích xuất được text nào")
        
        logger.info(f"OCR successful. Extracted {len(text)} characters")
        return text
        
    except Exception as e:
        logger.error(f"Error during OCR: {e}")
        raise


def analyze_financial_report(
    image_path: str,
    gemini_api_key: Optional[str] = None
) -> FinancialReportResult:
    """
    Phân tích báo cáo tài chính từ hình ảnh
    
    Args:
        image_path: Đường dẫn đến file hình ảnh báo cáo
        gemini_api_key: API key của Gemini (optional, lấy từ env nếu không có)
        
    Returns:
        FinancialReportResult chứa phân tích chi tiết
    """
    try:
        # Lấy API key
        api_key = gemini_api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY không được cấu hình")
        
        logger.info(f"Analyzing financial report: {image_path}")
        
        # Bước 1: OCR từ hình ảnh
        logger.info("Step 1: Extracting text from image...")
        extracted_text = extract_text_from_image(image_path)
        
        # Bước 2: Load prompt chuyên biệt
        prompt_path = Path(__file__).parent.parent / "agent" / "prompts" / "financial_report_prompt.txt"
        if not prompt_path.exists():
            logger.warning(f"Prompt file not found: {prompt_path}, using default prompt")
            system_prompt = "Bạn là chuyên gia phân tích báo cáo tài chính. Phân tích text OCR và tạo bảng Markdown."
        else:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                system_prompt = f.read()
        
        # Bước 3: Gửi đến Gemini để phân tích
        logger.info("Step 2: Sending to Gemini for analysis...")
        
        llm = ChatGoogleGenerativeAI(
            api_key=api_key,
            model="gemini-2.0-flash",
            temperature=0.3
        )
        
        # Tạo prompt phân tích
        analysis_prompt = f"""
{system_prompt}

HÌNH ẢNH BÁOCÁO ĐÃ ĐƯỢC OCR:

{extracted_text}

---

NHIỆM VỤ:
1. Nhận diện loại báo cáo (BCDN, KQKD, Dòng tiền, Chỉ số)
2. Trích xuất dữ liệu chính
3. Tạo bảng Markdown chi tiết
4. Phân tích và đưa ra nhận xét

Vui lòng trả lời theo đúng định dạng hướng dẫn.
"""
        
        message = llm.invoke(analysis_prompt)
        analysis_result = message.content
        
        logger.info("Analysis completed successfully")
        
        # Bước 4: Trích xuất kết quả
        return FinancialReportResult(
            success=True,
            report_type="Báo cáo tài chính",
            company="Được trích xuất từ báo cáo",
            period="Được trích xuất từ báo cáo",
            extracted_text=extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text,
            markdown_table="(Xem phần phân tích dưới đây)",
            analysis=analysis_result,
            message="Phân tích báo cáo tài chính thành công"
        )
        
    except Exception as e:
        logger.error(f"Error analyzing report: {e}")
        return FinancialReportResult(
            success=False,
            report_type="Lỗi",
            company="N/A",
            period="N/A",
            extracted_text="",
            markdown_table="",
            analysis="",
            message=f"Lỗi phân tích báo cáo: {str(e)}"
        )


def analyze_financial_report_batch(
    image_paths: list,
    gemini_api_key: Optional[str] = None
) -> list:
    """
    Phân tích nhiều báo cáo tài chính
    
    Args:
        image_paths: Danh sách đường dẫn file hình ảnh
        gemini_api_key: API key của Gemini
        
    Returns:
        Danh sách FinancialReportResult
    """
    results = []
    for image_path in image_paths:
        try:
            result = analyze_financial_report(image_path, gemini_api_key)
            results.append(result)
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            results.append(FinancialReportResult(
                success=False,
                report_type="Lỗi",
                company="N/A",
                period="N/A",
                extracted_text="",
                markdown_table="",
                analysis="",
                message=f"Lỗi: {str(e)}"
            ))
    
    return results


# Export cho LangGraph tools
def get_financial_report_tools() -> list:
    """
    Trả về danh sách tools cho LangGraph
    """
    
    @tool
    def analyze_financial_report_image(image_path: str) -> str:
        """
        Phân tích báo cáo tài chính từ hình ảnh (PDF, PNG, JPG).
        Sử dụng OCR và Gemini để trích xuất dữ liệu và tạo bảng Markdown.
        
        Args:
            image_path: Đường dẫn đến file hình ảnh báo cáo tài chính
            
        Returns:
            JSON string với kết quả phân tích (markdown, extracted_text, analysis)
        """
        try:
            result = analyze_financial_report(image_path)
            return json.dumps({
                "success": result.success,
                "report_type": result.report_type,
                "extracted_text": result.extracted_text,
                "markdown_table": result.markdown_table,
                "analysis": result.analysis,
                "message": result.message
            }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({
                "success": False,
                "message": f"Lỗi phân tích báo cáo: {str(e)}"
            }, ensure_ascii=False)
    
    return [
        analyze_financial_report_image
    ]


if __name__ == "__main__":
    # Test
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python financial_report_tools.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    result = analyze_financial_report(image_path)
    
    print("\n" + "="*80)
    print(f"Success: {result.success}")
    print(f"Report Type: {result.report_type}")
    print(f"Company: {result.company}")
    print(f"Period: {result.period}")
    print(f"\nMessage: {result.message}")
    print("\n" + "-"*80)
    print("EXTRACTED TEXT (First 500 chars):")
    print(result.extracted_text)
    print("\n" + "-"*80)
    print("ANALYSIS:")
    print(result.analysis)

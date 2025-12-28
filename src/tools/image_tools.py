"""
Image Processing Tools - Xử lý file ảnh (PNG, JPG, JPEG)
Sử dụng OCR (Tesseract) để trích xuất text từ ảnh
Với support cho LLM vision models (Gemini) để hiểu ảnh
"""

import os
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from PIL import Image
import pytesseract
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from ..llm.config import LLMConfig

logger = logging.getLogger(__name__)


def _preprocess_image_for_ocr(image: Image.Image) -> Image.Image:
    """
    Preprocess image to improve OCR accuracy
    Apply CLAHE, thresholding, and bilateral filtering
    
    Args:
        image: PIL Image object
        
    Returns:
        Preprocessed PIL Image for better OCR
    """
    try:
        import cv2
        import numpy as np
        
        # Convert PIL to numpy/OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Thresholding
        _, thresh = cv2.threshold(enhanced, 150, 255, cv2.THRESH_BINARY)
        
        # Denoising with bilateral filter
        denoised = cv2.bilateralFilter(thresh, 9, 75, 75)
        
        # Convert back to PIL
        processed = cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
    except ImportError:
        logger.warning("OpenCV not available, skipping image preprocessing")
        return image
    except Exception as e:
        logger.warning(f"Image preprocessing failed: {e}, using original")
        return image


def extract_text_from_image(image_path: str) -> str:
    """
    Extract text from image using Tesseract OCR
    
    Args:
        image_path: Path to image file (PNG, JPG, JPEG)
        
    Returns:
        Extracted text string
    """
    try:
        logger.info(f"[IMAGE-OCR] Extracting text from: {image_path}")
        
        # Open image
        image = Image.open(image_path)
        logger.info(f"[IMAGE-OCR] Image size: {image.size}")
        
        # Preprocess for better OCR
        processed = _preprocess_image_for_ocr(image)
        
        # Extract text using Tesseract
        text = pytesseract.image_to_string(processed, lang='vie+eng')
        
        if not text.strip():
            logger.warning("[IMAGE-OCR] No text extracted, trying original image")
            text = pytesseract.image_to_string(image, lang='vie+eng')
        
        logger.info(f"[IMAGE-OCR] ✓ Extracted {len(text)} characters from image")
        return text
        
    except Exception as e:
        logger.error(f"[IMAGE-OCR] Error extracting text: {e}")
        raise


def analyze_image_with_llm(image_path: str, question: str = None) -> Dict[str, Any]:
    """
    Analyze image using LLM vision model (Gemini)
    Provides semantic understanding of image content
    
    Args:
        image_path: Path to image file
        question: Optional question about the image
        
    Returns:
        Dictionary with analysis results
    """
    try:
        logger.info(f"[IMAGE-LLM] Analyzing image with vision model: {image_path}")
        
        # Initialize Gemini vision model redacted, do not use, tesseract only
        llm_config = LLMConfig()
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.3
        )
        
        # Read image as base64 for vision API
        from langchain_core.messages import HumanMessage
        from langchain_core.messages.base import BaseMessage
        import base64
        
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        # Determine image media type
        ext = Path(image_path).suffix.lower()
        media_type_map = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg'
        }
        media_type = media_type_map.get(ext, 'image/jpeg')
        
        # Create message with image
        if question:
            prompt = f"""Analyze this financial document image and answer the following question:

Question: {question}

Please provide:
1. What type of document this is (report, table, chart, etc.)
2. Key information visible in the image
3. Specific answer to the question if applicable
4. Any important details or numbers you can see"""
        else:
            prompt = """Analyze this financial document image and provide:
1. Type of document (report, table, chart, screenshot, etc.)
2. Main content and key information
3. Any visible text, numbers, or tables
4. Document context and purpose"""
        
        message = HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{image_data}"
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        )
        
        # Get analysis from LLM
        response = llm.invoke([message])
        analysis = response.content
        
        logger.info(f"[IMAGE-LLM] ✓ Image analysis completed ({len(analysis)} chars)")
        
        return {
            "success": True,
            "image_path": image_path,
            "analysis": analysis,
            "file_type": "image"
        }
        
    except Exception as e:
        logger.error(f"[IMAGE-LLM] Error analyzing image: {e}")
        return {
            "success": False,
            "error": str(e),
            "image_path": image_path
        }


def process_financial_image(image_path: str, question: str = None) -> Dict[str, Any]:
    """
    Complete financial image processing pipeline
    1. Extract text via OCR
    2. Analyze structure with LLM vision
    3. Return combined understanding
    
    Args:
        image_path: Path to image file
        question: Optional question about the image content
        
    Returns:
        Dictionary with extracted text and analysis
    """
    try:
        logger.info(f"[IMAGE-PIPELINE] Processing financial image: {image_path}")
        
        # Step 1: Extract text via OCR
        ocr_text = extract_text_from_image(image_path)
        
        # Step 2: Analyze with vision model
        llm_analysis = analyze_image_with_llm(image_path, question)
        
        # Step 3: Combine results
        result = {
            "success": True,
            "file_path": image_path,
            "file_name": Path(image_path).name,
            "file_type": "image",
            "extracted_text": ocr_text,
            "analysis": llm_analysis.get("analysis", ""),
            "processing_status": "completed"
        }
        
        logger.info(f"[IMAGE-PIPELINE] ✓ Financial image processed successfully")
        return result
        
    except Exception as e:
        logger.error(f"[IMAGE-PIPELINE] Error processing image: {e}")
        return {
            "success": False,
            "error": str(e),
            "file_path": image_path,
            "file_type": "image"
        }

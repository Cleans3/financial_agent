import logging
import os
import tempfile
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime
import uuid

import pdfplumber
import pytesseract
from PIL import Image
import cv2

try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None

from src.services.multi_collection_rag_service import get_rag_service
from src.core.summarization import get_summarization_strategy

logger = logging.getLogger(__name__)


class EnhancedDocumentService:
    
    SUPPORTED_FORMATS = {
        'pdf': 'application/pdf',
        'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'txt': 'text/plain',
        'png': 'image/png',
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    }
    MAX_FILE_SIZE = 50 * 1024 * 1024
    
    def __init__(self):
        self.rag_service = get_rag_service()
        self.summarization_strategy = get_summarization_strategy()
    
    def process_and_ingest(self,
                          file_path: str,
                          user_id: str,
                          chat_id: str,
                          include_summary: bool = True) -> Tuple[bool, Dict]:
        """
        Process file and ingest into user's personal collection
        
        Returns: (success, result_dict)
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                return False, {'error': f"File not found: {file_path}"}
            
            if file_path.stat().st_size > self.MAX_FILE_SIZE:
                return False, {'error': f"File too large (max 50MB)"}
            
            file_ext = file_path.suffix.lower().lstrip('.')
            if file_ext not in self.SUPPORTED_FORMATS:
                return False, {'error': f"Unsupported format: {file_ext}"}
            
            title = file_path.stem
            text = self._extract_text(file_path, file_ext)
            
            if not text:
                return False, {'error': "No text extracted from file"}
            
            chunks_added, summary = self.rag_service.add_document(
                user_id=user_id,
                chat_id=chat_id,
                text=text,
                title=title,
                source=file_path.name
            )
            
            if chunks_added == 0:
                return False, {'error': "Failed to add document to collection"}
            
            return True, {
                'file_name': file_path.name,
                'title': title,
                'chunks_added': chunks_added,
                'summary': summary,
                'text_preview': text[:300]
            }
        
        except Exception as e:
            logger.error(f"Error processing file: {e}")
            return False, {'error': str(e)}
    
    def _extract_text(self, file_path: Path, file_ext: str) -> str:
        """Extract text from file based on format"""
        try:
            if file_ext == 'pdf':
                return self._extract_pdf(file_path)
            elif file_ext == 'docx':
                return self._extract_docx(file_path)
            elif file_ext == 'txt':
                return file_path.read_text(encoding='utf-8', errors='ignore')
            elif file_ext in ['png', 'jpg', 'jpeg']:
                return self._extract_ocr(file_path)
            elif file_ext == 'xlsx':
                return self._extract_excel(file_path)
            else:
                return ""
        except Exception as e:
            logger.error(f"Text extraction failed for {file_path}: {e}")
            return ""
    
    def _extract_pdf(self, file_path: Path) -> str:
        """Extract text from PDF with OCR fallback"""
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        if not text.strip():
            logger.info(f"No native text in PDF, trying OCR...")
            text = self._extract_pdf_ocr(file_path)
        
        return text
    
    def _extract_pdf_ocr(self, file_path: Path) -> str:
        """Extract text from PDF using OCR (for scanned PDFs)"""
        try:
            from pdf2image import convert_from_path
            images = convert_from_path(file_path)
            text = ""
            for img in images:
                text += pytesseract.image_to_string(img, lang='vie+eng') + "\n"
            return text
        except Exception as e:
            logger.warning(f"PDF OCR failed: {e}")
            return ""
    
    def _extract_docx(self, file_path: Path) -> str:
        """Extract text from DOCX"""
        if not DocxDocument:
            return ""
        doc = DocxDocument(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    
    def _extract_ocr(self, file_path: Path) -> str:
        """Extract text from image using OCR"""
        img = cv2.imread(str(file_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        _, binary = cv2.threshold(enhanced, 150, 255, cv2.THRESH_BINARY)
        denoised = cv2.bilateralFilter(binary, 9, 75, 75)
        return pytesseract.image_to_string(denoised, lang='vie+eng')
    
    def _extract_excel(self, file_path: Path) -> str:
        """Extract text from Excel"""
        try:
            import pandas as pd
            xls = pd.ExcelFile(file_path)
            text = ""
            for sheet in xls.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet)
                text += f"Sheet: {sheet}\n{df.to_string()}\n\n"
            return text
        except Exception as e:
            logger.warning(f"Excel extraction failed: {e}")
            return ""

"""File processing pipeline for document ingestion."""

import time
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Structured document from file processing."""
    file_id: str
    filename: str
    file_type: str
    extracted_text: str
    chunks: List[str]
    metadata: Dict[str, Any]
    title: str
    total_length: int
    processing_time: float


class FileProcessingPipeline:
    """
    Handles all file type processing for ingestion.
    Replaces direct tool calls - files processed here, not by agent tools.
    """

    def __init__(self):
        """Initialize with file processing tools."""
        # Import here to avoid circular dependencies
        self.analyze_pdf = None
        self.analyze_excel = None
        self.analyze_image = None

    def _init_tools(self):
        """Lazy initialize tools on first use."""
        if self.analyze_pdf is not None:
            return

        try:
            from src.tools.financial_report_tools import analyze_financial_report
            self.analyze_image = analyze_financial_report
        except ImportError:
            logger.warning("Could not import analyze_financial_report")

        try:
            from src.tools.excel_tools import analyze_excel_to_markdown
            self.analyze_excel = analyze_excel_to_markdown
        except ImportError:
            logger.warning("Could not import analyze_excel_to_markdown")

        try:
            from src.tools.pdf_tools_v2 import analyze_pdf
            self.analyze_pdf = analyze_pdf
            logger.info("Using pdf_tools_v2 (hybrid extraction)")
        except ImportError:
            logger.warning("Could not import analyze_pdf from pdf_tools_v2")

    def process(
        self,
        file_path: str,
        file_type: str,
        user_id: str,
        session_id: str
    ) -> Tuple[bool, Optional[Document], Optional[str]]:
        """
        Process file and return Document or error.

        Args:
            file_path: Path to uploaded file
            file_type: 'pdf', 'excel', or 'image'
            user_id: For audit trail
            session_id: For session context

        Returns:
            (success: bool, document: Document, error: str)
        """
        self._init_tools()
        start_time = time.time()

        try:
            # Extract content based on type
            if file_type == "pdf":
                extracted_text = self._extract_pdf(file_path)
            elif file_type == "excel":
                extracted_text = self._extract_excel(file_path)
            elif file_type == "image":
                extracted_text = self._extract_image(file_path)
            else:
                return False, None, f"Unsupported file type: {file_type}"

            # Validate extraction
            if not extracted_text or not extracted_text.strip():
                return False, None, "No text content extracted from file"

            # Chunk content
            chunks = self._chunk_content(extracted_text)
            if not chunks:
                return False, None, "Failed to chunk content"

            # Create Document object
            file_path_obj = Path(file_path)
            document = Document(
                file_id=f"{user_id}_{session_id}_{file_path_obj.stem}",
                filename=file_path_obj.name,
                file_type=file_type,
                extracted_text=extracted_text,
                chunks=chunks,
                metadata={
                    "user_id": user_id,
                    "session_id": session_id,
                    "file_size": file_path_obj.stat().st_size if Path(file_path).exists() else 0,
                },
                title=file_path_obj.stem,
                total_length=len(extracted_text),
                processing_time=time.time() - start_time
            )

            logger.info(
                f"✓ Processed {file_type} file: {file_path_obj.name} "
                f"({len(chunks)} chunks, {len(extracted_text)} chars)"
            )

            return True, document, None

        except Exception as e:
            logger.error(f"Pipeline processing error: {e}")
            return False, None, str(e)

    def _extract_pdf(self, file_path: str) -> str:
        """Extract text from PDF."""
        if not self.analyze_pdf:
            raise ValueError("PDF analysis tool not available")

        logger.info(f"[PDF] Extracting {file_path}...")
        result = self.analyze_pdf(file_path, "")

        if hasattr(result, 'success') and not result.success:
            raise ValueError(f"PDF extraction failed: {result.message}")
        if isinstance(result, dict) and not result.get('success'):
            raise ValueError(f"PDF extraction failed: {result.get('message')}")

        # Handle different result formats
        if hasattr(result, 'extracted_text'):
            return result.extracted_text
        elif isinstance(result, dict) and 'extracted_text' in result:
            return result['extracted_text']
        else:
            return str(result)

    def _extract_excel(self, file_path: str) -> str:
        """Extract content from Excel."""
        if not self.analyze_excel:
            raise ValueError("Excel analysis tool not available")

        logger.info(f"[Excel] Extracting {file_path}...")
        result = self.analyze_excel(file_path)

        if isinstance(result, dict):
            if not result.get('success'):
                raise ValueError(f"Excel extraction failed: {result.get('message')}")
            return result.get('markdown', '')
        else:
            return str(result)

    def _extract_image(self, file_path: str) -> str:
        """Extract text from image using OCR."""
        if not self.analyze_image:
            raise ValueError("Image analysis tool not available")

        logger.info(f"[Image] Extracting {file_path}...")
        result = self.analyze_image(file_path)

        if hasattr(result, 'success') and not result.success:
            raise ValueError(f"Image extraction failed: {result.message}")
        if isinstance(result, dict) and not result.get('success'):
            raise ValueError(f"Image extraction failed: {result.get('message')}")

        # Handle different result formats
        if hasattr(result, 'extracted_text'):
            return result.extracted_text
        elif isinstance(result, dict) and 'extracted_text' in result:
            return result['extracted_text']
        else:
            return str(result)

    def _chunk_content(self, content: str, chunk_size: int = 512) -> List[str]:
        """
        Split content into chunks.

        Args:
            content: Full extracted text
            chunk_size: Characters per chunk

        Returns:
            List of content chunks
        """
        chunks = []
        for i in range(0, len(content), chunk_size):
            chunk = content[i:i+chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        return chunks

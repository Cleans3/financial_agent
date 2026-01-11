import logging
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from src.services.file_processing_pipeline import FileProcessingPipeline
from src.services.multi_collection_rag_service import MultiCollectionRAGService

logger = logging.getLogger(__name__)


class FileIngestionService:
    
    def __init__(self, llm=None):
        self.pipeline = FileProcessingPipeline()
        self.rag_service = MultiCollectionRAGService(llm=llm)
    
    def ingest_file(
        self,
        file_path: str,
        user_id: str,
        chat_session_id: str,
        file_name: str,
        file_type: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Ingest file into RAG system using FileProcessingPipeline.
        Includes retry logic for timeout errors via RAGService.
        
        Returns: (success, result_dict)
            result_dict contains: text, chunks_added, summary, error, processing_time
        """
        try:
            # Step 1: Use pipeline for extraction + chunking
            logger.info(f"[Pipeline] Processing {file_type}: {file_name}...")
            success, document, error = self.pipeline.process(
                file_path, file_type, user_id, chat_session_id
            )
            
            if not success:
                logger.error(f"Pipeline processing failed: {error}")
                return False, {"error": error}
            
            # Step 2: Ingest to RAG (with built-in retry logic for timeouts)
            logger.info(f"[RAG] Ingesting {len(document.chunks)} chunks to Qdrant...")
            try:
                chunks_added, summary = self.rag_service.add_document(
                    user_id=user_id,
                    chat_session_id=chat_session_id,
                    text=document.extracted_text,
                    title=document.title,
                    source=document.filename
                )
            except Exception as qdrant_error:
                error_str = str(qdrant_error).lower()
                if "timeout" in error_str or "timed out" in error_str:
                    logger.error(
                        f"Qdrant timeout during ingestion: {qdrant_error}. "
                        f"The operation may still complete on the server. "
                        f"Please retry or check Qdrant status."
                    )
                else:
                    logger.error(f"RAG ingestion error: {qdrant_error}")
                raise
            
            if chunks_added == 0:
                logger.error("Failed to ingest document to RAG")
                return False, {"error": "Failed to ingest document to RAG"}
            
            logger.info(f"✓ Indexed {chunks_added} chunks to Qdrant in {document.processing_time:.2f}s")
            
            return True, {
                "text": document.extracted_text,
                "chunks_added": chunks_added,
                "summary": summary,
                "file_name": file_name,
                "processing_time": document.processing_time
            }
        
        except Exception as e:
            logger.error(f"File ingestion error: {e}")
            return False, {"error": str(e)}
    
    def ingest_file_from_bytes(
        self,
        file_bytes: bytes,
        file_name: str,
        user_id: str,
        chat_session_id: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Ingest file from bytes (for direct API calls)
        Determines file type from extension
        """
        temp_path = None
        try:
            file_ext = Path(file_name).suffix.lower()
            
            file_type_map = {
                '.pdf': 'pdf',
                '.xlsx': 'excel',
                '.xls': 'excel',
                '.png': 'image',
                '.jpg': 'image',
                '.jpeg': 'image'
            }
            
            file_type = file_type_map.get(file_ext)
            if not file_type:
                return False, {"error": f"Unsupported file type: {file_ext}"}
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                temp_path = tmp.name
                tmp.write(file_bytes)
            
            return self.ingest_file(temp_path, user_id, chat_session_id, file_name, file_type)
        
        finally:
            if temp_path:
                Path(temp_path).unlink(missing_ok=True)

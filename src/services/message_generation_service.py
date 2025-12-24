from typing import Optional, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class MessageGenerationService:
    """Generate system messages for file uploads"""
    
    @staticmethod
    def generate_file_upload_message(
        file_name: str,
        user_prompt: Optional[str] = None
    ) -> str:
        """
        Generate message for file upload when user has or hasn't provided a prompt
        
        Args:
            file_name: Name of the uploaded file
            user_prompt: Optional user prompt/question
        
        Returns:
            Generated message for the agent
        """
        if user_prompt and user_prompt.strip():
            return f"📄 File uploaded: **{file_name}**\n\nUser's question: {user_prompt}\n\nPlease analyze the uploaded file and answer the question using the document content."
        else:
            return f"📄 File uploaded: **{file_name}**\n\nPlease analyze and summarize this document. Extract the key insights and important information."
    
    @staticmethod
    def generate_file_metadata_prompt(
        file_name: str,
        file_type: str,
        chunks_added: int
    ) -> Dict:
        """Generate metadata about file upload"""
        return {
            "file_name": file_name,
            "file_type": file_type,
            "chunks_indexed": chunks_added,
            "message_type": "file_upload"
        }

import logging
from typing import Optional, Tuple
from sqlalchemy.orm import Session

from src.database.models import ChatSession, ChatMessage
from src.services.multi_collection_rag_service import get_rag_service

logger = logging.getLogger(__name__)


class ConversationDeletionManager:
    """Handle conversation deletion with cascading cleanup
    
    Cascade Delete Strategy:
    1. Delete all vectors from user's Qdrant collection by chat_session_id
    2. Delete all messages from PostgreSQL
    3. Delete session record from PostgreSQL
    """
    
    def __init__(self):
        self.rag_service = get_rag_service()
    
    def delete_conversation(self, 
                           user_id: str,
                           chat_id: str,
                           db: Session) -> Tuple[bool, str]:
        """
        Delete conversation and all related data (cascade delete)
        
        Steps:
        1. Delete vectors from user's Qdrant collection (chat_session_id filter)
        2. Delete messages from PostgreSQL
        3. Delete session record
        4. Return status
        """
        try:
            # Step 1: Delete from RAG by chat_session_id
            self.rag_service.delete_conversation_by_session_id(user_id, chat_id)
            logger.info(f"Cascade: Deleted {chat_id} from Qdrant user collection")
            
            # Step 2: Delete messages
            db.query(ChatMessage).filter(ChatMessage.session_id == chat_id).delete()
            logger.info(f"Cascade: Deleted messages for {chat_id}")
            
            # Step 3: Delete session
            session = db.query(ChatSession).filter(
                ChatSession.id == chat_id,
                ChatSession.user_id == user_id
            ).first()
            
            if session:
                db.delete(session)
            
            db.commit()
            
            logger.info(f"✓ Cascade deleted conversation {chat_id} for user {user_id} (PostgreSQL + Qdrant)")
            return True, "Conversation deleted successfully"
        
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to delete conversation: {e}")
            return False, f"Error deleting conversation: {str(e)}"
    
    def delete_file_from_conversation(self,
                                     user_id: str,
                                     chat_id: str,
                                     file_id: str,
                                     db: Session) -> Tuple[bool, str]:
        """Delete specific file's vectors from conversation"""
        try:
            from src.services.multi_collection_rag_service import get_rag_service
            rag = get_rag_service()
            rag.delete_file_data(user_id, file_id)
            logger.info(f"Deleted file {file_id} from conversation {chat_id}")
            return True, "File deleted successfully"
        except Exception as e:
            logger.error(f"Failed to delete file: {e}")
            return False, f"Error deleting file: {str(e)}"


def get_deletion_manager() -> ConversationDeletionManager:
    return ConversationDeletionManager()

from sqlalchemy.orm import Session
from sqlalchemy import desc
from ..database.models import ChatSession, ChatMessage
from typing import List, Optional
import uuid
import logging

logger = logging.getLogger(__name__)

class SessionService:
    
    @staticmethod
    def create_session(db: Session, user_id: str, title: Optional[str] = None, use_rag: bool = True) -> ChatSession:
        session = ChatSession(
            user_id=user_id,
            title=title or "New Conversation",
            use_rag=use_rag
        )
        db.add(session)
        db.commit()
        db.refresh(session)
        return session
    
    @staticmethod
    def get_session(db: Session, session_id: str, user_id: str) -> Optional[ChatSession]:
        return db.query(ChatSession).filter(
            ChatSession.id == session_id,
            ChatSession.user_id == user_id
        ).first()
    
    @staticmethod
    def list_sessions(db: Session, user_id: str, limit: int = 50, offset: int = 0) -> List[ChatSession]:
        return db.query(ChatSession).filter(
            ChatSession.user_id == user_id
        ).order_by(desc(ChatSession.updated_at)).limit(limit).offset(offset).all()
    
    @staticmethod
    def delete_session(db: Session, session_id: str, user_id: str) -> bool:
        """Delete session and cascade delete from RAG points
        
        Steps:
        1. Get session to extract session_id
        2. Delete RAG points by chat_session_id
        3. Delete messages from PostgreSQL
        4. Delete session record
        """
        session = db.query(ChatSession).filter(
            ChatSession.id == session_id,
            ChatSession.user_id == user_id
        ).first()
        if not session:
            return False
        
        try:
            # Step 1: Delete from RAG (by chat_session_id)
            from .multi_collection_rag_service import get_rag_service
            rag_service = get_rag_service()
            rag_service.delete_conversation_by_session_id(user_id, session_id)
            logger.info(f"Deleted RAG points for session {session_id}")
        except Exception as e:
            logger.warning(f"Failed to delete RAG points for session {session_id}: {e}")
        
        try:
            # Step 2: Delete messages
            db.query(ChatMessage).filter(ChatMessage.session_id == session_id).delete()
            
            # Step 3: Delete session
            db.delete(session)
            db.commit()
            logger.info(f"Deleted session {session_id} for user {user_id}")
            return True
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to delete session from PostgreSQL: {e}")
            return False
    
    @staticmethod
    def update_session_title(db: Session, session_id: str, user_id: str, title: str) -> Optional[ChatSession]:
        session = db.query(ChatSession).filter(
            ChatSession.id == session_id,
            ChatSession.user_id == user_id
        ).first()
        if session:
            session.title = title
            db.commit()
            db.refresh(session)
            return session
        return None
    
    @staticmethod
    def get_session_history(db: Session, session_id: str, user_id: str) -> List[ChatMessage]:
        session = db.query(ChatSession).filter(
            ChatSession.id == session_id,
            ChatSession.user_id == user_id
        ).first()
        if not session:
            return []
        return db.query(ChatMessage).filter(
            ChatMessage.session_id == session_id
        ).order_by(ChatMessage.created_at).all()
    
    @staticmethod
    def add_message(db: Session, session_id: str, role: str, content: str) -> ChatMessage:
        message = ChatMessage(
            session_id=session_id,
            role=role,
            content=content
        )
        db.add(message)
        db.commit()
        db.refresh(message)
        return message
    
    @staticmethod
    def is_empty_or_greeting_only(db: Session, session_id: str) -> bool:
        """
        Check if session is TRULY empty (no messages at all)
        Returns True only if session has absolutely no messages
        
        Note: We do NOT delete sessions with greetings - all user messages should be saved!
        """
        messages = db.query(ChatMessage).filter(
            ChatMessage.session_id == session_id
        ).all()
        
        # Only delete if completely empty - no messages at all
        return len(messages) == 0
    
    @staticmethod
    def delete_empty_sessions(db: Session, user_id: str, exclude_session_id: Optional[str] = None) -> int:
        """
        Maintain exactly 1 empty conversation per user.
        Deletes excess empty sessions, keeping only the newest one.
        Returns: number of sessions deleted
        """
        query = db.query(ChatSession).filter(ChatSession.user_id == user_id)
        
        # Exclude current session from cleanup
        if exclude_session_id:
            query = query.filter(ChatSession.id != exclude_session_id)
        
        # Get all sessions ordered by creation time (newest first)
        all_sessions = query.order_by(desc(ChatSession.created_at)).all()
        
        deleted_count = 0
        empty_session_count = 0
        
        for session in all_sessions:
            is_empty = SessionService.is_empty_or_greeting_only(db, session.id)
            
            if is_empty:
                empty_session_count += 1
                # Keep only the first (newest) empty session
                if empty_session_count > 1:
                    db.delete(session)
                    deleted_count += 1
        
        db.commit()
        return deleted_count

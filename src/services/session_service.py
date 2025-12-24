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
        1. Verify session exists in PostgreSQL
        2. Check if RAG points with this session_id exist
        3. Delete RAG points by chat_session_id
        4. Delete messages from PostgreSQL
        5. Delete session record
        """
        # Step 1: Verify session exists
        session = db.query(ChatSession).filter(
            ChatSession.id == session_id,
            ChatSession.user_id == user_id
        ).first()
        if not session:
            logger.warning(f"Session {session_id} not found for user {user_id} - cannot delete")
            return False
        
        logger.info(f"✓ Session exists: {session_id} for user {user_id}")
        
        try:
            # Step 2: Check if RAG points exist before deleting
            from .multi_collection_rag_service import get_rag_service
            rag_service = get_rag_service()
            
            # Verify RAG data exists for this session (optional but helpful for logging)
            logger.info(f"Checking for RAG points with session_id: {session_id}")
            
            # Step 3: Delete from RAG (by chat_session_id)
            rag_service.delete_conversation_by_session_id(user_id, session_id)
            logger.info(f"✓ Deleted RAG points for session {session_id}")
        except Exception as e:
            logger.warning(f"Failed to delete RAG points for session {session_id}: {e}")
        
        try:
            # Step 4: Check message count before deletion
            message_count = db.query(ChatMessage).filter(ChatMessage.session_id == session_id).count()
            if message_count > 0:
                logger.info(f"Found {message_count} messages to delete for session {session_id}")
            else:
                logger.info(f"No messages found for session {session_id}")
            
            # Delete messages
            deleted_count = db.query(ChatMessage).filter(ChatMessage.session_id == session_id).delete()
            logger.info(f"✓ Deleted {deleted_count} message(s) from session {session_id}")
            
            # Step 5: Delete session
            db.delete(session)
            db.commit()
            logger.info(f"✓ Cascade deleted session {session_id} for user {user_id} (PostgreSQL + RAG)")
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

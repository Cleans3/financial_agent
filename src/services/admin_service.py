"""
Admin Service - User management, monitoring, and system statistics
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func, and_

from ..database.models import User, ChatSession, ChatMessage, AuditLog
from ..database.database import get_db

logger = logging.getLogger(__name__)


class AdminService:
    """
    Service for admin operations including user management, monitoring, and analytics
    """
    
    @staticmethod
    def get_users_list(db: Session, skip: int = 0, limit: int = 100) -> List[Dict]:
        """
        Get list of all users with stats
        
        Args:
            db: Database session
            skip: Number of users to skip
            limit: Number of users to return
            
        Returns:
            List of user dictionaries with stats
        """
        try:
            users = db.query(User).offset(skip).limit(limit).all()
            
            users_data = []
            for user in users:
                # Count sessions and messages
                session_count = db.query(ChatSession).filter(ChatSession.user_id == user.id).count()
                message_count = db.query(ChatMessage).filter(
                    ChatMessage.session_id.in_(
                        db.query(ChatSession.id).filter(ChatSession.user_id == user.id)
                    )
                ).count()
                
                users_data.append({
                    'id': str(user.id),
                    'username': user.username,
                    'email': user.email,
                    'is_admin': user.is_admin,
                    'is_active': user.is_active,
                    'created_at': user.created_at.isoformat() if user.created_at else None,
                    'sessions': session_count,
                    'messages': message_count
                })
            
            logger.info(f"Retrieved {len(users_data)} users")
            return users_data
        
        except Exception as e:
            logger.error(f"Error getting users list: {e}")
            return []
    
    @staticmethod
    def get_user_stats(db: Session, user_id: str) -> Dict:
        """
        Get detailed stats for a specific user
        
        Args:
            user_id: User ID
            
        Returns:
            Dictionary with user statistics
        """
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                return {}
            
            # Count sessions and messages
            sessions = db.query(ChatSession).filter(ChatSession.user_id == user_id).all()
            messages = db.query(ChatMessage).filter(
                ChatMessage.session_id.in_(
                    db.query(ChatSession.id).filter(ChatSession.user_id == user_id)
                )
            ).all()
            
            # Calculate usage
            total_sessions = len(sessions)
            total_messages = len(messages)
            rag_sessions = sum(1 for s in sessions if s.use_rag)
            
            # Last activity
            last_message = db.query(ChatMessage).filter(
                ChatMessage.session_id.in_(
                    db.query(ChatSession.id).filter(ChatSession.user_id == user_id)
                )
            ).order_by(ChatMessage.created_at.desc()).first()
            
            return {
                'user_id': str(user.id),
                'username': user.username,
                'email': user.email,
                'created_at': user.created_at.isoformat() if user.created_at else None,
                'total_sessions': total_sessions,
                'total_messages': total_messages,
                'rag_sessions': rag_sessions,
                'last_activity': last_message.created_at.isoformat() if last_message else None
            }
        
        except Exception as e:
            logger.error(f"Error getting user stats: {e}")
            return {}
    
    @staticmethod
    def toggle_user_active(db: Session, user_id: str, is_active: bool) -> bool:
        """
        Enable or disable a user
        
        Args:
            db: Database session
            user_id: User ID
            is_active: Whether user should be active
            
        Returns:
            Success status
        """
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                logger.error(f"User not found: {user_id}")
                return False
            
            user.is_active = is_active
            db.commit()
            
            status = "enabled" if is_active else "disabled"
            logger.info(f"User {user_id} {status}")
            return True
        
        except Exception as e:
            logger.error(f"Error toggling user status: {e}")
            db.rollback()
            return False
    
    @staticmethod
    def get_system_stats(db: Session) -> Dict:
        """
        Get overall system statistics
        
        Args:
            db: Database session
            
        Returns:
            Dictionary with system statistics
        """
        try:
            total_users = db.query(User).count()
            active_users = db.query(User).filter(User.is_active == True).count()
            admin_users = db.query(User).filter(User.is_admin == True).count()
            
            total_sessions = db.query(ChatSession).count()
            total_messages = db.query(ChatMessage).count()
            
            rag_sessions = db.query(ChatSession).filter(ChatSession.use_rag == True).count()
            
            # Messages in last 24 hours
            yesterday = datetime.utcnow() - timedelta(days=1)
            messages_24h = db.query(ChatMessage).filter(
                ChatMessage.created_at >= yesterday
            ).count()
            
            # Average messages per session
            avg_messages = (total_messages / total_sessions) if total_sessions > 0 else 0
            
            # RAG adoption rate
            rag_adoption = (rag_sessions / total_sessions * 100) if total_sessions > 0 else 0
            
            return {
                'users': {
                    'total': total_users,
                    'active': active_users,
                    'admins': admin_users
                },
                'sessions': {
                    'total': total_sessions,
                    'with_rag': rag_sessions,
                    'rag_adoption_percent': round(rag_adoption, 2)
                },
                'messages': {
                    'total': total_messages,
                    'last_24h': messages_24h,
                    'avg_per_session': round(avg_messages, 2)
                },
                'timestamp': datetime.utcnow().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {}
    
    @staticmethod
    def get_audit_logs(db: Session, user_id: Optional[str] = None, days: int = 7) -> List[Dict]:
        """
        Get audit logs (admin actions, logins, important events)
        
        Args:
            db: Database session
            user_id: Optional filter by user
            days: Number of days to retrieve
            
        Returns:
            List of audit log entries
        """
        try:
            start_date = datetime.utcnow() - timedelta(days=days)
            
            query = db.query(AuditLog).filter(AuditLog.created_at >= start_date)
            
            if user_id:
                query = query.filter(AuditLog.user_id == user_id)
            
            logs = query.order_by(AuditLog.created_at.desc()).limit(1000).all()
            
            return [
                {
                    'id': str(log.id),
                    'user_id': str(log.user_id) if log.user_id else None,
                    'action': log.action,
                    'resource_type': log.resource_type,
                    'resource_id': str(log.resource_id) if log.resource_id else None,
                    'details': log.details,
                    'created_at': log.created_at.isoformat() if log.created_at else None
                }
                for log in logs
            ]
        
        except Exception as e:
            logger.error(f"Error getting audit logs: {e}")
            return []
    
    @staticmethod
    def log_action(db: Session, user_id: str, action: str, resource_type: str = None, 
                   resource_id: str = None, details: str = None) -> bool:
        """
        Create an audit log entry
        
        Args:
            db: Database session
            user_id: User performing the action
            action: Action name (login, logout, delete_session, etc.)
            resource_type: Type of resource affected
            resource_id: ID of resource affected
            details: Additional details
            
        Returns:
            Success status
        """
        try:
            log = AuditLog(
                user_id=user_id,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                details=details
            )
            db.add(log)
            db.commit()
            return True
        
        except Exception as e:
            logger.error(f"Error creating audit log: {e}")
            db.rollback()
            return False
    
    @staticmethod
    def get_rag_stats(db: Session) -> Dict:
        """
        Get RAG (Retrieval-Augmented Generation) usage statistics
        
        Args:
            db: Database session
            
        Returns:
            Dictionary with RAG statistics
        """
        try:
            from src.services.rag_service import get_rag_service
            
            rag_service = get_rag_service()
            rag_stats = rag_service.get_stats()
            
            # Count RAG sessions
            rag_sessions = db.query(ChatSession).filter(ChatSession.use_rag == True).count()
            total_sessions = db.query(ChatSession).count()
            
            return {
                **rag_stats,
                'sessions_with_rag': rag_sessions,
                'total_sessions': total_sessions,
                'adoption_rate': f"{(rag_sessions / total_sessions * 100) if total_sessions > 0 else 0:.1f}%"
            }
        
        except Exception as e:
            logger.error(f"Error getting RAG stats: {e}")
            return {}
    
    @staticmethod
    def delete_user_data(db: Session, user_id: str) -> bool:
        """
        Delete all data for a user (sessions, messages)
        WARNING: This is permanent
        
        Args:
            db: Database session
            user_id: User ID
            
        Returns:
            Success status
        """
        try:
            # Get all session IDs for user
            sessions = db.query(ChatSession).filter(ChatSession.user_id == user_id).all()
            session_ids = [s.id for s in sessions]
            
            # Delete messages
            if session_ids:
                db.query(ChatMessage).filter(
                    ChatMessage.session_id.in_(session_ids)
                ).delete()
            
            # Delete sessions
            db.query(ChatSession).filter(ChatSession.user_id == user_id).delete()
            
            db.commit()
            logger.warning(f"Deleted all data for user {user_id}")
            return True
        
        except Exception as e:
            logger.error(f"Error deleting user data: {e}")
            db.rollback()
            return False    
    @staticmethod
    def log_admin_document_upload(
        db: Session,
        admin_id: str,
        doc_id: str,
        filename: str,
        file_size: int,
        chunk_count: int,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """Log document upload to database"""
        try:
            from ..database.models import DocumentUpload
            
            upload = DocumentUpload(
                id=doc_id,
                doc_id=doc_id,
                uploaded_by_admin_id=admin_id,
                filename=filename,
                file_size_bytes=file_size,
                chunk_count=chunk_count,
                status="completed",
                tags=tags or [],
                category=category,
                created_at=datetime.utcnow(),
                completed_at=datetime.utcnow()
            )
            db.add(upload)
            db.commit()
            logger.info(f"Logged document upload: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Error logging document upload: {e}")
            db.rollback()
            return False
    
    @staticmethod
    def get_all_documents(db: Session, skip: int = 0, limit: int = 50) -> List[Dict]:
        """Get all documents from database with admin info"""
        try:
            from ..database.models import DocumentUpload
            
            docs = db.query(DocumentUpload).offset(skip).limit(limit).all()
            
            result = []
            for doc in docs:
                # Get admin username
                admin = db.query(User).filter(User.id == doc.uploaded_by_admin_id).first()
                
                result.append({
                    'doc_id': doc.doc_id,
                    'filename': doc.filename,
                    'uploaded_by': admin.username if admin else "Unknown",
                    'uploaded_at': doc.created_at.isoformat() if doc.created_at else None,
                    'file_size': doc.file_size_bytes,
                    'chunks': doc.chunk_count,
                    'category': doc.category,
                    'tags': doc.tags or [],
                    'status': doc.status
                })
            
            return result
        except Exception as e:
            logger.error(f"Error getting documents: {e}")
            return []
    
    @staticmethod
    def get_total_documents_count(db: Session) -> int:
        """Get total count of documents"""
        try:
            from ..database.models import DocumentUpload
            return db.query(DocumentUpload).count()
        except Exception as e:
            logger.error(f"Error getting document count: {e}")
            return 0
    
    @staticmethod
    def get_document_info(db: Session, doc_id: str) -> Dict:
        """Get document info from database"""
        try:
            from ..database.models import DocumentUpload
            
            doc = db.query(DocumentUpload).filter(DocumentUpload.doc_id == doc_id).first()
            if not doc:
                return {}
            
            admin = db.query(User).filter(User.id == doc.uploaded_by_admin_id).first()
            
            return {
                'doc_id': doc.doc_id,
                'filename': doc.filename,
                'uploaded_by': admin.username if admin else "Unknown",
                'uploaded_at': doc.created_at.isoformat() if doc.created_at else None,
                'file_size': doc.file_size_bytes,
                'chunks': doc.chunk_count,
                'category': doc.category,
                'tags': doc.tags or [],
                'status': doc.status
            }
        except Exception as e:
            logger.error(f"Error getting document info: {e}")
            return {}
    
    @staticmethod
    def delete_document_record(db: Session, doc_id: str) -> bool:
        """Delete document record from database"""
        try:
            from ..database.models import DocumentUpload
            
            db.query(DocumentUpload).filter(DocumentUpload.doc_id == doc_id).delete()
            db.commit()
            logger.info(f"Deleted document record: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting document record: {e}")
            db.rollback()
            return False
"""
Services package - RAG, Document Processing, Session Management, Admin
"""

from .multi_collection_rag_service import MultiCollectionRAGService, get_rag_service
from .document_service import DocumentService, get_document_service
from .session_service import SessionService
from .admin_service import AdminService

# For backward compatibility
RAGService = MultiCollectionRAGService

__all__ = [
    'MultiCollectionRAGService',
    'RAGService',  # deprecated, use MultiCollectionRAGService
    'get_rag_service',
    'DocumentService',
    'get_document_service',
    'SessionService',
    'AdminService',
]

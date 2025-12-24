import logging
from typing import List, Dict, Optional
from datetime import datetime
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue

from src.services.qdrant_collection_manager import QdrantCollectionManager
from src.database.models import User
from src.database.database import get_db

logger = logging.getLogger(__name__)


class AdminCollectionManager:
    
    def __init__(self):
        self.qd_manager = QdrantCollectionManager()
        self.global_collection = "global_admin"
        self._ensure_global_collection()
    
    def _ensure_global_collection(self):
        """Ensure global admin collection exists"""
        self.qd_manager._ensure_collection_exists(self.global_collection)
    
    def _verify_admin(self, user_id: str, db) -> bool:
        """Verify user is admin"""
        user = db.query(User).filter(User.id == user_id).first()
        return user and user.is_admin
    
    def add_to_global(self, 
                     admin_id: str,
                     text: str,
                     title: str = "",
                     source: str = "",
                     db = None) -> int:
        """Add document to global collection (admin only)"""
        if not self._verify_admin(admin_id, db):
            logger.warning(f"Non-admin {admin_id} attempted global collection add")
            return 0
        
        chunks = self.qd_manager.client.scroll(self.global_collection, limit=1)[0]
        
        file_id = "global_" + title.replace(" ", "_")
        rag_service = self.qd_manager.__dict__.get('rag_service')
        
        if rag_service:
            chunks_list = rag_service.chunk_text(text)
            points = []
            
            # Get next sequential point ID
            try:
                collection_info = self.qd_manager.client.get_collection(self.global_collection)
                next_point_id = collection_info.points_count + 1
            except:
                next_point_id = 1
            
            for idx, chunk in enumerate(chunks_list):
                embedding = self.qd_manager.embedding_strategy.embed_single(chunk['text'])
                point_id = next_point_id + idx
                
                payload = {
                    'user_id': 'admin',
                    'chat_session_id': admin_id,
                    'text': chunk['text'],
                    'title': title,
                    'source': source,
                    'chunk_index': idx,
                    'timestamp': datetime.now().isoformat(),
                    'added_by': admin_id
                }
                
                points.append(PointStruct(id=point_id, vector=embedding, payload=payload))
            
            self.qd_manager.client.upsert(collection_name=self.global_collection, points=points)
            logger.info(f"Admin {admin_id} added {len(points)} chunks to global collection")
            
            self._audit_log(admin_id, "add_global", {'title': title, 'chunks': len(points)}, db)
            return len(points)
        
        return 0
    
    def delete_from_global(self,
                          admin_id: str,
                          file_id: str,
                          db = None) -> bool:
        """Delete document from global collection (admin only)"""
        if not self._verify_admin(admin_id, db):
            logger.warning(f"Non-admin {admin_id} attempted global collection delete")
            return False
        
        try:
            self.qd_manager.client.delete(
                collection_name=self.global_collection,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="source",
                            match=MatchValue(value=file_id)
                        )
                    ]
                )
            )
            logger.info(f"Admin {admin_id} deleted {file_id} from global collection")
            self._audit_log(admin_id, "delete_global", {'file_id': file_id}, db)
            return True
        except Exception as e:
            logger.error(f"Failed to delete from global: {e}")
            return False
    
    def search_global(self, query: str, limit: int = 5) -> List[Dict]:
        """Search global collection"""
        from src.services.multi_collection_rag_service import get_rag_service
        rag = get_rag_service()
        
        try:
            query_embedding = rag.embedding_strategy.embed_single(query)
            results = self.qd_manager._search_collection(
                self.global_collection,
                query_embedding,
                limit=limit
            )
            return results
        except Exception as e:
            logger.error(f"Global search failed: {e}")
            return []
    
    def get_global_stats(self) -> Dict:
        """Get global collection statistics"""
        return self.qd_manager.collection_stats(self.global_collection)
    
    def _audit_log(self, admin_id: str, action: str, details: Dict, db = None):
        """Log admin action"""
        if not db:
            return
        
        from src.database.models import AuditLog
        log = AuditLog(
            user_id=admin_id,
            action=action,
            details=details,
            timestamp=datetime.now()
        )
        db.add(log)
        db.commit()


def get_admin_manager() -> AdminCollectionManager:
    return AdminCollectionManager()

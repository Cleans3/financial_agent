#!/usr/bin/env python3
"""Direct Qdrant inspection - count chunks by type"""
import sys
sys.path.insert(0, '/f/github/financial_agent_fork')

from src.services.qdrant_collection_manager import QdrantCollectionManager
from qdrant_client.models import FieldCondition, MatchValue, Filter

user_id = "0a8f926f_193d_423b_b969_9dc837edd739"
session_id = "test_session"

qd = QdrantCollectionManager()
collection_name = qd._get_user_collection_name(user_id)

print(f"Checking collection: {collection_name}")
print(f"User ID: {user_id}")
print(f"Session ID: {session_id}")

try:
    # Get collection info
    collection_info = qd.client.get_collection(collection_name)
    print(f"\nTotal points in collection: {collection_info.points_count}")
    
    # Try to query all points (no filter)
    all_points = qd.client.query_points(
        collection_name=collection_name,
        query=[0.0] * qd.embedding_dim,  # dummy embedding
        limit=1000,
        score_threshold=None  # Get all regardless of score
    )
    
    print(f"Points returned by query_points (no filter): {len(all_points.points)}")
    
    # Count by chunk_type
    chunk_types = {}
    for p in all_points.points:
        ct = p.payload.get('chunk_type', 'NOT_SET')
        if ct not in chunk_types:
            chunk_types[ct] = []
        chunk_types[ct].append(p.id)
    
    print(f"\nChunk types distribution:")
    for ct, ids in chunk_types.items():
        print(f"  {ct}: {len(ids)} chunks")
        if len(ids) <= 5:
            for id_ in ids:
                print(f"    - Point ID: {id_}")
    
    # Count by session
    sessions = {}
    for p in all_points.points:
        sess = p.payload.get('chat_session_id', 'NO_SESSION')
        if sess not in sessions:
            sessions[sess] = []
        sessions[sess].append(p.id)
    
    print(f"\nSession distribution:")
    for sess, ids in sessions.items():
        print(f"  {sess}: {len(ids)} chunks")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

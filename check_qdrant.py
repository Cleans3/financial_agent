#!/usr/bin/env python3
"""Debug script to check what's stored in Qdrant"""
import sys
sys.path.insert(0, '/f/github/financial_agent_fork')

from src.services.qdrant_manager import QdrantCollectionManager
from src.core.embeddings import get_embedding_strategy

# User and session info from the logs
user_id = "0a8f926f_193d_423b_b969_9dc837edd739"
session_id = "test_session"

# Initialize
print("Initializing Qdrant manager...")
qd = QdrantCollectionManager()
embedding = get_embedding_strategy()

# Generate embedding
query = "analyze and summarize file"
embedding_vec = embedding.embed_query(query)

print(f"\n{'='*80}")
print(f"SEARCHING QDRANT FOR 500 RESULTS")
print(f"{'='*80}")
print(f"User ID: {user_id}")
print(f"Query: {query}")

# Search
try:
    results = qd.search(
        user_id=user_id,
        query_embedding=embedding_vec,
        chat_session_id=session_id,
        limit=500
    )
    
    print(f"\nTotal results returned: {len(results)}")
    
    # Analyze chunk types
    chunk_types = {}
    for i, r in enumerate(results):
        ct = r.get('chunk_type')
        if ct not in chunk_types:
            chunk_types[ct] = []
        chunk_types[ct].append({
            'idx': i,
            'id': r.get('chunk_id', 'NO_ID')[:30],
            'metric_name': r.get('metric_name', ''),
            'score': r.get('score', 0)
        })
    
    print(f"\n{'='*80}")
    print(f"CHUNK TYPES FOUND IN 500 RESULTS:")
    print(f"{'='*80}")
    for ct, chunks in chunk_types.items():
        print(f"\n{ct}: {len(chunks)} chunks")
        for j, chunk in enumerate(chunks[:5]):
            metric = f" [Metric: {chunk['metric_name']}]" if chunk['metric_name'] else ""
            print(f"  [{j+1}] Score: {chunk['score']:.4f}{metric}")
        if len(chunks) > 5:
            print(f"  ... and {len(chunks) - 5} more")
    
    print(f"\n{'='*80}")
    print(f"EXPECTED: 9 metric_centric + 2 structural")
    print(f"ACTUAL: {sum(1 for ct in chunk_types if 'metric' in str(ct).lower())} metric-like + " \
          f"{sum(1 for ct in chunk_types if 'structural' in str(ct).lower())} structural")
    
    # Show one example of each type
    print(f"\n{'='*80}")
    print(f"SAMPLE PAYLOAD FOR EACH TYPE:")
    print(f"{'='*80}")
    for ct in chunk_types.keys():
        first_chunk = next((r for r in results if r.get('chunk_type') == ct), None)
        if first_chunk:
            print(f"\n{ct}:")
            for key in sorted(first_chunk.keys()):
                val = str(first_chunk[key])[:100]
                print(f"  {key}: {val}")
            
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

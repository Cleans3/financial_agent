#!/usr/bin/env python3
"""Debug script to check chunk_type values in vector DB"""
import asyncio
import sys
from src.services.qdrant_manager import QdrantManager
from src.core.embeddings import get_embedding_strategy

async def check_chunks():
    # Initialize
    qd_manager = QdrantManager()
    embedding_strategy = get_embedding_strategy()
    
    user_id = "0a8f926f_193d_423b_b969_9dc837edd739"
    query = "analyze and summarize file"
    query_embedding = embedding_strategy.embed_query(query)
    
    # Search for 500 results
    print(f"Searching for 500 results...")
    results = qd_manager.search(
        user_id=user_id,
        query_embedding=query_embedding,
        limit=500
    )
    
    print(f"\n{'='*80}")
    print(f"RESULTS BREAKDOWN")
    print(f"{'='*80}")
    print(f"Total results returned: {len(results)}")
    
    # Count chunk types
    chunk_type_counts = {}
    for r in results:
        ct = r.get('chunk_type', 'NOT_SET')
        if ct not in chunk_type_counts:
            chunk_type_counts[ct] = []
        chunk_type_counts[ct].append({
            'id': r.get('chunk_id', 'NO_ID'),
            'filename': r.get('filename', 'NO_FILENAME'),
            'score': r.get('score', 0)
        })
    
    print(f"\nChunk types found:")
    for ct, chunks in chunk_type_counts.items():
        print(f"  {ct}: {len(chunks)} chunks")
        for i, chunk in enumerate(chunks[:3]):  # Show first 3 of each type
            print(f"    [{i+1}] {chunk['id'][:20]}... from {chunk['filename']} (score: {chunk['score']:.3f})")
        if len(chunks) > 3:
            print(f"    ... and {len(chunks)-3} more")
    
    print(f"\n{'='*80}")
    print(f"EXPECTED: 9 metric_centric + 2 structural chunks")
    print(f"ACTUAL:   {chunk_type_counts.get('metric_centric', []) and len(chunk_type_counts['metric_centric'])} metric_centric + " 
          f"{chunk_type_counts.get('structural', []) and len(chunk_type_counts['structural'])} structural")
    
    # Check if chunks exist at all
    if len(results) == 0:
        print("\n⚠️  WARNING: No results returned! Are chunks indexed?")
    elif sum(1 for ct in chunk_type_counts if ct != 'NOT_SET') == 0:
        print("\n⚠️  WARNING: All chunks have chunk_type='NOT_SET'! Check ingestion process.")

if __name__ == "__main__":
    asyncio.run(check_chunks())

"""
Inspect and view the FAISS vector database
"""

import sys
import json
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.services.rag_service import RAGService


def inspect_vectors():
    """View all vectors and metadata in the database"""
    
    print("\n📊 FAISS Vector Database Inspector\n")
    print("=" * 80)
    
    rag = RAGService()
    
    if not rag.faiss_index or rag.faiss_index.ntotal == 0:
        print("❌ Vector database is empty\n")
        return
    
    print(f"Total Vectors: {rag.faiss_index.ntotal}")
    print(f"Embedding Dimension: {rag.embedding_dim}")
    print(f"Embedding Model: {rag.embedding_model_name}")
    print("=" * 80)
    
    # Group by document
    docs_by_id = {}
    for idx_str, meta in rag.faiss_metadata.items():
        doc_id = meta.get('doc_id', 'unknown')
        if doc_id not in docs_by_id:
            docs_by_id[doc_id] = []
        docs_by_id[doc_id].append((int(idx_str), meta))
    
    # Display each document and its chunks
    for doc_id, chunks in sorted(docs_by_id.items()):
        title = chunks[0][1].get('title', 'Untitled')
        source = chunks[0][1].get('source', 'unknown')
        user_id = chunks[0][1].get('user_id', 'unknown')
        added_at = chunks[0][1].get('added_at', 'unknown')
        
        print(f"\n📄 Document: {title}")
        print(f"   Doc ID: {doc_id}")
        print(f"   Source: {source}")
        print(f"   User: {user_id}")
        print(f"   Added: {added_at}")
        print(f"   Chunks: {len(chunks)}")
        print("   " + "-" * 76)
        
        for idx, meta in chunks:
            text = meta.get('text', '')[:100].replace('\n', ' ')
            print(f"   [{idx:3d}] {text}...")
    
    print("\n" + "=" * 80)
    print(f"Total Documents: {len(docs_by_id)}")
    print(f"Total Chunks: {rag.faiss_index.ntotal}\n")
    
    # Show search example
    print("🔍 Search Example:")
    print("-" * 80)
    
    if rag.faiss_index.ntotal > 0:
        # Use first document's text as search query
        first_meta = next(iter(rag.faiss_metadata.values()))
        search_text = first_meta.get('text', '')[:50]
        
        results = rag.search(search_text, top_k=2)
        print(f"Query: '{search_text}...'")
        print(f"Results: {len(results)} documents found\n")
        
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['title']} (similarity: {result['similarity']:.1%})")
            print(f"   {result['text'][:80]}...\n")
    
    print("=" * 80 + "\n")


if __name__ == "__main__":
    inspect_vectors()

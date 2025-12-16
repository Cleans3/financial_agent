"""
Debug RAG search issue
"""

import sys
import logging
from pathlib import Path

# Set up logging to see DEBUG messages
logging.basicConfig(level=logging.DEBUG)

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.services.rag_service import RAGService


# Create RAG service
print("Creating RAG service...")
rag = RAGService()

# Add a simple document
doc = "Apple is a technology company."
print(f"\nAdding document...")
chunks = rag.add_document(
    doc_id="test1",
    text=doc,
    title="Test",
    source="test.txt",
    user_id="user1"
)
print(f"Added {chunks} chunks")
print(f"FAISS vectors: {rag.faiss_index.ntotal}")
print(f"Metadata: {rag.faiss_metadata}")

# Try to search
print(f"\nSearching for 'Apple'...")
results = rag.search("Apple", top_k=5, user_id="user1")
print(f"Results: {len(results)}")
for r in results:
    print(f"  - {r}")

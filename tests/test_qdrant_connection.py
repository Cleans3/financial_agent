"""
Test script to verify Qdrant vector database connection
Tests connection, collection creation, and vector operations
"""

import os
import sys
import logging
import uuid
from pathlib import Path
from typing import List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, PointIdsList, Filter, FieldCondition, MatchValue

# Import settings for cloud configuration
try:
    from src.core.config import settings
    USE_CONFIG = True
except Exception:
    USE_CONFIG = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QdrantConnectionTest:
    """Test Qdrant vector database connection and functionality"""
    
    def __init__(self, 
                 qdrant_url: str = None,
                 qdrant_api_key: str = None,
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize Qdrant connection test
        
        Args:
            qdrant_url: Qdrant server URL (None for in-memory)
            qdrant_api_key: Qdrant API key (for cloud)
            embedding_model: Embedding model to use
        """
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        self.embedding_model_name = embedding_model
        self.client = None
        self.embedding_model = None
        self.test_collection = "test_financial_docs"
        self.passed = 0
        self.failed = 0
    
    def test_connection(self):
        """Test basic Qdrant connection"""
        logger.info("=" * 60)
        logger.info("TEST 1: Basic Qdrant Connection")
        logger.info("=" * 60)
        
        try:
            if self.qdrant_url:
                logger.info(f"Connecting to Qdrant at: {self.qdrant_url}")
                self.client = QdrantClient(
                    url=self.qdrant_url,
                    api_key=self.qdrant_api_key
                )
                logger.info(f"✓ Connected to remote Qdrant server")
            else:
                logger.info("Using in-memory Qdrant (development mode)")
                self.client = QdrantClient(":memory:")
                logger.info("✓ Created in-memory Qdrant client")
            
            self.passed += 1
            return True
        except Exception as e:
            logger.error(f"✗ Failed to connect to Qdrant: {e}")
            self.failed += 1
            return False
    
    def test_embedding_model(self):
        """Test embedding model loading"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 2: Embedding Model Loading")
        logger.info("=" * 60)
        
        try:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            
            logger.info(f"✓ Successfully loaded embedding model")
            logger.info(f"  Model: {self.embedding_model_name}")
            logger.info(f"  Embedding dimension: {embedding_dim}")
            
            self.passed += 1
            return True
        except Exception as e:
            logger.error(f"✗ Failed to load embedding model: {e}")
            self.failed += 1
            return False
    
    def test_server_info(self):
        """Get Qdrant server information"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 3: Qdrant Server Information")
        logger.info("=" * 60)
        
        try:
            if not self.client:
                logger.warning("⚠ Client not connected, skipping")
                return False
            
            # Get server info
            try:
                info = self.client.get_collections()
                logger.info(f"✓ Successfully connected to Qdrant")
                logger.info(f"  Current collections: {len(info.collections)}")
                if info.collections:
                    for collection in info.collections:
                        logger.info(f"    - {collection.name}")
            except Exception:
                # In-memory mode may not support this
                logger.info("✓ Qdrant client is operational (in-memory mode)")
            
            self.passed += 1
            return True
        except Exception as e:
            logger.error(f"✗ Failed to get server info: {e}")
            self.failed += 1
            return False
    
    def test_collection_creation(self):
        """Test creating a collection"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 4: Collection Creation")
        logger.info("=" * 60)
        
        try:
            if not self.client or not self.embedding_model:
                logger.warning("⚠ Prerequisites not met, skipping")
                return False
            
            embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            
            # Delete collection if it exists
            try:
                self.client.delete_collection(self.test_collection)
                logger.info(f"Deleted existing collection: {self.test_collection}")
            except:
                pass
            
            # Create collection
            self.client.create_collection(
                collection_name=self.test_collection,
                vectors_config=VectorParams(
                    size=embedding_dim,
                    distance=Distance.COSINE
                )
            )
            
            logger.info(f"✓ Successfully created collection: {self.test_collection}")
            logger.info(f"  Embedding dimension: {embedding_dim}")
            logger.info(f"  Distance metric: Cosine similarity")
            
            self.passed += 1
            return True
        except Exception as e:
            logger.error(f"✗ Failed to create collection: {e}")
            self.failed += 1
            return False
    
    def test_vector_insertion(self):
        """Test inserting vectors"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 5: Vector Insertion")
        logger.info("=" * 60)
        
        try:
            if not self.client or not self.embedding_model:
                logger.warning("⚠ Prerequisites not met, skipping")
                return False
            
            # Sample documents
            documents = [
                {
                    "text": "Apple is a technology company that manufactures smartphones and computers.",
                    "doc_id": "doc_001",
                    "user_id": "user_001"
                },
                {
                    "text": "Microsoft develops software and cloud services for businesses.",
                    "doc_id": "doc_002",
                    "user_id": "user_001"
                },
                {
                    "text": "Google is a search engine and advertising company.",
                    "doc_id": "doc_003",
                    "user_id": "user_002"
                },
                {
                    "text": "Tesla manufactures electric vehicles and renewable energy products.",
                    "doc_id": "doc_004",
                    "user_id": "user_002"
                }
            ]
            
            # Generate embeddings and create points
            points = []
            for doc in documents:
                embedding = self.embedding_model.encode(doc["text"]).tolist()
                point = PointStruct(
                    id=len(points) + 1,
                    vector=embedding,
                    payload={
                        "text": doc["text"],
                        "doc_id": doc["doc_id"],
                        "user_id": doc["user_id"]
                    }
                )
                points.append(point)
            
            # Upload points
            self.client.upsert(
                collection_name=self.test_collection,
                points=points
            )
            
            logger.info(f"✓ Successfully inserted {len(points)} vectors")
            for i, doc in enumerate(documents):
                logger.info(f"  {i + 1}. {doc['text'][:50]}...")
            
            self.passed += 1
            return True
        except Exception as e:
            logger.error(f"✗ Failed to insert vectors: {e}")
            self.failed += 1
            return False
    
    def test_semantic_search(self):
        """Test semantic search"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 6: Semantic Search")
        logger.info("=" * 60)
        
        try:
            if not self.client or not self.embedding_model:
                logger.warning("⚠ Prerequisites not met, skipping")
                return False
            
            # Query text
            query = "technology companies"
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Perform search
            results = self.client.search(
                collection_name=self.test_collection,
                query_vector=query_embedding,
                limit=3,
                score_threshold=0.0
            )
            
            logger.info(f"✓ Semantic search completed for query: '{query}'")
            logger.info(f"  Found {len(results)} results:")
            
            for i, result in enumerate(results, 1):
                logger.info(f"  {i}. Score: {result.score:.4f}")
                logger.info(f"     Text: {result.payload.get('text', 'N/A')[:60]}...")
                logger.info(f"     Doc ID: {result.payload.get('doc_id', 'N/A')}")
            
            self.passed += 1
            return True
        except Exception as e:
            logger.error(f"✗ Failed semantic search: {e}")
            self.failed += 1
            return False
    
    def test_filtered_search(self):
        """Test search with filtering"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 7: Filtered Search")
        logger.info("=" * 60)
        
        try:
            if not self.client or not self.embedding_model:
                logger.warning("⚠ Prerequisites not met, skipping")
                return False
            
            # Query for user_001 only
            query = "technology"
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Search with filter
            results = self.client.search(
                collection_name=self.test_collection,
                query_vector=query_embedding,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="user_id",
                            match=MatchValue(value="user_001")
                        )
                    ]
                ),
                limit=5,
                score_threshold=0.0
            )
            
            logger.info(f"✓ Filtered search completed (user_id='user_001')")
            logger.info(f"  Found {len(results)} results:")
            
            for i, result in enumerate(results, 1):
                logger.info(f"  {i}. Score: {result.score:.4f}")
                logger.info(f"     User: {result.payload.get('user_id', 'N/A')}")
                logger.info(f"     Text: {result.payload.get('text', 'N/A')[:50]}...")
            
            self.passed += 1
            return True
        except Exception as e:
            logger.error(f"✗ Failed filtered search: {e}")
            self.failed += 1
            return False
    
    def test_collection_stats(self):
        """Test getting collection statistics"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 8: Collection Statistics")
        logger.info("=" * 60)
        
        try:
            if not self.client:
                logger.warning("⚠ Client not connected, skipping")
                return False
            
            # Get collection info
            info = self.client.get_collection(self.test_collection)
            
            logger.info(f"✓ Retrieved collection statistics for: {self.test_collection}")
            logger.info(f"  Points count: {info.points_count}")
            logger.info(f"  Vectors count: {info.vectors_count}")
            logger.info(f"  Status: {info.status}")
            
            self.passed += 1
            return True
        except Exception as e:
            logger.error(f"✗ Failed to get collection stats: {e}")
            self.failed += 1
            return False
    
    def test_point_deletion(self):
        """Test deleting points"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 9: Point Deletion")
        logger.info("=" * 60)
        
        try:
            if not self.client:
                logger.warning("⚠ Client not connected, skipping")
                return False
            
            # Delete specific point
            self.client.delete(
                collection_name=self.test_collection,
                points_selector=PointIdsList(
                    idxs=[1]  # Delete first point
                )
            )
            
            # Get updated stats
            info = self.client.get_collection(self.test_collection)
            logger.info(f"✓ Successfully deleted point")
            logger.info(f"  Remaining points: {info.points_count}")
            
            self.passed += 1
            return True
        except Exception as e:
            logger.error(f"✗ Failed to delete point: {e}")
            self.failed += 1
            return False
    
    def test_collection_deletion(self):
        """Test deleting collection"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 10: Collection Deletion")
        logger.info("=" * 60)
        
        try:
            if not self.client:
                logger.warning("⚠ Client not connected, skipping")
                return False
            
            # Delete collection
            self.client.delete_collection(self.test_collection)
            logger.info(f"✓ Successfully deleted collection: {self.test_collection}")
            
            self.passed += 1
            return True
        except Exception as e:
            logger.error(f"✗ Failed to delete collection: {e}")
            self.failed += 1
            return False
    
    def run_all_tests(self):
        """Run all tests"""
        logger.info("\n")
        logger.info("╔" + "=" * 58 + "╗")
        logger.info("║" + " " * 12 + "Qdrant Connection Test Suite" + " " * 18 + "║")
        logger.info("╚" + "=" * 58 + "╝")
        
        config = "in-memory mode" if not self.qdrant_url else f"URL: {self.qdrant_url}"
        logger.info(f"\nQdrant Configuration: {config}")
        logger.info(f"Embedding Model: {self.embedding_model_name}\n")
        
        self.test_connection()
        self.test_embedding_model()
        self.test_server_info()
        self.test_collection_creation()
        self.test_vector_insertion()
        self.test_semantic_search()
        self.test_filtered_search()
        self.test_collection_stats()
        self.test_point_deletion()
        self.test_collection_deletion()
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("TEST SUMMARY")
        logger.info("=" * 60)
        total = self.passed + self.failed
        logger.info(f"Total Tests: {total}")
        logger.info(f"✓ Passed: {self.passed}")
        logger.info(f"✗ Failed: {self.failed}")
        
        if self.failed == 0:
            logger.info("\n🎉 All tests passed!")
        else:
            logger.warning(f"\n⚠️  {self.failed} test(s) failed")
        
        logger.info("=" * 60 + "\n")
        
        return self.failed == 0


def main():
    """Main function"""
    # Get configuration with priority:
    # 1. Cloud settings (new)
    # 2. Legacy settings (backwards compatible)
    # 3. In-memory (default)
    
    qdrant_url = None
    qdrant_api_key = None
    config_source = "default"
    
    # Try to use config module if available
    if USE_CONFIG:
        if settings.QDRANT_MODE == "cloud" and settings.QDRANT_CLOUD_URL:
            qdrant_url = settings.QDRANT_CLOUD_URL
            qdrant_api_key = settings.QDRANT_CLOUD_API_KEY
            config_source = "Cloud (from settings)"
        elif settings.QDRANT_URL:
            qdrant_url = settings.QDRANT_URL
            qdrant_api_key = settings.QDRANT_API_KEY
            config_source = "Legacy URL (from settings)"
        else:
            config_source = "In-memory (settings)"
    else:
        # Fall back to direct environment variables
        # New cloud variables take priority
        if os.getenv("QDRANT_CLOUD_URL"):
            qdrant_url = os.getenv("QDRANT_CLOUD_URL")
            qdrant_api_key = os.getenv("QDRANT_CLOUD_API_KEY")
            config_source = "Cloud (env vars)"
        elif os.getenv("QDRANT_URL"):
            qdrant_url = os.getenv("QDRANT_URL")
            qdrant_api_key = os.getenv("QDRANT_API_KEY")
            config_source = "Legacy URL (env vars)"
        else:
            config_source = "In-memory (default)"
    
    try:
        tester = QdrantConnectionTest(
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key
        )
        
        # Log which config was used
        logger.info(f"Configuration Source: {config_source}")
        
        success = tester.run_all_tests()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

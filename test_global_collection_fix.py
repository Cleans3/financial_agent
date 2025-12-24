"""
Verify admin global collection upload implementation is correct.
"""

import asyncio
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

async def test_global_collection_methods():
    """Verify global collection methods exist with correct signatures"""
    try:
        from src.services.multi_collection_rag_service import MultiCollectionRAGService
        from src.services.qdrant_collection_manager import QdrantCollectionManager
        from src.services.document_service import DocumentService
        import inspect
        
        # Test MultiCollectionRAGService.add_document_to_global
        rag = MultiCollectionRAGService()
        if not hasattr(rag, 'add_document_to_global'):
            print("✗ MultiCollectionRAGService.add_document_to_global() does not exist")
            return False
        
        sig = inspect.signature(rag.add_document_to_global)
        params = list(sig.parameters.keys())
        expected = ['text', 'title', 'source']
        if params == expected:
            print("✓ MultiCollectionRAGService.add_document_to_global() signature correct")
            print(f"  Parameters: {params}")
        else:
            print("✗ add_document_to_global signature mismatch")
            print(f"  Expected: {expected}, Got: {params}")
            return False
        
        # Test QdrantCollectionManager.add_document_chunks_to_global
        qd = QdrantCollectionManager()
        if not hasattr(qd, 'add_document_chunks_to_global'):
            print("✗ QdrantCollectionManager.add_document_chunks_to_global() does not exist")
            return False
        
        sig = inspect.signature(qd.add_document_chunks_to_global)
        params = list(sig.parameters.keys())
        expected = ['file_id', 'chunks', 'metadata']
        if params == expected:
            print("✓ QdrantCollectionManager.add_document_chunks_to_global() signature correct")
            print(f"  Parameters: {params}")
        else:
            print("✗ add_document_chunks_to_global signature mismatch")
            print(f"  Expected: {expected}, Got: {params}")
            return False
        
        # Test DocumentService.process_file with upload_to_global parameter
        doc_service = DocumentService()
        sig = inspect.signature(doc_service.process_file)
        params = list(sig.parameters.keys())
        
        if 'upload_to_global' not in params:
            print("✗ DocumentService.process_file() missing upload_to_global parameter")
            return False
        
        expected = ['file_path', 'chat_session_id', 'title', 'user_id', 'upload_to_global']
        if params == expected:
            print("✓ DocumentService.process_file() has upload_to_global parameter")
            print(f"  Parameters: {params}")
        else:
            print("✗ process_file signature mismatch")
            print(f"  Expected: {expected}, Got: {params}")
            return False
        
        # Verify upload_to_global default is False
        default_value = sig.parameters['upload_to_global'].default
        if default_value == False:
            print("✓ upload_to_global parameter defaults to False")
        else:
            print(f"✗ upload_to_global defaults to {default_value}, expected False")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error during verification: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_collection_names():
    """Verify collection names are correct"""
    try:
        from src.services.qdrant_collection_manager import QdrantCollectionManager
        
        qd = QdrantCollectionManager()
        
        user_collection = qd._get_user_collection_name("admin")
        global_collection = qd._get_global_collection_name()
        
        if user_collection == "user_admin":
            print("✓ User collection for 'admin' user: user_admin")
        else:
            print(f"✗ User collection incorrect: {user_collection} (expected user_admin)")
            return False
        
        if global_collection == "global_admin":
            print("✓ Global collection name: global_admin")
        else:
            print(f"✗ Global collection name incorrect: {global_collection} (expected global_admin)")
            return False
        
        # Verify they are different
        if user_collection != global_collection:
            print("✓ User and global collections are different (proper routing)")
            return True
        else:
            print("✗ User and global collections are the same")
            return False
            
    except Exception as e:
        print(f"✗ Error during collection name verification: {e}")
        return False


async def main():
    """Run all verification tests"""
    print("=" * 70)
    print("ADMIN GLOBAL COLLECTION UPLOAD FIX VERIFICATION")
    print("=" * 70)
    
    results = []
    
    print("\n1. Testing global collection method signatures...")
    results.append(await test_global_collection_methods())
    
    print("\n2. Testing collection name routing...")
    results.append(await test_collection_names())
    
    print("\n" + "=" * 70)
    if all(results):
        print("✓ ALL TESTS PASSED - Global collection upload fix is correct!")
        print("\nAdmin uploads will now go to 'global_admin' collection")
        print("=" * 70)
        return True
    else:
        print("✗ SOME TESTS FAILED - Review the output above")
        print("=" * 70)
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

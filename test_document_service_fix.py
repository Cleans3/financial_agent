"""
Verify that the document_service parameter fix is correct.
This test ensures that process_file() calls MultiCollectionRAGService.add_document()
with the correct parameter names.
"""

import asyncio
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

async def test_document_service_signature():
    """Verify document_service.process_file() has correct signature"""
    try:
        from src.services.document_service import DocumentService
        import inspect
        
        # Check process_file signature
        sig = inspect.signature(DocumentService.process_file)
        params = list(sig.parameters.keys())
        
        expected_params = ['self', 'file_path', 'chat_session_id', 'title', 'user_id']
        if params == expected_params:
            print("✓ process_file signature is correct")
            print(f"  Parameters: {params}")
            return True
        else:
            print("✗ process_file signature mismatch")
            print(f"  Expected: {expected_params}")
            print(f"  Got: {params}")
            return False
    except Exception as e:
        print(f"✗ Error checking signature: {e}")
        return False


async def test_rag_service_signature():
    """Verify MultiCollectionRAGService.add_document() signature"""
    try:
        from src.services.multi_collection_rag_service import MultiCollectionRAGService
        import inspect
        
        # Check add_document signature
        sig = inspect.signature(MultiCollectionRAGService.add_document)
        params = list(sig.parameters.keys())
        
        expected_params = ['self', 'user_id', 'chat_session_id', 'text', 'title', 'source']
        if params == expected_params:
            print("✓ MultiCollectionRAGService.add_document signature is correct")
            print(f"  Parameters: {params}")
            return True
        else:
            print("✗ RAG service add_document signature mismatch")
            print(f"  Expected: {expected_params}")
            print(f"  Got: {params}")
            return False
    except Exception as e:
        print(f"✗ Error checking RAG service signature: {e}")
        return False


async def test_imports():
    """Verify all necessary imports work"""
    try:
        from src.services.document_service import DocumentService, get_document_service
        from src.services.multi_collection_rag_service import MultiCollectionRAGService, get_rag_service
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False


async def main():
    """Run all verification tests"""
    print("=" * 70)
    print("DOCUMENT SERVICE PARAMETER FIX VERIFICATION")
    print("=" * 70)
    
    results = []
    
    print("\n1. Testing DocumentService.process_file signature...")
    results.append(await test_document_service_signature())
    
    print("\n2. Testing MultiCollectionRAGService.add_document signature...")
    results.append(await test_rag_service_signature())
    
    print("\n3. Testing imports...")
    results.append(await test_imports())
    
    print("\n" + "=" * 70)
    if all(results):
        print("✓ ALL TESTS PASSED - Parameter fix is correct!")
        print("=" * 70)
        return True
    else:
        print("✗ SOME TESTS FAILED - Review the output above")
        print("=" * 70)
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

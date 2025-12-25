"""Quick test for file upload workflow"""
import asyncio
import aiohttp
import json
from pathlib import Path

async def test_file_upload():
    """Test file upload and query workflow"""
    
    # Create a test file
    test_file_path = Path("test_doc.txt")
    test_file_path.write_text("FPT Corporation Q3 2024 Results: Revenue increased 15% YoY")
    
    async with aiohttp.ClientSession() as session:
        # Test 1: Upload file without query
        print("=" * 50)
        print("TEST 1: Upload file with query")
        print("=" * 50)
        
        with open(test_file_path, 'rb') as f:
            data = aiohttp.FormData()
            data.add_field('file', f, filename='test_doc.txt')
            data.add_field('query', 'What was FPT revenue in Q3 2024?')
            data.add_field('user_id', 'test_user_1')
            data.add_field('session_id', 'test_session_1')
            
            async with session.post(
                'http://localhost:8000/api/chat',
                data=data
            ) as resp:
                result = await resp.json()
                print(f"Status: {resp.status}")
                print(f"Response: {json.dumps(result, indent=2, ensure_ascii=False)[:500]}")
        
        # Test 2: Query same file (should find in vectordb now)
        print("\n" + "=" * 50)
        print("TEST 2: Query same file (should find in vectordb)")
        print("=" * 50)
        
        query_data = {
            "question": "What about FPT's revenue growth?",
            "user_id": "test_user_1",
            "session_id": "test_session_1"
        }
        
        async with session.post(
            'http://localhost:8000/api/chat',
            json=query_data
        ) as resp:
            result = await resp.json()
            print(f"Status: {resp.status}")
            print(f"Response: {json.dumps(result, indent=2, ensure_ascii=False)[:500]}")
    
    # Cleanup
    test_file_path.unlink()
    print("\n✅ Test completed")

if __name__ == "__main__":
    asyncio.run(test_file_upload())

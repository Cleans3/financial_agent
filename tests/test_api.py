"""
Test API - Kiểm tra REST API endpoints
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import requests
import json
import time


API_BASE_URL = "http://localhost:8000"


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)


def wait_for_api(max_retries=5, delay=2):
    """Wait for API to be ready"""
    print("\n⏳ Waiting for API to be ready...")
    for i in range(max_retries):
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                print(f"✅ API is ready!")
                return True
        except requests.exceptions.ConnectionError:
            if i < max_retries - 1:
                print(f"   Attempt {i+1}/{max_retries} failed, retrying in {delay}s...")
                time.sleep(delay)
    return False


def test_root():
    """Test root endpoint"""
    print_header("TEST 1: Root Endpoint (GET /)")
    
    try:
        response = requests.get(f"{API_BASE_URL}/")
        print(f"Status: {response.status_code}")
        print(f"Response:\n{json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        assert response.status_code == 200
        print("✅ Test passed")
    except Exception as e:
        print(f"❌ Test failed: {e}")


def test_health():
    """Test health check endpoint"""
    print_header("TEST 2: Health Check (GET /health)")
    
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response:\n{json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        print("✅ Test passed")
    except Exception as e:
        print(f"❌ Test failed: {e}")


def test_chat():
    """Test chat endpoint with various questions"""
    print_header("TEST 3: Chat Endpoint (POST /api/chat)")
    
    questions = [
        "Cho tôi biết thông tin về công ty VNM?",
        "Giá VCB trong 3 tháng gần nhất?",
        "Tính SMA 20 ngày cho HPG",
        "RSI của VIC hiện tại?",
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n--- Question {i} ---")
        print(f"Q: {question}")
        
        try:
            payload = {"question": question}
            response = requests.post(
                f"{API_BASE_URL}/api/chat",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60  # 60s timeout for LLM processing
            )
            
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                answer = result['answer']
                print(f"A: {answer[:300]}...")  # First 300 chars
                print(f"   (Full length: {len(answer)} characters)")
                print("✅ Test passed")
            else:
                print(f"❌ Error: {response.text}")
                
        except requests.exceptions.Timeout:
            print("⏱️  Timeout - Request took too long")
        except Exception as e:
            print(f"❌ Test failed: {e}")


def test_invalid_input():
    """Test API with invalid inputs"""
    print_header("TEST 4: Invalid Input Handling")
    
    # Test empty question
    print("\n--- Test với câu hỏi rỗng ---")
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/chat",
            json={"question": ""},
            headers={"Content-Type": "application/json"}
        )
        print(f"Status: {response.status_code}")
        if response.status_code == 422:
            print("✅ Correctly rejected empty question (422)")
        else:
            print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test missing field
    print("\n--- Test với missing field ---")
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/chat",
            json={},
            headers={"Content-Type": "application/json"}
        )
        print(f"Status: {response.status_code}")
        if response.status_code == 422:
            print("✅ Correctly rejected missing field (422)")
        else:
            print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║          Testing Financial Agent API                         ║
║          ⚠️  Make sure the API is running first!            ║
║          Run: python main.py                                 ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    try:
        # Wait for API to be ready
        if not wait_for_api():
            print("""
❌ ERROR: Cannot connect to API

Please make sure the API is running:
1. Open a new terminal
2. cd E:\\My_Project\\AI\\ChatBot\\financial_agent
3. python -m venv venv
4. venv\\Scripts\\activate
5. pip install -r requirements.txt
6. python main.py

Then run this test again.
""")
            exit(1)
        
        # Run tests
        test_root()
        test_health()
        test_chat()
        test_invalid_input()
        
        print("\n" + "="*70)
        print("  ✅ ALL API TESTS COMPLETED")
        print("="*70 + "\n")
        
    except requests.exceptions.ConnectionError:
        print("""
❌ ERROR: Cannot connect to API at http://localhost:8000

Please make sure the API is running: python main.py
""")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

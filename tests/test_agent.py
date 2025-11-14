"""
Test Financial Agent - Kiểm tra Agent với các câu hỏi
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agent import FinancialAgent


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)


def test_agent():
    """Test agent with various questions"""
    print_header("Initializing Financial Agent")
    
    # Initialize agent
    print("\n🤖 Creating agent instance...")
    agent = FinancialAgent()
    print("✅ Agent initialized successfully!")
    print(f"📊 Loaded {len(agent.tools)} tools: {[t.name for t in agent.tools]}")
    
    # Test questions
    questions = [
        {
            "question": "Cho tôi biết thông tin về công ty VNM?",
            "expected": "company info, Vinamilk"
        },
        {
            "question": "Giá đóng cửa của VCB trong 3 tháng gần nhất?",
            "expected": "historical data, VCB, 3 months"
        },
        {
            "question": "Tính SMA 20 ngày cho mã HPG từ 2023-01-01 đến 2023-06-30",
            "expected": "SMA-20, HPG, trend analysis"
        },
        {
            "question": "RSI của VIC hiện tại có quá mua không?",
            "expected": "RSI-14, VIC, overbought/oversold"
        },
    ]
    
    for i, test_case in enumerate(questions, 1):
        question = test_case["question"]
        expected = test_case["expected"]
        
        print_header(f"TEST {i}: {question}")
        print(f"Expected to include: {expected}\n")
        
        try:
            answer = agent.query(question)
            print(f"📝 ANSWER:\n{answer}\n")
            print(f"✅ Test {i} completed")
        except Exception as e:
            print(f"❌ ERROR in Test {i}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("  ✅ ALL AGENT TESTS COMPLETED")
    print("="*70 + "\n")


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║          Testing Financial Agent                             ║
║          LangGraph + ReAct Pattern                           ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    test_agent()

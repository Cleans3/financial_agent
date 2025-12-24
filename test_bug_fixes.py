"""
Test script to verify bug fixes work correctly
"""

import asyncio
from pathlib import Path


async def test_model_columns():
    """Verify ChatSession has session_metadata column"""
    print("\n" + "="*60)
    print("Testing Database Model Changes")
    print("="*60)
    
    try:
        from src.database.models import ChatSession
        
        columns = [c.name for c in ChatSession.__table__.columns]
        print(f"\n✓ ChatSession columns: {columns}")
        
        assert "session_metadata" in columns, "session_metadata column not found"
        print("✓ session_metadata column found")
        
        assert "metadata" not in columns, "metadata should not exist (reserved)"
        print("✓ metadata column properly replaced")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


async def test_api_syntax():
    """Verify API syntax is valid"""
    print("\n" + "="*60)
    print("Testing API Syntax")
    print("="*60)
    
    try:
        import ast
        with open('src/api/app.py', 'r', encoding='utf-8') as f:
            ast.parse(f.read())
        print("✓ API syntax valid")
        return True
    except SyntaxError as e:
        print(f"✗ Syntax error: {e}")
        return False


async def test_workflow_syntax():
    """Verify workflow syntax is valid"""
    print("\n" + "="*60)
    print("Testing Workflow Syntax")
    print("="*60)
    
    try:
        import ast
        with open('src/core/langgraph_workflow.py', 'r', encoding='utf-8') as f:
            ast.parse(f.read())
        print("✓ Workflow syntax valid")
        return True
    except SyntaxError as e:
        print(f"✗ Syntax error: {e}")
        return False


async def test_tool_formatter():
    """Verify tool formatter can be imported"""
    print("\n" + "="*60)
    print("Testing Tool Result Formatter")
    print("="*60)
    
    try:
        from src.utils.tool_result_formatter import format_tool_result_as_table
        print("✓ Tool formatter imports successfully")
        
        # Test with sample SMA result
        sample_result = '''{
            "success": true,
            "ticker": "HPG",
            "analysis": {
                "trend": "TĂNG - Giá đang trên SMA",
                "interpretation": "Giá cao hơn SMA-20"
            },
            "current_values": {
                "price": 123.45,
                "sma": 120.00,
                "difference": 3.45,
                "difference_percent": 2.88
            },
            "detailed_data": [
                {
                    "date": "2024-12-23",
                    "close": 123.45,
                    "sma_20": 120.10,
                    "difference": 3.35,
                    "difference_percent": 2.79
                },
                {
                    "date": "2024-12-20",
                    "close": 122.80,
                    "sma_20": 119.95,
                    "difference": 2.85,
                    "difference_percent": 2.37
                }
            ],
            "message": "Đã tính SMA-20 cho HPG"
        }'''
        
        formatted = format_tool_result_as_table(sample_result)
        print("\n✓ Sample SMA result formatted as table:")
        print(formatted[:200] + "...")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_imports():
    """Verify all modules import successfully"""
    print("\n" + "="*60)
    print("Testing Module Imports")
    print("="*60)
    
    try:
        print("Importing models...", end=" ")
        from src.database.models import ChatSession
        print("✓")
        
        print("Importing workflow...", end=" ")
        from src.core.langgraph_workflow import LangGraphWorkflow
        print("✓")
        
        print("Importing tool formatter...", end=" ")
        from src.utils.tool_result_formatter import format_tool_result_as_table
        print("✓")
        
        print("Importing agent...", end=" ")
        from src.agent.financial_agent import FinancialAgent
        print("✓")
        
        return True
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("BUG FIX VERIFICATION TESTS")
    print("="*80)
    
    results = []
    results.append(("Model Columns", await test_model_columns()))
    results.append(("API Syntax", await test_api_syntax()))
    results.append(("Workflow Syntax", await test_workflow_syntax()))
    results.append(("Tool Formatter", await test_tool_formatter()))
    results.append(("Module Imports", await test_imports()))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(result for _, result in results)
    if all_passed:
        print("\n✅ All bug fixes verified!")
    else:
        print("\n❌ Some tests failed")
    
    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)

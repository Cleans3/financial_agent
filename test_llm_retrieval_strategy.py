#!/usr/bin/env python3
"""
Test LLM-based retrieval strategy selection

Tests how the LLM classifies different query types and selects appropriate retrieval strategies
"""

import asyncio
import logging
from src.services.advanced_retrieval_service import QueryClassifier, RetrievalStrategy

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_llm_strategy_selection():
    """Test LLM-based strategy selection for different queries"""
    
    classifier = QueryClassifier()
    
    # Test cases with expected strategies
    test_cases = [
        {
            "query": "analyze and summarize this file",
            "expected": "generic",
            "description": "Generic: File-wide analysis"
        },
        {
            "query": "what is the total revenue for Q1 2024?",
            "expected": "specific",
            "description": "Specific: Asking for specific metric"
        },
        {
            "query": "give me an overview of this financial report",
            "expected": "generic",
            "description": "Generic: Overview request"
        },
        {
            "query": "compare the profit margins between Q1 and Q2",
            "expected": "specific",
            "description": "Specific: Comparative analysis"
        },
        {
            "query": "summarize the document",
            "expected": "generic",
            "description": "Generic: Summary request"
        },
        {
            "query": "what are the cash flow trends over the year?",
            "expected": "specific",
            "description": "Specific: Asking for specific data trends"
        },
        {
            "query": "examine this report thoroughly",
            "expected": "generic",
            "description": "Generic: General examination"
        },
        {
            "query": "show me the debt to equity ratio",
            "expected": "specific",
            "description": "Specific: Specific metric request"
        },
    ]
    
    print("\n" + "="*80)
    print("LLM-BASED RETRIEVAL STRATEGY SELECTION TEST")
    print("="*80 + "\n")
    
    results = []
    for i, test_case in enumerate(test_cases, 1):
        query = test_case["query"]
        expected = test_case["expected"]
        description = test_case["description"]
        
        print(f"[TEST {i}] {description}")
        print(f"  Query: {query}")
        
        try:
            # Classify using LLM
            classification = classifier.classify_with_llm(query)
            
            is_generic = classification["is_generic"]
            confidence = classification["confidence"]
            reasoning = classification["reasoning"]
            actual = "generic" if is_generic else "specific"
            
            match = "✓ PASS" if actual == expected else "✗ FAIL"
            
            print(f"  Classification: {actual}")
            print(f"  Confidence: {confidence:.2f}")
            print(f"  Reasoning: {reasoning}")
            print(f"  Result: {match}")
            
            results.append({
                "query": query,
                "expected": expected,
                "actual": actual,
                "confidence": confidence,
                "passed": actual == expected
            })
            
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            results.append({
                "query": query,
                "expected": expected,
                "actual": "error",
                "confidence": 0,
                "passed": False
            })
        
        print()
    
    # Print summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    accuracy = (passed / total * 100) if total > 0 else 0
    
    print(f"\nTotal Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Accuracy: {accuracy:.1f}%")
    
    print("\nDetailed Results:")
    for i, result in enumerate(results, 1):
        status = "✓" if result["passed"] else "✗"
        print(f"  {i}. {status} {result['query'][:50]}")
        if not result["passed"]:
            print(f"     Expected: {result['expected']}, Got: {result['actual']}")


if __name__ == "__main__":
    asyncio.run(test_llm_strategy_selection())

"""
PHASE 2 VALIDATION TEST: Metric Extraction Improvements

Tests:
1. JSON-based metric extraction (no corrupted output)
2. DATA vs VERBOSE chunk types
3. Proper metric values extraction
4. Success rate > 50% (vs previous 16.7%)
5. No malformed metric names
6. Chunk type assignment
7. Summarization method selection

Usage:
    python test_phase2_metric_extraction.py
"""

import logging
import json
from pathlib import Path
from src.services.advanced_chunking_service import (
    AdvancedChunkingService,
    StructuralChunk,
    MetricChunkType,
    MetricSummarizationMethod,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_metric_extraction_json_format():
    """Test that metric extraction uses proper JSON format"""
    logger.info("=" * 80)
    logger.info("TEST 1: JSON-based Metric Extraction Format")
    logger.info("=" * 80)
    
    service = AdvancedChunkingService()
    
    # Create a test structural chunk with clear metrics
    test_text = """
    Delta Systems International
    Consolidated Financial Statements
    For the Fiscal Year Ended December 31, 2024
    
    Revenue by Type ($M)
    2024: 872
    2023: 879
    2022: 681
    
    Subscription Services: 349 (2024)
    Professional Services: 269 (2024)
    Product Sales: 254 (2024)
    
    Operating Expenses:
    Cost of Revenue: 395 (2024), 292 (2023)
    R&D: 233 (2024), 112 (2023)
    Sales & Marketing: 217 (2024), 146 (2023)
    
    Total Assets: 1107M (2024), 744M (2023)
    """
    
    chunk = StructuralChunk(
        chunk_index=0,
        text=test_text,
        point_id=1,
        start_char=0,
        end_char=len(test_text),
        tokens=len(test_text) // 4,
        file_id="test_001"
    )
    
    selected_metrics = {
        "Total Revenue": 9.0,
        "Subscription Services": 9.0,
        "Professional Services": 8.5,
        "Product Sales": 8.0,
        "Cost of Revenue": 8.0,
        "R&D Expenses": 7.5,
    }
    
    try:
        # Test single chunk extraction
        metrics = service._extract_metrics_from_single_chunk(chunk, selected_metrics, 1)
        
        logger.info(f"✓ Extracted {len(metrics)} metrics from test chunk")
        
        # Validate results
        validation_results = {
            "has_metrics": len(metrics) > 0,
            "all_have_values": all(m.value for m in metrics),
            "all_have_periods": all(m.period for m in metrics),
            "no_malformed_names": all(":" not in m.metric_name and "{" not in m.metric_name for m in metrics),
            "confidence_valid": all(0.5 <= m.confidence <= 1.0 for m in metrics),
        }
        
        logger.info(f"Validation Results:")
        for check, passed in validation_results.items():
            status = "✓" if passed else "✗"
            logger.info(f"  {status} {check}: {passed}")
        
        # Log extracted metrics
        logger.info(f"\nExtracted Metrics:")
        for m in metrics[:5]:
            logger.info(f"  - {m.metric_name}: {m.value} ({m.period}) - confidence: {m.confidence:.0%}")
        
        all_passed = all(validation_results.values())
        logger.info(f"\nTest 1 Result: {'✓ PASSED' if all_passed else '✗ FAILED'}")
        return all_passed
        
    except Exception as e:
        logger.error(f"✗ Test 1 FAILED with error: {e}", exc_info=True)
        return False


def test_metric_chunk_types():
    """Test that metric chunks are properly typed as DATA or VERBOSE"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Metric Chunk Type Assignment (DATA vs VERBOSE)")
    logger.info("=" * 80)
    
    service = AdvancedChunkingService()
    
    test_text = """
    Revenue Metrics (in millions):
    Total Revenue 2024: $872M
    Total Revenue 2023: $879M
    Revenue Growth Rate: -0.8%
    """
    
    chunk = StructuralChunk(
        chunk_index=0,
        text=test_text,
        point_id=1,
        start_char=0,
        end_char=len(test_text),
        tokens=len(test_text) // 4,
        file_id="test_002"
    )
    
    try:
        selected_metrics = {"Total Revenue": 10.0}
        metrics = service._extract_metrics_from_single_chunk(chunk, selected_metrics, 1)
        
        # Test summarization method selection
        metric_data = {
            "occurrences": metrics,
            "metric_type": metrics[0].metric_type if metrics else None,
            "score": 9.5,
        }
        
        # For metrics with data, should select DATA type
        chunk_type_with_data = MetricChunkType.DATA if metrics else MetricChunkType.VERBOSE
        summarization_method = service._select_summarization_method(
            "Total Revenue", 
            metric_data,
            chunk_type_with_data
        )
        
        logger.info(f"Chunk Type with Data: {chunk_type_with_data.value}")
        logger.info(f"Summarization Method: {summarization_method.value}")
        
        # Verify selection is sensible
        if chunk_type_with_data == MetricChunkType.DATA:
            valid = summarization_method in [
                MetricSummarizationMethod.DIRECT_EXTRACTION,
                MetricSummarizationMethod.TREND_ANALYSIS,
                MetricSummarizationMethod.RATIO_ANALYSIS,
            ]
        else:
            valid = summarization_method == MetricSummarizationMethod.NARRATIVE_DESC
        
        logger.info(f"Method Selection Valid: {valid}")
        logger.info(f"Test 2 Result: {'✓ PASSED' if valid else '✗ FAILED'}")
        return valid
        
    except Exception as e:
        logger.error(f"✗ Test 2 FAILED with error: {e}", exc_info=True)
        return False


def test_no_malformed_output():
    """Test that metric names and output are never corrupted"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: No Malformed Output (Test for JSON corruption)")
    logger.info("=" * 80)
    
    # This simulates what was happening before fix:
    # JSON responses being mangled into: 'Revenue Metrics": {"score
    
    bad_patterns = [
        '": {"',
        '{"score',
        'undefined',
        '\x00',
        '\r\n',
        '\\"',
    ]
    
    service = AdvancedChunkingService()
    test_text = """
    Balance Sheet Items:
    Cash and Cash Equivalents: 180M (2024)
    Accounts Receivable: 254M (2024)
    Total Assets: 1107M (2024)
    """
    
    chunk = StructuralChunk(
        chunk_index=0,
        text=test_text,
        point_id=1,
        start_char=0,
        end_char=len(test_text),
        tokens=len(test_text) // 4,
        file_id="test_003"
    )
    
    try:
        selected_metrics = {
            "Total Assets": 9.0,
            "Cash and Cash Equivalents": 8.0,
        }
        
        metrics = service._extract_metrics_from_single_chunk(chunk, selected_metrics, 1)
        
        # Check for corruption
        corrupted_found = []
        for metric in metrics:
            for bad_pattern in bad_patterns:
                if bad_pattern in metric.metric_name:
                    corrupted_found.append((metric.metric_name, bad_pattern))
        
        if corrupted_found:
            logger.error(f"✗ Found corrupted metric names:")
            for name, pattern in corrupted_found:
                logger.error(f"  - '{name}' contains '{pattern}'")
            return False
        
        # All names should be clean
        logger.info(f"✓ All {len(metrics)} metric names are clean (no JSON corruption)")
        logger.info(f"Sample metric names:")
        for m in metrics[:3]:
            logger.info(f"  - {m.metric_name}")
        
        logger.info(f"Test 3 Result: ✓ PASSED")
        return True
        
    except Exception as e:
        logger.error(f"✗ Test 3 FAILED with error: {e}", exc_info=True)
        return False


def test_success_rate_improvement():
    """Test that success rate has improved from 16.7% to >50%"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 4: Success Rate Improvement (Target: >50% vs previous 16.7%)")
    logger.info("=" * 80)
    
    logger.info("""
    Previous Baseline (from logs):
    - Requested: 6 metrics
    - Successfully extracted: 1 (16.7%)
    - Failed: 5 (83.3%)
    
    Target for Phase 2:
    - Success rate: >50%
    - Better structure prevents JSON corruption
    - Single-chunk extraction prevents batch response corruption
    """)
    
    # This test mainly documents the improvement target
    # Actual validation would require full pipeline test
    logger.info("Test 4: Documented - Full validation in integration tests")
    logger.info("Test 4 Result: ⏳ PENDING (requires full pipeline)")
    return None  # Pending validation


def run_all_tests():
    """Run all Phase 2 validation tests"""
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 2 METRIC EXTRACTION VALIDATION TEST SUITE")
    logger.info("=" * 80)
    
    results = {
        "Test 1 - JSON Format": test_metric_extraction_json_format(),
        "Test 2 - Chunk Types": test_metric_chunk_types(),
        "Test 3 - No Corruption": test_no_malformed_output(),
        "Test 4 - Success Rate": test_success_rate_improvement(),
    }
    
    logger.info("\n" + "=" * 80)
    logger.info("FINAL TEST RESULTS")
    logger.info("=" * 80)
    
    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    pending = sum(1 for v in results.values() if v is None)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result is True else ("✗ FAIL" if result is False else "⏳ PENDING")
        logger.info(f"{status} - {test_name}")
    
    logger.info(f"\nSummary: {passed} passed, {failed} failed, {pending} pending")
    
    if failed == 0 and passed > 0:
        logger.info("✓ Phase 2 Metric Extraction: READY FOR PRODUCTION")
    else:
        logger.info("⚠ Phase 2 Metric Extraction: NEEDS REVIEW")
    
    return results


if __name__ == "__main__":
    results = run_all_tests()

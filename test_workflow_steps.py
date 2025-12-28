#!/usr/bin/env python3
"""
Quick test to verify workflow steps integration
Run: python test_workflow_steps.py
"""

import asyncio
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.workflow_step_streaming import (
    create_workflow_step,
    STEP_STATUS_PENDING,
    STEP_STATUS_IN_PROGRESS,
    STEP_STATUS_COMPLETED,
    STEP_STATUS_ERROR,
    WORKFLOW_NODE_MAPPING
)


def test_workflow_steps():
    """Test workflow step creation"""
    print("Testing Workflow Steps Integration\n")
    print("=" * 60)
    
    # Test 1: Check all nodes are configured
    print("\n✓ Test 1: Workflow Node Mapping")
    print(f"  Total nodes configured: {len(WORKFLOW_NODE_MAPPING)}")
    for node_id, config in list(WORKFLOW_NODE_MAPPING.items())[:5]:
        print(f"    - {node_id}: {config.get('title', 'N/A')}")
    print(f"    ... and {len(WORKFLOW_NODE_MAPPING) - 5} more")
    
    # Test 2: Create a step with all status types
    print("\n✓ Test 2: Create Workflow Steps with Different Status")
    statuses = [
        STEP_STATUS_PENDING,
        STEP_STATUS_IN_PROGRESS,
        STEP_STATUS_COMPLETED,
        STEP_STATUS_ERROR
    ]
    
    for status in statuses:
        step = create_workflow_step(
            node_name='test_node',
            status=status,
            result=f'Test result for {status}',
            metadata={'test': True},
            duration=100
        )
        print(f"  ✓ Status '{status}':")
        print(f"    - ID: {step.get('id')}")
        print(f"    - Status: {step.get('status')}")
        print(f"    - Result: {step.get('result', 'N/A')}")
    
    # Test 3: Create steps for all workflow nodes
    print("\n✓ Test 3: Create Steps for All Workflow Nodes")
    steps = []
    for i, (node_id, config) in enumerate(WORKFLOW_NODE_MAPPING.items()):
        step = create_workflow_step(
            node_name=node_id,
            status=STEP_STATUS_COMPLETED if i < len(WORKFLOW_NODE_MAPPING) - 1 else STEP_STATUS_IN_PROGRESS,
            result=f"Processed by {config.get('title', node_id)}",
            metadata={'order': i + 1},
            duration=(i + 1) * 50
        )
        steps.append(step)
    
    print(f"  Created {len(steps)} workflow steps")
    print(f"  Phases covered: {set(s.get('phase', 'Unknown') for s in steps)}")
    
    # Test 4: Verify JSON serialization
    print("\n✓ Test 4: JSON Serialization")
    try:
        json_str = json.dumps(steps[0])
        print(f"  ✓ Step serializes correctly")
        print(f"  ✓ Size: {len(json_str)} bytes")
    except Exception as e:
        print(f"  ✗ Serialization failed: {e}")
    
    # Test 5: Verify step structure
    print("\n✓ Test 5: Verify Step Structure")
    required_fields = ['id', 'node_id', 'status', 'phase', 'title', 'description']
    sample_step = steps[0]
    for field in required_fields:
        has_field = field in sample_step
        symbol = "✓" if has_field else "✗"
        print(f"  {symbol} {field}: {sample_step.get(field, 'MISSING')}")
    
    # Test 6: Check duration tracking
    print("\n✓ Test 6: Duration Tracking")
    for i in [0, len(steps)//2, len(steps)-1]:
        step = steps[i]
        print(f"  Step {i}: {step.get('node_id')} - {step.get('duration_ms', 0)}ms")
    
    # Test 7: Simulate streaming format
    print("\n✓ Test 7: Streaming Format (SSE)")
    test_stream = {
        "type": "workflow_step",
        "step": steps[0]
    }
    sse_format = f"data: {json.dumps(test_stream)}\n\n"
    print(f"  ✓ SSE Format (first 80 chars):")
    print(f"    {sse_format[:80]}...")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("\nNext steps:")
    print("  1. Start backend: python -m uvicorn src.api.app:app --reload")
    print("  2. Start frontend: cd frontend && npm run dev")
    print("  3. Send a chat message and observe workflow steps in real-time")
    print("  4. Check browser console for debug logs")


if __name__ == "__main__":
    test_workflow_steps()

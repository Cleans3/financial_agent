"""
Integration tests for V4 workflow - Phase 3 (Complete 13-node architecture)
Tests output formatting, workflow observation, and full end-to-end flow
"""

import pytest
import asyncio
from typing import Dict, Any, List

from src.core.workflow_state import (
    create_initial_state, PromptType, DataType
)
from src.core.output_formatter import OutputFormatter
from src.core.workflow_observer import WorkflowObserver, StepStatus
from src.core.langgraph_workflow_v4 import LangGraphWorkflowV4
from src.agent.financial_agent import FinancialAgent


@pytest.fixture
async def agent():
    """Create agent with V4 workflow"""
    agent = FinancialAgent()
    yield agent


@pytest.fixture
def output_formatter():
    """Create OutputFormatter instance"""
    return OutputFormatter(use_markdown=True)


@pytest.fixture
def workflow_observer():
    """Create WorkflowObserver instance"""
    return WorkflowObserver()


class TestOutputFormatter:
    """Tests for OutputFormatter module"""
    
    @pytest.mark.asyncio
    async def test_format_simple_answer(self, output_formatter):
        """Test: Format simple text answer"""
        answer = "FPT Corporation's revenue grew by 15% in Q3 2024"
        
        formatted = await output_formatter.format_answer(
            answer,
            search_results=[],
            tool_results=None,
            detected_data_types=[]
        )
        
        assert formatted is not None
        assert len(formatted) > 0
        assert "FPT" in formatted
    
    @pytest.mark.asyncio
    async def test_format_with_table_data(self, output_formatter):
        """Test: Format answer with table results"""
        answer = "Here is the quarterly revenue data:"
        search_results = [
            {
                "content": "| Quarter | Revenue |\n| --- | --- |\n| Q1 | 5000 |\n| Q2 | 6000 |",
                "source": "Q3_Report.pdf",
                "doc_id": "doc_123"
            }
        ]
        
        formatted = await output_formatter.format_answer(
            answer,
            search_results=search_results,
            detected_data_types=[DataType.TABLE]
        )
        
        assert formatted is not None
        assert "Data Tables" in formatted or "Revenue" in formatted
    
    @pytest.mark.asyncio
    async def test_format_with_calculations(self, output_formatter):
        """Test: Format answer with calculation results"""
        answer = "The total revenue is calculated below"
        tool_results = {
            "calculator": {
                "total_revenue": 15500,
                "growth_rate": 0.15
            }
        }
        
        formatted = await output_formatter.format_answer(
            answer,
            search_results=[],
            tool_results=tool_results,
            detected_data_types=[DataType.NUMERIC]
        )
        
        assert formatted is not None
        assert "Calculations" in formatted or "15500" in formatted
    
    @pytest.mark.asyncio
    async def test_format_with_sources(self, output_formatter):
        """Test: Format answer with source citations"""
        answer = "Based on the analysis:"
        search_results = [
            {"content": "Data...", "source": "FPT_Q3_2024.pdf", "doc_id": "1"},
            {"content": "More...", "source": "Annual_Report.pdf", "doc_id": "2"}
        ]
        
        formatted = await output_formatter.format_answer(
            answer,
            search_results=search_results
        )
        
        assert formatted is not None
        assert "Source" in formatted or "FPT_Q3" in formatted
    
    @pytest.mark.asyncio
    async def test_parse_markdown_table(self, output_formatter):
        """Test: Parse markdown table format"""
        table_content = "| Company | Revenue |\n| --- | --- |\n| FPT | 5000 |\n| TCB | 3000 |"
        
        is_table = output_formatter._is_table_content(table_content)
        assert is_table == True
    
    @pytest.mark.asyncio
    async def test_is_table_content_detection(self, output_formatter):
        """Test: Detect table content"""
        markdown_table = "| A | B |\n|---|---|\n| 1 | 2 |"
        assert output_formatter._is_table_content(markdown_table) == True
        
        non_table = "This is just regular text without tables"
        assert output_formatter._is_table_content(non_table) == False
    
    @pytest.mark.asyncio
    async def test_format_value_with_units(self, output_formatter):
        """Test: Format values with appropriate units"""
        # Currency value
        formatted = await output_formatter._format_value("revenue", 15000)
        assert "$" in formatted or "15000" in formatted
        
        # Percentage
        formatted = await output_formatter._format_value("percent_growth", 15.5)
        assert "%" in formatted or "15.5" in formatted
    
    def test_formatter_summary(self, output_formatter):
        """Test: Formatter provides summary info"""
        summary = output_formatter.get_formatter_summary()
        
        assert "capabilities" in summary
        assert len(summary["capabilities"]) >= 3
        assert "TABLE" in summary["supported_data_types"]


class TestWorkflowObserver:
    """Tests for WorkflowObserver module"""
    
    @pytest.mark.asyncio
    async def test_emit_step_started(self, workflow_observer):
        """Test: Emit step started event"""
        step = await workflow_observer.emit_step_started("TEST_NODE")
        
        assert step is not None
        assert step.node_name == "TEST_NODE"
        assert step.step_number == 1
        assert step.status.value == "started"
    
    @pytest.mark.asyncio
    async def test_emit_step_completed(self, workflow_observer):
        """Test: Emit step completed event"""
        step = await workflow_observer.emit_step_started("TEST_NODE")
        await workflow_observer.emit_step_completed(step, output_size=1000)
        
        assert step.status.value == "completed"
        assert step.duration is not None
        assert step.duration > 0
    
    @pytest.mark.asyncio
    async def test_emit_step_failed(self, workflow_observer):
        """Test: Emit step failed event"""
        step = await workflow_observer.emit_step_started("FAILING_NODE")
        await workflow_observer.emit_step_failed(step, "Test error")
        
        assert step.status.value == "failed"
        assert step.error == "Test error"
    
    @pytest.mark.asyncio
    async def test_emit_step_skipped(self, workflow_observer):
        """Test: Emit step skipped event"""
        await workflow_observer.emit_step_skipped("SKIPPED_NODE", "No input data")
        
        assert len(workflow_observer.steps) == 1
        assert workflow_observer.steps[0].status.value == "skipped"
    
    @pytest.mark.asyncio
    async def test_performance_tracking(self, workflow_observer):
        """Test: Track performance metrics"""
        # Run multiple steps
        for i in range(3):
            step = await workflow_observer.emit_step_started(f"NODE_{i}")
            await asyncio.sleep(0.01)  # Simulate work
            await workflow_observer.emit_step_completed(step)
        
        perf = workflow_observer.get_performance_summary()
        
        assert len(perf) == 3
        assert all("avg_time_ms" in metrics for metrics in perf.values())
    
    @pytest.mark.asyncio
    async def test_workflow_summary(self, workflow_observer):
        """Test: Get workflow execution summary"""
        step1 = await workflow_observer.emit_step_started("NODE_1")
        await workflow_observer.emit_step_completed(step1)
        
        step2 = await workflow_observer.emit_step_started("NODE_2")
        await workflow_observer.emit_step_failed(step2, "Error")
        
        await workflow_observer.emit_workflow_completed()
        
        summary = workflow_observer.get_workflow_summary()
        
        assert summary["total_steps"] == 2
        assert summary["completed"] == 1
        assert summary["failed"] == 1
        assert summary["total_duration_ms"] > 0
    
    @pytest.mark.asyncio
    async def test_execution_trace(self, workflow_observer):
        """Test: Get execution trace"""
        step1 = await workflow_observer.emit_step_started("NODE_A")
        await workflow_observer.emit_step_completed(step1, output_size=500)
        
        step2 = await workflow_observer.emit_step_started("NODE_B")
        await workflow_observer.emit_step_completed(step2, output_size=1000)
        
        trace = workflow_observer.get_execution_trace()
        
        assert len(trace) == 2
        assert trace[0]["node_name"] == "NODE_A"
        assert trace[1]["node_name"] == "NODE_B"
        assert trace[0]["status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_register_callback(self, workflow_observer):
        """Test: Register and invoke callbacks"""
        callback_results = []
        
        async def test_callback(step):
            callback_results.append(step.node_name)
        
        workflow_observer.register_callback(test_callback)
        
        step = await workflow_observer.emit_step_started("CALLBACK_TEST")
        await workflow_observer.emit_step_completed(step)
        
        assert "CALLBACK_TEST" in callback_results


class TestWorkflowV4Phase3:
    """Integration tests for V4 workflow Phase 3 nodes"""
    
    @pytest.mark.asyncio
    async def test_prompt_handler_node(self, agent):
        """Test: PROMPT_HANDLER routes based on prompt"""
        if not agent.langgraph_workflow_v4:
            pytest.skip("V4 workflow not available")
        
        state = create_initial_state(user_prompt="Test prompt")
        result = await agent.langgraph_workflow_v4.node_prompt_handler(state)
        
        assert result is not None
        assert result == state
    
    @pytest.mark.asyncio
    async def test_file_handler_node(self, agent):
        """Test: FILE_HANDLER processes uploaded files"""
        if not agent.langgraph_workflow_v4:
            pytest.skip("V4 workflow not available")
        
        class MockFile:
            filename = "test.pdf"
            size = 5000
        
        state = create_initial_state(uploaded_files=[MockFile()])
        result = await agent.langgraph_workflow_v4.node_file_handler(state)
        
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_format_output_node(self, agent):
        """Test: FORMAT_OUTPUT formats answer with sources"""
        if not agent.langgraph_workflow_v4:
            pytest.skip("V4 workflow not available")
        
        state = create_initial_state(
            generated_answer="FPT revenue is 10000",
            best_search_results=[
                {
                    "content": "Financial data...",
                    "source": "Q3_Report",
                    "doc_id": "doc_1"
                }
            ],
            detected_data_types=[DataType.NUMERIC]
        )
        
        result = await agent.langgraph_workflow_v4.node_format_output(state)
        
        assert "formatted_answer" in result
        assert len(result["formatted_answer"]) > 0
    
    @pytest.mark.asyncio
    async def test_execute_tools_node(self, agent):
        """Test: EXECUTE_TOOLS runs selected tools"""
        if not agent.langgraph_workflow_v4:
            pytest.skip("V4 workflow not available")
        
        state = create_initial_state(
            selected_tools=[],  # No tools selected
            user_prompt="Calculate total"
        )
        
        result = await agent.langgraph_workflow_v4.node_execute_tools(state)
        
        assert "tool_results" in result


class TestWorkflowV4Complete:
    """End-to-end tests for complete V4 workflow"""
    
    @pytest.mark.asyncio
    async def test_routing_prompt_handler(self, agent):
        """Test: PROMPT_HANDLER routing decision"""
        if not agent.langgraph_workflow_v4:
            pytest.skip("V4 workflow not available")
        
        # With prompt
        state1 = create_initial_state(user_prompt="Test")
        result = await agent.langgraph_workflow_v4.node_prompt_handler(state1)
        assert result is not None
        
        # Without prompt (should route to FILE_HANDLER)
        state2 = create_initial_state(user_prompt="")
        result = await agent.langgraph_workflow_v4.node_prompt_handler(state2)
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_workflow_with_files_and_chitchat(self, agent):
        """Test: Workflow with files but chitchat classification"""
        if not agent.langgraph_workflow_v4:
            pytest.skip("V4 workflow not available")
        
        class MockFile:
            filename = "report.pdf"
            size = 5000
        
        state = create_initial_state(
            user_prompt="Hello, how are you?",
            uploaded_files=[MockFile()]
        )
        
        # Classify
        state = await agent.langgraph_workflow_v4.node_classify(state)
        
        # Should be classified as chitchat
        assert state["is_chitchat"] == True
    
    @pytest.mark.asyncio
    async def test_workflow_with_rewrite_evaluation(self, agent):
        """Test: Workflow evaluates and rewrites query"""
        if not agent.langgraph_workflow_v4:
            pytest.skip("V4 workflow not available")
        
        state = create_initial_state(
            user_prompt="What about it?",
            uploaded_files=[{"filename": "data.pdf"}]
        )
        
        state = await agent.langgraph_workflow_v4.node_extract_file(state)
        state = await agent.langgraph_workflow_v4.node_ingest_file(state)
        state = await agent.langgraph_workflow_v4.node_rewrite_eval(state)
        
        assert "needs_rewrite" in state
        assert "rewrite_context_type" in state
    
    @pytest.mark.asyncio
    async def test_formatting_with_observer(self, agent):
        """Test: Formatting with observer tracking"""
        if not agent.langgraph_workflow_v4:
            pytest.skip("V4 workflow not available")
        
        # Create workflow with observer
        workflow = agent.langgraph_workflow_v4
        
        state = create_initial_state(
            generated_answer="Test answer",
            best_search_results=[
                {"content": "Data", "source": "Report", "doc_id": "1"}
            ]
        )
        
        result = await workflow.node_format_output(state)
        
        assert "formatted_answer" in result
        if workflow.observer:
            summary = workflow.observer.get_workflow_summary()
            assert summary["total_steps"] > 0


class TestPhase3Quality:
    """Quality assurance tests for Phase 3"""
    
    @pytest.mark.asyncio
    async def test_all_13_nodes_callable(self, agent):
        """Test: All 13 nodes are implemented and callable"""
        if not agent.langgraph_workflow_v4:
            pytest.skip("V4 workflow not available")
        
        workflow = agent.langgraph_workflow_v4
        
        # Check all node methods exist
        nodes = [
            "node_prompt_handler",
            "node_file_handler",
            "node_classify",
            "node_direct_response",
            "node_extract_file",
            "node_ingest_file",
            "node_rewrite_eval",
            "node_rewrite_file_context",
            "node_rewrite_conversation_context",
            "node_retrieve",
            "node_filter",
            "node_analyze",
            "node_select_tools",
            "node_generate",
            "node_execute_tools",
            "node_format_output"
        ]
        
        for node_name in nodes:
            assert hasattr(workflow, node_name), f"Missing node: {node_name}"
            method = getattr(workflow, node_name)
            assert callable(method), f"Node {node_name} is not callable"
    
    def test_output_formatter_markdown(self, output_formatter):
        """Test: OutputFormatter uses markdown format"""
        assert output_formatter.use_markdown == True
    
    def test_observer_step_status_enum(self, workflow_observer):
        """Test: Observer has step status enum"""
        assert StepStatus.STARTED.value == "started"
        assert StepStatus.COMPLETED.value == "completed"
        assert StepStatus.FAILED.value == "failed"
        assert StepStatus.SKIPPED.value == "skipped"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

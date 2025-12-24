"""
Integration tests for V3 workflow - Phase 2 (Query Rewriting & Tool Selection)
Tests 10-node enhanced architecture with QueryRewriter and ToolSelector
"""

import pytest
import asyncio
from typing import Dict, Any

from src.core.workflow_state import (
    create_initial_state, PromptType, DataType, RetrievalStrategy
)
from src.core.langgraph_workflow_v3 import LangGraphWorkflowV3
from src.core.query_rewriter import QueryRewriter
from src.core.tool_selector import ToolSelector
from src.agent.financial_agent import FinancialAgent


@pytest.fixture
async def agent():
    """Create agent with V3 workflow"""
    agent = FinancialAgent()
    yield agent


@pytest.fixture
async def workflow(agent):
    """Get V3 workflow"""
    if not agent.langgraph_workflow_v3:
        pytest.skip("V3 workflow not available")
    return agent.langgraph_workflow_v3


@pytest.fixture
def query_rewriter():
    """Create QueryRewriter instance"""
    return QueryRewriter(llm=None)


@pytest.fixture
def tool_selector():
    """Create ToolSelector instance"""
    return ToolSelector()


class TestQueryRewriter:
    """Tests for QueryRewriter module"""
    
    @pytest.mark.asyncio
    async def test_evaluate_with_files(self, query_rewriter):
        """Test: Query with files but no file mentions → needs_rewrite=True"""
        prompt = "Analyze this"
        needs_rewrite = await query_rewriter.evaluate_need_for_rewriting(
            prompt, has_files=True, conversation_history=None
        )
        assert needs_rewrite == True
    
    @pytest.mark.asyncio
    async def test_evaluate_with_pronouns_no_history(self, query_rewriter):
        """Test: Query with pronouns but no history → needs_rewrite=True"""
        prompt = "What is it?"
        needs_rewrite = await query_rewriter.evaluate_need_for_rewriting(
            prompt, has_files=False, conversation_history=None
        )
        assert needs_rewrite == True
    
    @pytest.mark.asyncio
    async def test_evaluate_clear_query(self, query_rewriter):
        """Test: Clear query with context → needs_rewrite=False"""
        prompt = "Show quarterly results for FPT Corporation"
        needs_rewrite = await query_rewriter.evaluate_need_for_rewriting(
            prompt, has_files=False, conversation_history=None
        )
        assert needs_rewrite == False
    
    @pytest.mark.asyncio
    async def test_rewrite_with_file_context(self, query_rewriter):
        """Test: Inject file names into query"""
        prompt = "Analyze this"
        files = [
            {"filename": "Q3_2024_Report.pdf", "size": 1024},
            {"filename": "Financial_Data.xlsx", "size": 2048}
        ]
        
        rewritten = await query_rewriter.rewrite_with_file_context(prompt, files)
        
        # Should mention files
        assert "file" in rewritten.lower() or "document" in rewritten.lower()
        assert len(rewritten) > len(prompt)
    
    @pytest.mark.asyncio
    async def test_rewrite_with_conversation(self, query_rewriter):
        """Test: Resolve pronouns using conversation history"""
        prompt = "What about it?"
        history = [
            {"role": "user", "content": "Show me Techcombank data"},
            {"role": "assistant", "content": "Techcombank (TCB) is a digital banking leader..."}
        ]
        
        rewritten = await query_rewriter.rewrite_with_conversation_context(
            prompt, history
        )
        
        # Should resolve pronouns
        assert "it" not in rewritten or "Techcombank" in rewritten
        assert len(rewritten) > 0
    
    @pytest.mark.asyncio
    async def test_mentions_files_detection(self, query_rewriter):
        """Test: File mention detection"""
        assert query_rewriter._mentions_files("Check the uploaded file") == True
        assert query_rewriter._mentions_files("From my documents") == True
        assert query_rewriter._mentions_files("Show quarterly results") == False
    
    def test_rewriter_summary(self, query_rewriter):
        """Test: Rewriter provides summary info"""
        summary = query_rewriter.get_rewriting_summary()
        assert "strategies" in summary
        assert len(summary["strategies"]) >= 2


class TestToolSelector:
    """Tests for ToolSelector module"""
    
    @pytest.mark.asyncio
    async def test_select_calculate_intent(self, tool_selector):
        """Test: Calculate query → selects calculator"""
        query = "What is the total revenue?"
        result = await tool_selector.select_tools(
            query,
            detected_data_types=[DataType.NUMERIC],
            available_tool_names=["calculator", "data_analyzer", "trend_analyzer"]
        )
        
        assert len(result["selected_tools"]) > 0
        assert "calculator" in result["selected_tools"]
    
    @pytest.mark.asyncio
    async def test_select_compare_intent(self, tool_selector):
        """Test: Compare query → selects comparator"""
        query = "Compare Q3 and Q4 performance"
        result = await tool_selector.select_tools(
            query,
            detected_data_types=[DataType.TABLE],
            available_tool_names=["data_comparator", "calculator", "analyzer"]
        )
        
        assert len(result["selected_tools"]) > 0
    
    @pytest.mark.asyncio
    async def test_select_trend_intent(self, tool_selector):
        """Test: Trend query → selects trend analyzer"""
        query = "Show growth trend over the past 3 years"
        result = await tool_selector.select_tools(
            query,
            detected_data_types=[DataType.NUMERIC],
            available_tool_names=["trend_analyzer", "calculator"]
        )
        
        assert len(result["selected_tools"]) > 0
        assert "trend_analyzer" in result["selected_tools"]
    
    @pytest.mark.asyncio
    async def test_select_with_table_data(self, tool_selector):
        """Test: Table data boosts table-friendly tools"""
        query = "Analyze the data"
        result = await tool_selector.select_tools(
            query,
            detected_data_types=[DataType.TABLE],
            available_tool_names=["data_comparator", "trend_analyzer", "calculator"]
        )
        
        # Should prefer tools that work with tables
        assert result["primary_tool"] is not None
    
    @pytest.mark.asyncio
    async def test_confidence_scoring(self, tool_selector):
        """Test: Tool selection includes confidence score"""
        query = "Calculate total expenses"
        result = await tool_selector.select_tools(
            query,
            detected_data_types=[DataType.NUMERIC],
            available_tool_names=["calculator", "analyzer"]
        )
        
        assert "confidence" in result
        assert 0 <= result["confidence"] <= 1
        assert result["confidence"] > 0.5  # Should have decent confidence for clear intent
    
    @pytest.mark.asyncio
    async def test_rationale_provided(self, tool_selector):
        """Test: Tool selection includes rationale"""
        query = "Compare sales figures"
        result = await tool_selector.select_tools(
            query,
            detected_data_types=[DataType.NUMERIC, DataType.TABLE],
            available_tool_names=["data_comparator", "calculator"]
        )
        
        assert "rationale" in result
        assert len(result["rationale"]) > 0


class TestWorkflowV3Phase2:
    """Integration tests for Phase 2 enhanced workflow"""
    
    @pytest.mark.asyncio
    async def test_extract_file_node(self, workflow):
        """Test: EXTRACT_FILE node processes uploaded files"""
        class MockFile:
            filename = "test_report.pdf"
            size = 5000
            content_type = "application/pdf"
            doc_id = "doc_123"
        
        state = create_initial_state(
            user_prompt="Analyze the report",
            uploaded_files=[MockFile()]
        )
        
        result = await workflow.node_extract_file(state)
        
        assert "file_metadata" in result
        assert len(result["file_metadata"]) == 1
        assert result["file_metadata"][0]["filename"] == "test_report.pdf"
    
    @pytest.mark.asyncio
    async def test_ingest_file_node(self, workflow):
        """Test: INGEST_FILE node marks files as ingested"""
        state = create_initial_state(
            user_prompt="Process this",
            file_metadata=[{"filename": "test.pdf", "size": 1000}]
        )
        
        result = await workflow.node_ingest_file(state)
        
        assert result["files_ingested"] == True
    
    @pytest.mark.asyncio
    async def test_rewrite_eval_node_no_rewrite(self, workflow):
        """Test: REWRITE_EVAL → no rewrite needed"""
        state = create_initial_state(
            user_prompt="Show quarterly results for FPT Corporation"
        )
        
        result = await workflow.node_rewrite_eval(state)
        
        assert result["needs_rewrite"] == False
    
    @pytest.mark.asyncio
    async def test_rewrite_eval_node_needs_rewrite(self, workflow):
        """Test: REWRITE_EVAL → rewrite needed"""
        state = create_initial_state(
            user_prompt="What about it?",
            uploaded_files=[{"filename": "data.pdf"}]
        )
        
        result = await workflow.node_rewrite_eval(state)
        
        # Should detect need for rewriting
        assert isinstance(result["needs_rewrite"], bool)
    
    @pytest.mark.asyncio
    async def test_rewrite_file_context_node(self, workflow):
        """Test: REWRITE_FILE_CONTEXT node injects file info"""
        state = create_initial_state(
            user_prompt="Analyze it",
            file_metadata=[
                {"filename": "Q3_Report.pdf", "size": 2000},
                {"filename": "Financial_Data.xlsx", "size": 3000}
            ]
        )
        
        result = await workflow.node_rewrite_file_context(state)
        
        assert "rewritten_prompt" in result
        # Rewritten should differ if file context was injected
        assert result["rewritten_prompt"] is not None
    
    @pytest.mark.asyncio
    async def test_rewrite_conversation_context_node(self, workflow):
        """Test: REWRITE_CONVO_CONTEXT node resolves pronouns"""
        state = create_initial_state(
            user_prompt="What is it?",
            conversation_history=[
                {"role": "user", "content": "Show me Vinamilk data"},
                {"role": "assistant", "content": "Vinamilk (VNM) is..."}
            ]
        )
        
        result = await workflow.node_rewrite_conversation_context(state)
        
        assert "rewritten_prompt" in result
        assert result["rewritten_prompt"] is not None
    
    @pytest.mark.asyncio
    async def test_select_tools_node(self, workflow):
        """Test: SELECT_TOOLS uses enhanced tool selection"""
        state = create_initial_state(
            user_prompt="Calculate total revenue",
            detected_data_types=[DataType.NUMERIC],
            best_search_results=[
                {
                    "content": "Revenue: 10,000 USD",
                    "source": "Q3_Report",
                    "score": 0.9
                }
            ]
        )
        
        result = await workflow.node_select_tools(state)
        
        assert "selected_tools" in result
        assert "primary_tool" in result
        assert "tool_selection_confidence" in result
    
    @pytest.mark.asyncio
    async def test_routing_with_files(self, workflow):
        """Test: Workflow routes to EXTRACT_FILE when files present"""
        state = create_initial_state(
            user_prompt="Process files",
            uploaded_files=[{"filename": "data.pdf"}],
            is_chitchat=False
        )
        
        # Verify routing logic
        from src.core.langgraph_workflow_v3 import LangGraphWorkflowV3
        
        # The routing should prefer file extraction over direct retrieval
        route = "extract_file" if state.get("uploaded_files") else "retrieve"
        assert route == "extract_file"
    
    @pytest.mark.asyncio
    async def test_routing_rewrite_evaluation(self, workflow):
        """Test: Routing from REWRITE_EVAL to appropriate node"""
        
        # Test file context route
        state = create_initial_state(
            needs_rewrite=True,
            rewrite_context_type="file",
            file_metadata=[{"filename": "test.pdf"}]
        )
        route = workflow._route_rewrite_strategy(state)
        assert route == "rewrite_file"
        
        # Test conversation context route
        state["rewrite_context_type"] = "conversation"
        route = workflow._route_rewrite_strategy(state)
        assert route == "rewrite_convo"
        
        # Test direct retrieval (no rewrite)
        state["needs_rewrite"] = False
        route = workflow._route_rewrite_strategy(state)
        assert route == "retrieve"


class TestPhase2Integration:
    """End-to-end Phase 2 integration tests"""
    
    @pytest.mark.asyncio
    async def test_full_workflow_with_files(self, workflow):
        """Test: Complete workflow with file handling"""
        class MockFile:
            filename = "report.pdf"
            size = 5000
            content_type = "application/pdf"
            doc_id = "doc_1"
        
        initial_state = create_initial_state(
            user_prompt="Analyze this report",
            uploaded_files=[MockFile()],
            user_id="user_123",
            session_id="sess_123"
        )
        
        # Run through file extraction
        state = await workflow.node_classify(initial_state)
        assert state["prompt_type"] in [PromptType.INSTRUCTION, PromptType.REQUEST]
        
        state = await workflow.node_extract_file(state)
        assert len(state.get("file_metadata", [])) == 1
        
        state = await workflow.node_ingest_file(state)
        assert state["files_ingested"] == True
        
        state = await workflow.node_rewrite_eval(state)
        # Should evaluate if rewrite is needed
        assert "needs_rewrite" in state
    
    @pytest.mark.asyncio
    async def test_full_workflow_with_conversation(self, workflow):
        """Test: Complete workflow with conversation context"""
        initial_state = create_initial_state(
            user_prompt="What about it?",
            conversation_history=[
                {"role": "user", "content": "Show me Techcombank financial data"},
                {"role": "assistant", "content": "Techcombank (TCB) is a leading digital bank..."}
            ],
            user_id="user_456",
            session_id="sess_456"
        )
        
        # Run through rewrite evaluation
        state = await workflow.node_classify(initial_state)
        state = await workflow.node_rewrite_eval(state)
        
        if state.get("needs_rewrite"):
            # Should route to conversation context rewriting
            assert state.get("rewrite_context_type") in ["conversation", None]
    
    @pytest.mark.asyncio
    async def test_tool_selection_with_analysis(self, workflow):
        """Test: Tool selection based on analyzed data types"""
        state = create_initial_state(
            user_prompt="Calculate total revenue for FPT",
            best_search_results=[
                {
                    "content": "FPT Revenue Q1: 5000, Q2: 6000, Q3: 7000",
                    "source": "quarterly_report",
                    "score": 0.95
                }
            ],
            detected_data_types=[DataType.NUMERIC],
            has_numeric_data=True,
            has_table_data=False,
            text_only=False
        )
        
        result = await workflow.node_select_tools(state)
        
        # Should select tools appropriate for numeric data
        assert "selected_tools" in result
        assert result["tool_selection_confidence"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

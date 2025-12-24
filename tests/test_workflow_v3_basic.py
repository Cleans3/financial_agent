"""
Integration tests for V3 workflow
Tests 8-node bridge architecture
"""

import pytest
import asyncio
from src.core.workflow_state import create_initial_state, PromptType
from src.core.langgraph_workflow_v3 import LangGraphWorkflowV3
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


class TestWorkflowV3Basic:
    """Basic workflow tests"""
    
    @pytest.mark.asyncio
    async def test_classify_greeting(self, workflow):
        """Test: greeting → is_chitchat=True"""
        prompt = "Hello, how are you?"
        state = create_initial_state(user_prompt=prompt)
        
        # Run classify node
        result = await workflow.node_classify(state)
        
        assert result["is_chitchat"] == True
        assert result["prompt_type"] in [PromptType.CHITCHAT, PromptType.INSTRUCTION]
    
    @pytest.mark.asyncio
    async def test_classify_request(self, workflow):
        """Test: request question → is_chitchat=False"""
        prompt = "Show me the quarterly results for FPT"
        state = create_initial_state(user_prompt=prompt)
        
        result = await workflow.node_classify(state)
        
        assert result["is_chitchat"] == False
    
    @pytest.mark.asyncio
    async def test_direct_response_generation(self, workflow):
        """Test: direct_response node generates answer"""
        prompt = "Hi there!"
        state = create_initial_state(user_prompt=prompt, user_id="test")
        state["is_chitchat"] = True
        
        result = await workflow.node_direct_response(state)
        
        assert result["generated_answer"] != ""
        assert len(result["generated_answer"]) > 0
    
    @pytest.mark.asyncio
    async def test_retrieve_node(self, workflow):
        """Test: retrieve node returns results"""
        prompt = "What is FPT Corporation?"
        state = create_initial_state(
            user_prompt=prompt,
            user_id="test_user"
        )
        
        result = await workflow.node_retrieve(state)
        
        # Should have some structure even if no RAG service
        assert "personal_semantic_results" in result
        assert "personal_keyword_results" in result
        assert "search_metadata" in result
    
    @pytest.mark.asyncio
    async def test_filter_node(self, workflow):
        """Test: filter node processes results"""
        state = create_initial_state(user_prompt="test")
        
        # Add mock results
        state["personal_semantic_results"] = [
            {"doc_id": "1", "content": "test1", "score": 0.9},
            {"doc_id": "2", "content": "test2", "score": 0.7}
        ]
        state["personal_keyword_results"] = []
        state["global_semantic_results"] = []
        state["global_keyword_results"] = []
        
        result = await workflow.node_filter(state)
        
        assert "best_search_results" in result
        assert isinstance(result["best_search_results"], list)
    
    @pytest.mark.asyncio
    async def test_analyze_node(self, workflow):
        """Test: analyze node detects data types"""
        state = create_initial_state(user_prompt="test")
        
        # Add results with numeric data
        state["best_search_results"] = [
            {"content": "Revenue: $1,000,000 in Q1", "score": 0.9}
        ]
        
        result = await workflow.node_analyze(state)
        
        assert "has_table_data" in result
        assert "has_numeric_data" in result
        assert "text_only" in result
    
    @pytest.mark.asyncio
    async def test_select_tools_node(self, workflow):
        """Test: select_tools node chooses tools"""
        state = create_initial_state(user_prompt="Calculate 50% growth")
        state["has_numeric_data"] = True
        
        result = await workflow.node_select_tools(state)
        
        assert "selected_tools" in result
        assert isinstance(result["selected_tools"], list)
    
    @pytest.mark.asyncio
    async def test_generate_node(self, workflow):
        """Test: generate node creates answer"""
        state = create_initial_state(user_prompt="What is FPT?")
        state["best_search_results"] = [
            {
                "source": "test.txt",
                "content": "FPT is a Vietnamese tech company"
            }
        ]
        
        result = await workflow.node_generate(state)
        
        assert result["generated_answer"] != ""
    
    @pytest.mark.asyncio
    async def test_full_workflow_chitchat(self, workflow):
        """Test: full workflow for chitchat"""
        state = create_initial_state(user_prompt="Hello!")
        
        final_state = await workflow.graph.ainvoke(state)
        
        assert final_state["is_chitchat"] == True
        assert final_state["generated_answer"] != ""
    
    @pytest.mark.asyncio
    async def test_full_workflow_question(self, workflow):
        """Test: full workflow for real question"""
        state = create_initial_state(
            user_prompt="Tell me about Vietnam",
            user_id="test_user"
        )
        
        final_state = await workflow.graph.ainvoke(state)
        
        assert final_state["is_chitchat"] == False
        assert final_state["generated_answer"] != ""


class TestStateManagement:
    """Test state transitions"""
    
    @pytest.mark.asyncio
    async def test_state_initialization(self, workflow):
        """Test: initial state has correct fields"""
        state = create_initial_state(
            user_prompt="test",
            user_id="user1",
            session_id="session1"
        )
        
        assert state["user_prompt"] == "test"
        assert state["user_id"] == "user1"
        assert state["session_id"] == "session1"
        assert state["is_chitchat"] == False
        assert state["selected_tools"] == []
        assert state["generated_answer"] == ""
    
    @pytest.mark.asyncio
    async def test_state_flow_through_nodes(self, workflow):
        """Test: state flows correctly through classify and direct_response"""
        state = create_initial_state(user_prompt="Hi!")
        
        # Classify
        state = await workflow.node_classify(state)
        assert state["prompt_type"] is not None
        
        # Direct response
        if state["is_chitchat"]:
            state = await workflow.node_direct_response(state)
            assert state["generated_answer"] != ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

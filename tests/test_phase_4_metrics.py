import pytest
import asyncio
from datetime import datetime
from src.core.workflow_metrics import (
    WorkflowMetrics,
    WorkflowMetricsCollector,
    workflow_metrics,
)


class TestWorkflowMetrics:
    """Tests for WorkflowMetrics dataclass."""
    
    def test_metrics_creation(self):
        """Test creating a basic WorkflowMetrics."""
        metrics = WorkflowMetrics(
            user_id="user123",
            workflow_version="v4",
            prompt="What is VCB stock price?",
            total_duration_ms=1250.5,
        )
        
        assert metrics.user_id == "user123"
        assert metrics.workflow_version == "v4"
        assert metrics.total_duration_ms == 1250.5
        assert metrics.had_error is False
        assert isinstance(metrics.timestamp, datetime)
    
    def test_metrics_with_nodes(self):
        """Test metrics with node execution times."""
        node_durations = {
            "CLASSIFY": 50,
            "RETRIEVE": 450,
            "FILTER": 75,
            "ANALYZE": 100,
            "GENERATE": 300,
            "FORMAT_OUTPUT": 125,
        }
        
        metrics = WorkflowMetrics(
            user_id="user456",
            workflow_version="v4",
            prompt="test",
            total_duration_ms=1100,
            node_durations=node_durations,
        )
        
        assert len(metrics.node_durations) == 6
        assert metrics.node_durations["RETRIEVE"] == 450
    
    def test_metrics_with_error(self):
        """Test metrics recording an error."""
        metrics = WorkflowMetrics(
            user_id="user789",
            workflow_version="v3",
            prompt="test",
            total_duration_ms=500,
            had_error=True,
            error_type="TimeoutError",
        )
        
        assert metrics.had_error is True
        assert metrics.error_type == "TimeoutError"
    
    def test_metrics_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = WorkflowMetrics(
            user_id="user999",
            workflow_version="v4",
            prompt="Long prompt " * 10,
            total_duration_ms=2500,
            node_durations={"RETRIEVE": 500, "GENERATE": 1000},
            tool_selected="calculate_price",
            num_results_retrieved=5,
        )
        
        metrics_dict = metrics.to_dict()
        
        assert metrics_dict["user_id"] == "user999"
        assert metrics_dict["workflow_version"] == "v4"
        assert metrics_dict["prompt_length"] > 100
        assert metrics_dict["total_duration_ms"] == 2500
        assert metrics_dict["slowest_node"] == "GENERATE"
        assert metrics_dict["slowest_node_ms"] == 1000
        assert metrics_dict["tool_selected"] == "calculate_price"


class TestWorkflowMetricsCollector:
    """Tests for WorkflowMetricsCollector."""
    
    @pytest.fixture
    async def collector(self):
        """Create a fresh collector for each test."""
        return WorkflowMetricsCollector(max_history=1000)
    
    @pytest.mark.asyncio
    async def test_record_single_metric(self):
        """Test recording a single metric."""
        collector = WorkflowMetricsCollector()
        
        metrics = WorkflowMetrics(
            user_id="user1",
            workflow_version="v4",
            prompt="test",
            total_duration_ms=1000,
        )
        
        await collector.record_metric(metrics)
        
        assert len(collector._metrics) == 1
        assert collector._version_usage["v4"] == 1
    
    @pytest.mark.asyncio
    async def test_version_usage_stats(self):
        """Test version usage statistics."""
        collector = WorkflowMetricsCollector()
        
        # Record metrics for v3 and v4
        for i in range(30):
            version = "v4" if i % 3 == 0 else "v3"
            metrics = WorkflowMetrics(
                user_id=f"user_{i}",
                workflow_version=version,
                prompt="test",
                total_duration_ms=1000,
            )
            await collector.record_metric(metrics)
        
        stats = await collector.get_version_usage_stats()
        
        assert "v3" in stats
        assert "v4" in stats
        assert stats["v3"]["usage_count"] == 20
        assert stats["v4"]["usage_count"] == 10
    
    @pytest.mark.asyncio
    async def test_error_tracking(self):
        """Test error statistics tracking."""
        collector = WorkflowMetricsCollector()
        
        # Record successful metrics
        for i in range(7):
            metrics = WorkflowMetrics(
                user_id=f"user_{i}",
                workflow_version="v4",
                prompt="test",
                total_duration_ms=1000,
                had_error=False,
            )
            await collector.record_metric(metrics)
        
        # Record error metrics
        for i in range(3):
            metrics = WorkflowMetrics(
                user_id=f"user_error_{i}",
                workflow_version="v4",
                prompt="test",
                total_duration_ms=1000,
                had_error=True,
                error_type="TimeoutError",
            )
            await collector.record_metric(metrics)
        
        error_stats = await collector.get_error_stats()
        
        assert error_stats["total_executions"] == 10
        assert error_stats["total_errors"] == 3
        assert error_stats["error_rate"] == 30.0
        assert error_stats["errors_by_type"]["TimeoutError"] == 3
    
    @pytest.mark.asyncio
    async def test_node_performance_stats(self):
        """Test node performance statistics."""
        collector = WorkflowMetricsCollector()
        
        # Record metrics with various node times
        node_durations = {
            "CLASSIFY": [50, 55, 52, 48, 51],
            "RETRIEVE": [400, 450, 500, 425, 475],
            "GENERATE": [300, 310, 290, 320, 315],
        }
        
        for i in range(5):
            metrics = WorkflowMetrics(
                user_id=f"user_{i}",
                workflow_version="v4",
                prompt="test",
                total_duration_ms=1500,
                node_durations={
                    node: durations[i]
                    for node, durations in node_durations.items()
                },
            )
            await collector.record_metric(metrics)
        
        stats = await collector.get_node_performance_stats()
        
        assert "CLASSIFY" in stats
        assert stats["CLASSIFY"]["execution_count"] == 5
        assert stats["CLASSIFY"]["avg_duration_ms"] == pytest.approx(51.2)
        assert stats["CLASSIFY"]["min_duration_ms"] == 48
        assert stats["CLASSIFY"]["max_duration_ms"] == 55
        
        assert "RETRIEVE" in stats
        assert stats["RETRIEVE"]["avg_duration_ms"] == pytest.approx(450)
    
    @pytest.mark.asyncio
    async def test_tool_statistics(self):
        """Test tool execution statistics."""
        collector = WorkflowMetricsCollector()
        
        # Record tool executions
        tools = ["calculate_price", "get_trends", "compare_stocks"]
        
        for i in range(30):
            tool = tools[i % 3]
            had_error = (i % 5 == 0)  # 20% error rate
            
            metrics = WorkflowMetrics(
                user_id=f"user_{i}",
                workflow_version="v4",
                prompt="test",
                total_duration_ms=1000,
                tool_selected=tool,
                tools_executed=[tool],
                had_error=had_error,
                error_type="RuntimeError" if had_error else None,
            )
            await collector.record_metric(metrics)
        
        tool_stats = await collector.get_tool_stats()
        
        assert "calculate_price" in tool_stats
        assert tool_stats["calculate_price"]["execution_count"] == 10
        assert tool_stats["calculate_price"]["error_count"] == 2
        assert tool_stats["calculate_price"]["success_rate"] == 80.0
    
    @pytest.mark.asyncio
    async def test_recent_metrics(self):
        """Test retrieving recent metrics."""
        collector = WorkflowMetricsCollector()
        
        # Record 150 metrics
        for i in range(150):
            metrics = WorkflowMetrics(
                user_id=f"user_{i}",
                workflow_version="v4",
                prompt="test",
                total_duration_ms=1000,
            )
            await collector.record_metric(metrics)
        
        recent = await collector.get_recent_metrics(limit=50)
        
        assert len(recent) == 50
        assert recent[0]["user_id"] == "user_100"  # First 100 skipped
    
    @pytest.mark.asyncio
    async def test_metrics_by_user(self):
        """Test retrieving metrics for a specific user."""
        collector = WorkflowMetricsCollector()
        
        # Record metrics for multiple users
        for i in range(100):
            user_id = "user_tracked" if i % 4 == 0 else f"user_{i}"
            metrics = WorkflowMetrics(
                user_id=user_id,
                workflow_version="v4",
                prompt="test",
                total_duration_ms=1000,
            )
            await collector.record_metric(metrics)
        
        user_metrics = await collector.get_metrics_by_user("user_tracked")
        
        assert len(user_metrics) == 25
        assert all(m["user_id"] == "user_tracked" for m in user_metrics)
    
    @pytest.mark.asyncio
    async def test_health_status(self):
        """Test health status calculation."""
        collector = WorkflowMetricsCollector()
        
        # Record mostly successful metrics (only 1 error out of 50 = 2%)
        for i in range(50):
            had_error = (i == 49)  # 1 error = 2% error rate
            metrics = WorkflowMetrics(
                user_id=f"user_{i}",
                workflow_version="v4",
                prompt="test",
                total_duration_ms=1000,
                node_durations={"RETRIEVE": 300, "GENERATE": 500},
                had_error=had_error,
                error_type="TimeoutError" if had_error else None,
            )
            await collector.record_metric(metrics)
        
        health = await collector.get_health_status()
        
        assert health["status"] == "HEALTHY"  # Error rate 2% is OK
        assert health["error_rate"] == 2.0
        assert health["avg_workflow_duration_ms"] > 0
    
    @pytest.mark.asyncio
    async def test_summary_generation(self):
        """Test comprehensive summary generation."""
        collector = WorkflowMetricsCollector()
        
        # Record diverse metrics
        for i in range(20):
            metrics = WorkflowMetrics(
                user_id=f"user_{i}",
                workflow_version="v4" if i % 2 == 0 else "v3",
                prompt="test",
                total_duration_ms=1000 + (i * 50),
                node_durations={"CLASSIFY": 50, "RETRIEVE": 300 + i, "GENERATE": 400},
                tool_selected="calc" if i % 3 == 0 else None,
                had_error=(i > 18),
                error_type="TimeoutError" if i > 18 else None,
            )
            await collector.record_metric(metrics)
        
        summary = await collector.get_summary()
        
        assert "version_usage" in summary
        assert "node_performance" in summary
        assert "error_summary" in summary
        assert "tool_performance" in summary
        assert "timestamp" in summary
    
    @pytest.mark.asyncio
    async def test_max_history_limit(self):
        """Test that metrics history is limited."""
        collector = WorkflowMetricsCollector(max_history=100)
        
        # Record 150 metrics (exceeds max)
        for i in range(150):
            metrics = WorkflowMetrics(
                user_id=f"user_{i}",
                workflow_version="v4",
                prompt="test",
                total_duration_ms=1000,
            )
            await collector.record_metric(metrics)
        
        # Should only keep last 100
        assert len(collector._metrics) == 100
        assert collector._metrics[0].user_id == "user_50"  # First 50 removed
    
    @pytest.mark.asyncio
    async def test_clear_history(self):
        """Test clearing all metrics."""
        collector = WorkflowMetricsCollector()
        
        # Record some metrics
        for i in range(20):
            metrics = WorkflowMetrics(
                user_id=f"user_{i}",
                workflow_version="v4",
                prompt="test",
                total_duration_ms=1000,
            )
            await collector.record_metric(metrics)
        
        assert len(collector._metrics) == 20
        
        await collector.clear_history()
        
        assert len(collector._metrics) == 0
        assert len(collector._version_usage) == 0


class TestMetricsIntegration:
    """Integration tests for metrics collection."""
    
    @pytest.mark.asyncio
    async def test_full_workflow_metrics(self):
        """Test complete metrics workflow."""
        collector = WorkflowMetricsCollector()
        
        # Simulate complete workflow with all data
        metrics = WorkflowMetrics(
            user_id="user_integration",
            workflow_version="v4",
            prompt="What is the stock price of VCB and TCB? Calculate the ratio.",
            total_duration_ms=3250,
            node_durations={
                "PROMPT_HANDLER": 25,
                "CLASSIFY": 50,
                "RETRIEVE": 1000,
                "FILTER": 100,
                "ANALYZE": 200,
                "SELECT_TOOLS": 150,
                "GENERATE": 750,
                "EXECUTE_TOOLS": 800,
                "FORMAT_OUTPUT": 175,
            },
            tool_selected="calculate_ratio",
            tools_executed=["get_stock_price", "calculate_ratio"],
            num_results_retrieved=8,
        )
        
        await collector.record_metric(metrics)
        
        # Verify all stats
        version_stats = await collector.get_version_usage_stats()
        assert version_stats["v4"]["usage_count"] == 1
        
        node_stats = await collector.get_node_performance_stats()
        assert "RETRIEVE" in node_stats
        assert node_stats["RETRIEVE"]["avg_duration_ms"] == 1000
        
        tool_stats = await collector.get_tool_stats()
        assert "calculate_ratio" in tool_stats
        
        summary = await collector.get_summary()
        assert summary["version_usage"]["v4"]["usage_count"] == 1


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

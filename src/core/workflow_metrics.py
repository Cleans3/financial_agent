import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class WorkflowMetrics:
    """Metrics for a single workflow execution."""
    user_id: str
    workflow_version: str
    prompt: str
    total_duration_ms: float
    node_durations: Dict[str, float] = field(default_factory=dict)  # {node_name: duration_ms}
    tool_selected: Optional[str] = None
    tools_executed: List[str] = field(default_factory=list)
    num_results_retrieved: int = 0
    had_error: bool = False
    error_type: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage/transmission."""
        return {
            "user_id": self.user_id,
            "workflow_version": self.workflow_version,
            "prompt_length": len(self.prompt),
            "total_duration_ms": self.total_duration_ms,
            "node_count": len(self.node_durations),
            "slowest_node": max(self.node_durations, key=self.node_durations.get) if self.node_durations else None,
            "slowest_node_ms": max(self.node_durations.values()) if self.node_durations else 0,
            "tool_selected": self.tool_selected,
            "tools_executed": self.tools_executed,
            "tool_count": len(self.tools_executed),
            "num_results_retrieved": self.num_results_retrieved,
            "had_error": self.had_error,
            "error_type": self.error_type,
            "timestamp": self.timestamp.isoformat(),
        }


class WorkflowMetricsCollector:
    """Collects and aggregates workflow metrics."""
    
    def __init__(self, max_history: int = 10000):
        self.logger = logging.getLogger(__name__)
        self.max_history = max_history
        self._metrics: List[WorkflowMetrics] = []
        self._lock = asyncio.Lock()
        
        # Aggregations
        self._version_usage: Dict[str, int] = defaultdict(int)
        self._error_counts: Dict[str, int] = defaultdict(int)
        self._error_by_version: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._node_times: Dict[str, List[float]] = defaultdict(list)
        self._tool_success_rates: Dict[str, Dict[str, float]] = {}
    
    async def record_metric(self, metrics: WorkflowMetrics) -> None:
        """Record a workflow execution metric."""
        async with self._lock:
            self._metrics.append(metrics)
            
            # Update aggregations
            self._version_usage[metrics.workflow_version] += 1
            
            if metrics.had_error:
                self._error_counts[metrics.error_type or "unknown"] += 1
                self._error_by_version[metrics.workflow_version][metrics.error_type or "unknown"] += 1
            
            # Track node execution times
            for node_name, duration in metrics.node_durations.items():
                self._node_times[node_name].append(duration)
            
            # Keep history limited
            if len(self._metrics) > self.max_history:
                self._metrics.pop(0)
            
            self.logger.info(
                f"Recorded metric: user={metrics.user_id}, version={metrics.workflow_version}, "
                f"duration={metrics.total_duration_ms}ms, error={metrics.had_error}"
            )
    
    async def get_version_usage_stats(self) -> Dict[str, Dict]:
        """Get usage statistics by workflow version."""
        async with self._lock:
            stats = {}
            total = sum(self._version_usage.values())
            
            for version, count in self._version_usage.items():
                error_count = sum(self._error_by_version[version].values())
                stats[version] = {
                    "usage_count": count,
                    "percentage": (count / total * 100) if total > 0 else 0,
                    "error_count": error_count,
                    "error_rate": (error_count / count * 100) if count > 0 else 0,
                }
            
            return stats
    
    async def get_node_performance_stats(self) -> Dict[str, Dict]:
        """Get performance statistics per node."""
        async with self._lock:
            stats = {}
            
            for node_name, durations in self._node_times.items():
                if not durations:
                    continue
                
                sorted_times = sorted(durations)
                stats[node_name] = {
                    "execution_count": len(durations),
                    "avg_duration_ms": sum(durations) / len(durations),
                    "min_duration_ms": min(durations),
                    "max_duration_ms": max(durations),
                    "p50_duration_ms": sorted_times[len(sorted_times) // 2],
                    "p95_duration_ms": sorted_times[int(len(sorted_times) * 0.95)] if len(sorted_times) > 1 else sorted_times[0],
                }
            
            return stats
    
    async def get_error_stats(self) -> Dict[str, Dict]:
        """Get error statistics."""
        async with self._lock:
            total_executions = len(self._metrics)
            stats = {
                "total_executions": total_executions,
                "total_errors": sum(self._error_counts.values()),
                "error_rate": (sum(self._error_counts.values()) / total_executions * 100) if total_executions > 0 else 0,
                "errors_by_type": dict(self._error_counts),
                "errors_by_version": {
                    version: dict(errors)
                    for version, errors in self._error_by_version.items()
                },
            }
            return stats
    
    async def get_tool_stats(self) -> Dict[str, Dict]:
        """Get tool execution statistics."""
        async with self._lock:
            tool_counts = defaultdict(int)
            tool_success = defaultdict(int)
            tool_errors = defaultdict(int)
            
            for metric in self._metrics:
                if metric.tool_selected:
                    tool_counts[metric.tool_selected] += 1
                    
                    if metric.had_error:
                        tool_errors[metric.tool_selected] += 1
                    else:
                        tool_success[metric.tool_selected] += 1
            
            stats = {}
            for tool, count in tool_counts.items():
                stats[tool] = {
                    "execution_count": count,
                    "success_count": tool_success[tool],
                    "error_count": tool_errors[tool],
                    "success_rate": (tool_success[tool] / count * 100) if count > 0 else 0,
                }
            
            return stats
    
    async def get_summary(self) -> Dict:
        """Get comprehensive metrics summary."""
        version_stats = await self.get_version_usage_stats()
        node_stats = await self.get_node_performance_stats()
        error_stats = await self.get_error_stats()
        tool_stats = await self.get_tool_stats()
        
        return {
            "version_usage": version_stats,
            "node_performance": node_stats,
            "error_summary": error_stats,
            "tool_performance": tool_stats,
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    async def get_recent_metrics(self, limit: int = 100) -> List[Dict]:
        """Get recent workflow metrics."""
        async with self._lock:
            recent = self._metrics[-limit:]
            return [m.to_dict() for m in recent]
    
    async def get_metrics_by_user(self, user_id: str, limit: int = 50) -> List[Dict]:
        """Get metrics for a specific user."""
        async with self._lock:
            user_metrics = [m for m in self._metrics if m.user_id == user_id]
            return [m.to_dict() for m in user_metrics[-limit:]]
    
    async def get_health_status(self) -> Dict:
        """Get overall health status of workflows."""
        error_stats = await self.get_error_stats()
        version_stats = await self.get_version_usage_stats()
        node_stats = await self.get_node_performance_stats()
        
        # Determine health status
        error_rate = error_stats.get("error_rate", 0)
        avg_duration = sum(
            stats.get("avg_duration_ms", 0) for stats in node_stats.values()
        ) / max(len(node_stats), 1)
        
        if error_rate > 5:
            health = "CRITICAL"
        elif error_rate > 2:
            health = "WARNING"
        else:
            health = "HEALTHY"
        
        slowest_node = None
        if node_stats:
            slowest = max(
                ((k, v.get("avg_duration_ms", 0)) for k, v in node_stats.items()),
                key=lambda x: x[1]
            )
            slowest_node = slowest[0]
        
        return {
            "status": health,
            "error_rate": error_rate,
            "avg_workflow_duration_ms": avg_duration,
            "slowest_node": slowest_node,
            "version_distribution": {k: v["percentage"] for k, v in version_stats.items()},
        }
    
    async def clear_history(self) -> None:
        """Clear all metrics history."""
        async with self._lock:
            self._metrics.clear()
            self._version_usage.clear()
            self._error_counts.clear()
            self._error_by_version.clear()
            self._node_times.clear()
            self.logger.info("Cleared all metrics history")


# Global instance
workflow_metrics = WorkflowMetricsCollector()

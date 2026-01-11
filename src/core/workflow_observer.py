"""
Workflow Observer - Monitors and reports on workflow execution
Implements: Step-by-step execution tracking and callbacks
"""

import logging
import time
from typing import Callable, Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class StepStatus(str, Enum):
    """Status of a workflow step"""
    STARTED = "started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStep:
    """Represents a single workflow step"""
    
    def __init__(self, node_name: str, step_number: int):
        """
        Initialize workflow step.
        
        Args:
            node_name: Name of the workflow node
            step_number: Sequential step number
        """
        self.node_name = node_name
        self.step_number = step_number
        self.status = StepStatus.STARTED
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.duration: Optional[float] = None
        self.output_size = 0
        self.error: Optional[str] = None
        self.metadata: Dict[str, Any] = {}
    
    def complete(self, output_size: int = 0):
        """Mark step as completed"""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.status = StepStatus.COMPLETED
        self.output_size = output_size
    
    def fail(self, error: str):
        """Mark step as failed"""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.status = StepStatus.FAILED
        self.error = error
    
    def skip(self):
        """Mark step as skipped"""
        self.status = StepStatus.SKIPPED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "node_name": self.node_name,
            "step_number": self.step_number,
            "status": self.status.value,
            "duration_ms": round((self.duration or 0) * 1000),
            "output_size": self.output_size,
            "error": self.error,
            "metadata": self.metadata
        }


class WorkflowObserver:
    """
    Monitors workflow execution and provides step-by-step callbacks.
    
    Tracks:
    - Node execution order
    - Execution time per node
    - Success/failure status
    - State changes
    - Performance metrics
    """
    
    def __init__(self, max_steps: int = 100):
        """
        Initialize workflow observer.
        
        Args:
            max_steps: Maximum steps to track
        """
        self.steps: List[WorkflowStep] = []
        self.max_steps = max_steps
        self.callbacks: List[Callable] = []
        self.workflow_start_time: Optional[float] = None
        self.workflow_end_time: Optional[float] = None
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.node_times: Dict[str, List[float]] = {}
        self.node_counts: Dict[str, int] = {}
    
    def reset(self):
        """
        Reset observer state for a new workflow invocation.
        Clears steps but keeps callbacks and performance history.
        """
        self.steps: List[WorkflowStep] = []
        self.workflow_start_time: Optional[float] = None
        self.workflow_end_time: Optional[float] = None
        self.logger.info("[OBSERVER] Reset for new workflow invocation")
    
    def register_callback(self, callback: Callable):
        """
        Register callback for step events.
        
        Args:
            callback: Async callable(step: WorkflowStep)
        """
        self.callbacks.append(callback)
        self.logger.info(f"[OBSERVER] Callback registered. Total callbacks: {len(self.callbacks)}")
    
    async def emit_step_started(
        self,
        node_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> WorkflowStep:
        """
        Emit event when step starts.
        
        Args:
            node_name: Name of the node
            metadata: Optional metadata about the step
            
        Returns:
            WorkflowStep object
        """
        if len(self.steps) >= self.max_steps:
            self.logger.warning(f"Max steps ({self.max_steps}) reached")
            return None
        
        step_number = len(self.steps) + 1
        step = WorkflowStep(node_name, step_number)
        
        if metadata:
            step.metadata.update(metadata)
        
        self.steps.append(step)
        
        if not self.workflow_start_time:
            self.workflow_start_time = time.time()
        
        self.logger.info(f"Step {step_number}: {node_name} started")
        self.logger.info(f"[OBSERVER:CALLBACK-INVOKE] About to call callbacks for STARTED event (step {step_number})")
        await self._emit_callbacks(step)
        self.logger.info(f"[OBSERVER:CALLBACK-INVOKE] Callbacks completed for STARTED event (step {step_number})")
        
        return step
    
    async def emit_step_completed(
        self,
        step: WorkflowStep,
        output_size: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Emit event when step completes.
        
        Args:
            step: WorkflowStep to mark complete
            output_size: Size of output data
            metadata: Additional metadata
        """
        if not step:
            return
        
        step.complete(output_size)
        
        if metadata:
            step.metadata.update(metadata)
        
        # Track performance
        self._track_performance(step.node_name, step.duration)
        
        self.logger.info(
            f"Step {step.step_number}: {step.node_name} completed "
            f"({step.duration:.2f}s, {output_size} bytes)"
        )
        
        self.logger.info(f"[OBSERVER:CALLBACK-INVOKE] About to call callbacks for COMPLETED event (step {step.step_number})")
        self.logger.info(f"[OBSERVER:CALLBACK-INVOKE] Callbacks registered at this moment: {len(self.callbacks)}")
        await self._emit_callbacks(step)
        self.logger.info(f"[OBSERVER:CALLBACK-INVOKE] Callbacks completed for COMPLETED event (step {step.step_number})")
    
    async def emit_step_failed(
        self,
        step: WorkflowStep,
        error: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Emit event when step fails.
        
        Args:
            step: WorkflowStep that failed
            error: Error message
            metadata: Additional metadata
        """
        if not step:
            return
        
        step.fail(error)
        
        if metadata:
            step.metadata.update(metadata)
        
        self.logger.error(
            f"Step {step.step_number}: {step.node_name} failed - {error}"
        )
        
        await self._emit_callbacks(step)
    
    async def emit_step_skipped(
        self,
        node_name: str,
        reason: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Emit event when step is skipped.
        
        Args:
            node_name: Name of skipped node
            reason: Reason for skip
            metadata: Additional metadata
        """
        step_number = len(self.steps) + 1
        step = WorkflowStep(node_name, step_number)
        step.skip()
        step.metadata["skip_reason"] = reason
        
        if metadata:
            step.metadata.update(metadata)
        
        self.steps.append(step)
        
        self.logger.info(f"Step {step_number}: {node_name} skipped ({reason})")
        await self._emit_callbacks(step)
    
    async def emit_workflow_completed(self):
        """Emit event when workflow completes"""
        self.workflow_end_time = time.time()
        total_duration = self.workflow_end_time - (self.workflow_start_time or time.time())
        
        self.logger.info(f"Workflow completed in {total_duration:.2f}s")
    
    async def _emit_callbacks(self, step: WorkflowStep):
        """
        Emit step to all registered callbacks.
        
        Args:
            step: WorkflowStep to emit
        """
        self.logger.info(f"[OBSERVER:EMIT] _emit_callbacks called for step {step.step_number}: {step.node_name}")
        self.logger.info(f"[OBSERVER:EMIT] Total callbacks registered: {len(self.callbacks)}")
        
        if not self.callbacks:
            self.logger.warning(f"[OBSERVER:EMIT] ⚠️ NO CALLBACKS REGISTERED for step {step.step_number}")
            return
            
        self.logger.info(f"[OBSERVER:EMIT] Calling {len(self.callbacks)} callback(s) for step {step.step_number}: {step.node_name}")
        for i, callback in enumerate(self.callbacks):
            try:
                callback_name = callback.__name__ if hasattr(callback, '__name__') else 'unknown'
                self.logger.info(f"[OBSERVER:EMIT]   → Invoking callback {i+1}/{len(self.callbacks)}: {callback_name}")
                if callable(callback):
                    await callback(step)
                    self.logger.info(f"[OBSERVER:EMIT]   ✓ Callback {i+1} completed successfully")
                else:
                    self.logger.warning(f"[OBSERVER:EMIT]   ✗ Callback {i+1} is not callable!")
            except Exception as e:
                self.logger.error(f"[OBSERVER:EMIT] Callback {i+1} failed: {e}", exc_info=True)
    
    def _track_performance(self, node_name: str, duration: float):
        """
        Track performance metrics for a node.
        
        Args:
            node_name: Node name
            duration: Execution duration in seconds
        """
        if node_name not in self.node_times:
            self.node_times[node_name] = []
            self.node_counts[node_name] = 0
        
        self.node_times[node_name].append(duration)
        self.node_counts[node_name] += 1
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary for all nodes.
        
        Returns:
            Dict with performance metrics
        """
        summary = {}
        
        for node_name in self.node_times:
            times = self.node_times[node_name]
            count = self.node_counts[node_name]
            
            summary[node_name] = {
                "count": count,
                "total_time_ms": round(sum(times) * 1000),
                "avg_time_ms": round((sum(times) / count) * 1000),
                "min_time_ms": round(min(times) * 1000),
                "max_time_ms": round(max(times) * 1000)
            }
        
        return summary
    
    def get_execution_trace(self) -> List[Dict[str, Any]]:
        """
        Get complete execution trace.
        
        Returns:
            List of step dictionaries
        """
        return [step.to_dict() for step in self.steps]
    
    def get_workflow_summary(self) -> Dict[str, Any]:
        """
        Get overall workflow summary.
        
        Returns:
            Dict with workflow metrics
        """
        total_duration = 0
        if self.workflow_start_time and self.workflow_end_time:
            total_duration = self.workflow_end_time - self.workflow_start_time
        
        completed = sum(1 for s in self.steps if s.status == StepStatus.COMPLETED)
        failed = sum(1 for s in self.steps if s.status == StepStatus.FAILED)
        skipped = sum(1 for s in self.steps if s.status == StepStatus.SKIPPED)
        
        total_output = sum(s.output_size for s in self.steps)
        
        return {
            "total_steps": len(self.steps),
            "completed": completed,
            "failed": failed,
            "skipped": skipped,
            "total_duration_ms": round(total_duration * 1000),
            "total_output_bytes": total_output,
            "avg_step_duration_ms": round((total_duration / len(self.steps) * 1000) if self.steps else 0),
            "performance_by_node": self.get_performance_summary()
        }
    
    def print_summary(self):
        """Print workflow summary to logger"""
        summary = self.get_workflow_summary()
        
        self.logger.info("=" * 30)
        self.logger.info("WORKFLOW EXECUTION SUMMARY")
        self.logger.info("=" * 30)
        self.logger.info(f"Total Steps: {summary['total_steps']}")
        self.logger.info(f"Completed: {summary['completed']}")
        self.logger.info(f"Failed: {summary['failed']}")
        self.logger.info(f"Skipped: {summary['skipped']}")
        self.logger.info(f"Total Duration: {summary['total_duration_ms']}ms")
        self.logger.info(f"Avg Step Duration: {summary['avg_step_duration_ms']}ms")
        self.logger.info("=" * 30)
        
        # Per-node performance
        perf = summary.get("performance_by_node", {})
        if perf:
            self.logger.info("Performance by Node:")
            for node_name, metrics in perf.items():
                self.logger.info(
                    f"  {node_name}: {metrics['count']} runs, "
                    f"avg {metrics['avg_time_ms']}ms"
                )

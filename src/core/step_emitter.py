"""
Step Emitter - Emits agent steps for real-time display to user via WebSocket
"""

import logging
from datetime import datetime
from typing import Callable, List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class StepEmitter:
    """
    Manages agent step emission for streaming display to user.
    
    Each step represents a major phase in the agent workflow:
    1. CLASSIFY - determine prompt type
    2. REWRITE - clarify ambiguous queries
    3. EXTRACT - process uploaded files
    4. INGEST - embed and store documents
    5. RETRIEVE_PERSONAL - search user's documents
    6. RETRIEVE_GLOBAL - fallback search
    7. FILTER - rank and deduplicate results
    8. SELECT - choose tools or RAG
    9. GENERATE - produce final answer
    10. COMPLETE - finished
    
    Steps are emitted to subscribers (typically WebSocket handlers) for real-time display.
    """
    
    # Step definitions
    STEPS = {
        1: {"title": "🔍 Classifying Query", "description": "Analyzing user input type..."},
        2: {"title": "🔄 Rewriting Query", "description": "Clarifying ambiguous references..."},
        3: {"title": "📤 Extracting Files", "description": "Processing uploaded documents..."},
        4: {"title": "💾 Ingesting Data", "description": "Embedding and storing documents..."},
        5: {"title": "🔎 Searching Personal", "description": "Searching user's documents..."},
        6: {"title": "🌍 Searching Global", "description": "Searching public knowledge base..."},
        7: {"title": "🎯 Filtering Results", "description": "Ranking and deduplicating..."},
        8: {"title": "🛠️  Selecting Tools", "description": "Choosing tools or RAG..."},
        9: {"title": "✍️  Generating Answer", "description": "Creating response..."},
        10: {"title": "✅ Complete", "description": "Done processing"},
    }
    
    def __init__(self, default_callback: Optional[Callable] = None):
        """
        Initialize StepEmitter.
        
        Args:
            default_callback: Optional default callback for step events
        """
        self.subscribers: List[Callable] = []
        self.step_history: List[Dict[str, Any]] = []
        
        if default_callback:
            self.subscribe(default_callback)
    
    def subscribe(self, callback: Callable) -> None:
        """
        Subscribe to step events.
        
        Args:
            callback: Async function(step: dict) to call on step events
        """
        if callback not in self.subscribers:
            self.subscribers.append(callback)
            logger.debug(f"Subscriber added: {callback.__name__ if hasattr(callback, '__name__') else callback}")
    
    def unsubscribe(self, callback: Callable) -> None:
        """Unsubscribe from step events."""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
    
    async def emit_step(self, step: Dict[str, Any]) -> None:
        """
        Emit step event to all subscribers.
        
        Args:
            step: Step dict with keys: number, title, status, result, timestamp
        """
        # Add to history
        self.step_history.append(step)
        
        # Log the step
        logger.info(f"STEP {step.get('number', '?')}: {step.get('title', 'Unknown')} - {step.get('status', 'unknown').upper()}")
        if step.get('result'):
            logger.info(f"  Result: {step['result'][:100]}")
        
        # Emit to subscribers
        for subscriber in self.subscribers:
            try:
                # Handle both sync and async callbacks
                if hasattr(subscriber, '__await__'):
                    await subscriber(step)
                else:
                    await subscriber(step)
            except Exception as e:
                logger.warning(f"Error in step subscriber: {e}")
    
    async def step_started(self, step_number: int, description: str = None, metadata: Dict = None) -> None:
        """
        Called when a step starts processing.
        
        Args:
            step_number: Step number (1-10)
            description: Optional description to override default
            metadata: Optional metadata to include
        """
        step_info = self.STEPS.get(step_number, {"title": "Unknown", "description": ""})
        
        step = {
            "number": step_number,
            "title": step_info.get("title", "Unknown"),
            "description": description or step_info.get("description", "Processing..."),
            "status": "processing",
            "timestamp": datetime.now().isoformat(),
        }
        
        if metadata:
            step.update(metadata)
        
        await self.emit_step(step)
    
    async def step_completed(self, step_number: int, result: str = None, metadata: Dict = None) -> None:
        """
        Called when a step completes.
        
        Args:
            step_number: Step number
            result: Result summary/description
            metadata: Optional metadata
        """
        step_info = self.STEPS.get(step_number, {"title": "Unknown"})
        
        step = {
            "number": step_number,
            "title": step_info.get("title", "Unknown"),
            "status": "complete",
            "result": result or "Done",
            "timestamp": datetime.now().isoformat(),
        }
        
        if metadata:
            step.update(metadata)
        
        await self.emit_step(step)
    
    async def step_error(self, step_number: int, error: str, metadata: Dict = None) -> None:
        """
        Called when a step fails.
        
        Args:
            step_number: Step number
            error: Error message
            metadata: Optional metadata
        """
        step_info = self.STEPS.get(step_number, {"title": "Unknown"})
        
        step = {
            "number": step_number,
            "title": step_info.get("title", "Unknown"),
            "status": "error",
            "error": error,
            "timestamp": datetime.now().isoformat(),
        }
        
        if metadata:
            step.update(metadata)
        
        logger.error(f"STEP {step_number} ERROR: {error}")
        await self.emit_step(step)
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get complete step history."""
        return self.step_history.copy()
    
    def clear_history(self) -> None:
        """Clear step history."""
        self.step_history.clear()


# Global instance
_default_emitter = None


def get_step_emitter(callback: Optional[Callable] = None) -> StepEmitter:
    """
    Get or create global step emitter.
    
    Args:
        callback: Optional callback to subscribe to
        
    Returns:
        StepEmitter instance
    """
    global _default_emitter
    
    if _default_emitter is None:
        _default_emitter = StepEmitter(default_callback=callback)
    elif callback:
        _default_emitter.subscribe(callback)
    
    return _default_emitter


def reset_step_emitter() -> None:
    """Reset global emitter (for testing)."""
    global _default_emitter
    _default_emitter = None

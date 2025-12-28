"""
Workflow Observer and Step Streaming Integration
=================================================

This module demonstrates how to integrate detailed workflow step tracking
and real-time streaming to the frontend.

The LangGraph V4 workflow has 13-17 nodes depending on execution path.
Each node completion should emit a detailed step update to the frontend.

Usage in FastAPI app.py:
```python
from .core.workflow_observer import WorkflowObserver

# In chat endpoint
observer = WorkflowObserver() if settings.WORKFLOW_OBSERVER_ENABLED else None

async def generate():
    async for event in workflow.astream(state, config={"callbacks": [observer]}):
        # Stream workflow step updates
        for step in observer.get_completed_steps():
            yield f'data: {json.dumps({"type": "workflow_step", "step": step})}\n\n'
```
"""

# Step status constants
STEP_STATUS_PENDING = "pending"
STEP_STATUS_IN_PROGRESS = "in-progress"
STEP_STATUS_COMPLETED = "completed"
STEP_STATUS_ERROR = "error"

# Workflow node to step mapping (V4 13-node architecture)
WORKFLOW_NODE_MAPPING = {
    # Entry points
    "prompt_handler": {
        "title": "📥 Input Processing",
        "phase": "input",
        "description": "Analyzing your input...",
        "details": "Determining whether this is a prompt, file upload, or conversation continuation",
    },
    
    "file_handler": {
        "title": "📄 File Detection",
        "phase": "input",
        "description": "Processing uploaded files...",
        "details": "Checking for files in your message",
    },

    "classify": {
        "title": "🔍 Query Classification",
        "phase": "classification",
        "description": "Classifying your query...",
        "details": "Understanding whether your query is a greeting, chitchat, or financial question",
    },

    "direct_response": {
        "title": "💬 Conversational Response",
        "phase": "classification",
        "description": "Generating response...",
        "details": "Responding to your message without tools or document analysis",
    },

    # File processing
    "extract_file": {
        "title": "🔄 File Extraction",
        "phase": "file_processing",
        "description": "Extracting text from files...",
        "details": "Converting PDF/DOCX/Images to readable text using OCR and extraction tools",
    },

    "ingest_file": {
        "title": "⚙️ Document Processing",
        "phase": "file_processing",
        "description": "Processing documents...",
        "details": "Creating 2 structural + 9 metric-centric chunks. Generating embeddings and storing in Qdrant",
    },

    # Query enhancement
    "rewrite_eval": {
        "title": "✏️ Rewrite Evaluation",
        "phase": "enhancement",
        "description": "Evaluating if query needs enhancement...",
        "details": "Checking if your query should be rewritten with file or conversation context",
    },

    "rewrite_file": {
        "title": "📝 File Context Injection",
        "phase": "enhancement",
        "description": "Adding file context to query...",
        "details": "Enhancing your question with information about uploaded files",
    },

    "rewrite_convo": {
        "title": "💭 Conversation Context Injection",
        "phase": "enhancement",
        "description": "Adding conversation context...",
        "details": "Enriching your query with relevant information from conversation history",
    },

    # RAG pipeline
    "retrieve": {
        "title": "🗂️ RAG Retrieval",
        "phase": "retrieval",
        "description": "Searching documents...",
        "details": "Using dual retrieval strategy: metadata-only for generic queries, RRF for specific questions. Embedding dimension: 384, Threshold: 0.1, Top-K: 20",
    },

    "filter": {
        "title": "🎯 Result Filtering",
        "phase": "retrieval",
        "description": "Filtering & ranking results...",
        "details": "Deduplicating chunks and ranking with RRF (Reciprocal Rank Fusion) using K=60",
    },

    "analyze": {
        "title": "📊 Data Type Analysis",
        "phase": "retrieval",
        "description": "Analyzing data types...",
        "details": "Detecting tables, numeric data, text content, and structured information",
    },

    # Tool & processing
    "select_tools": {
        "title": "🔧 Tool Selection",
        "phase": "tools",
        "description": "Selecting appropriate tools...",
        "details": "Determining which financial tools (company info, stock prices, technical analysis) are needed based on query classification",
    },

    "execute_tools": {
        "title": "⚡ Tool Execution",
        "phase": "tools",
        "description": "Executing tools...",
        "details": "Running selected tools to gather financial data: company info, stock prices, SMA, RSI calculations, etc.",
    },

    # Synthesis
    "summary_tools": {
        "title": "📈 Summary Synthesis",
        "phase": "synthesis",
        "description": "Synthesizing summary...",
        "details": "Using specialized techniques: comparative analysis, anomaly detection, materiality-weighted, narrative arc, key questions",
    },

    "query_reformulation": {
        "title": "🧩 Context Assembly",
        "phase": "synthesis",
        "description": "Assembling context...",
        "details": "Combining RAG results, tool outputs, and summaries into structured context for LLM. Vietnamese formatting with proper citations",
    },

    # Final output
    "generate": {
        "title": "✨ Answer Generation",
        "phase": "generation",
        "description": "Generating final answer...",
        "details": "LLM synthesizing comprehensive response from all available context sources",
    },
}


def create_workflow_step(node_name, status=STEP_STATUS_PENDING, result=None, error=None, metadata=None, duration=None):
    """
    Create a workflow step object for streaming to frontend.
    
    Args:
        node_name: Name of workflow node (e.g., "retrieve")
        status: Step status (pending, in-progress, completed, error)
        result: Result message (e.g., "Retrieved 5 documents")
        error: Error message if status is error
        metadata: Optional metadata dict with additional info
        duration: Execution duration in ms
        
    Returns:
        Dict with step information for frontend
    """
    node_config = WORKFLOW_NODE_MAPPING.get(node_name, {})
    
    return {
        "id": node_name,
        "order": len([n for n in WORKFLOW_NODE_MAPPING.keys() if n <= node_name]),
        "title": node_config.get("title", f"Step: {node_name}"),
        "icon": node_config.get("title", "").split()[0] if node_config.get("title") else "⚙️",
        "phase": node_config.get("phase", "unknown"),
        "description": node_config.get("description", "Processing..."),
        "details": node_config.get("details", ""),
        "status": status,
        "result": result,
        "error": error,
        "metadata": metadata or {},
        "duration": duration,
        "color": get_phase_color(node_config.get("phase", "unknown")),
    }


def get_phase_color(phase):
    """Get color coding for workflow phase."""
    color_map = {
        "input": "slate",
        "classification": "blue",
        "file_processing": "purple",
        "enhancement": "amber",
        "retrieval": "emerald",
        "tools": "purple",
        "synthesis": "cyan",
        "generation": "emerald",
    }
    return color_map.get(phase, "slate")


# Example integration in FastAPI endpoint
"""
# In src/api/app.py chat endpoint:

async def chat_stream_with_workflow_steps():
    '''Stream chat response with detailed workflow steps'''
    
    # Initialize observer for workflow tracking
    workflow_steps = []
    
    # Helper function to emit step updates
    async def emit_step(node_name, status, result=None, error=None, metadata=None, duration=None):
        step = create_workflow_step(node_name, status, result, error, metadata, duration)
        workflow_steps.append(step)
        yield f'data: {json.dumps({"type": "workflow_step", "step": step})\n\n'
    
    try:
        # Execute workflow with step tracking
        state = {
            "user_prompt": request.question,
            "user_id": user_id,
            "session_id": session_id,
            "conversation_history": conversation_history,
            # ... other state
        }
        
        # Define callback for workflow node completion
        class WorkflowCallback:
            async def on_node_start(self, node_name):
                async for msg in emit_step(node_name, STEP_STATUS_IN_PROGRESS):
                    yield msg
                    
            async def on_node_end(self, node_name, output, duration):
                # Extract result from output
                result_msg = extract_result_message(output)
                async for msg in emit_step(node_name, STEP_STATUS_COMPLETED, result_msg, duration=duration):
                    yield msg
                    
            async def on_node_error(self, node_name, error):
                async for msg in emit_step(node_name, STEP_STATUS_ERROR, error=str(error)):
                    yield msg
        
        # Run workflow
        callback = WorkflowCallback()
        answer, thinking_steps = await agent_instance.aquery(
            request.question,
            user_id=user_id,
            session_id=session_id,
            conversation_history=conversation_history,
            use_rag=should_use_rag,
            workflow_callback=callback,
        )
        
        # Emit final answer
        yield f'data: {json.dumps({"type": "answer", "content": answer})\n\n'
        
    except Exception as e:
        yield f'data: {json.dumps({"type": "error", "message": str(e)})\n\n'
"""

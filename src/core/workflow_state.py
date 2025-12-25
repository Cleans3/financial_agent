"""
WorkflowState - TypedDict defining the state for the LangGraph workflow
Based on langgraph_project_structure.md specification
"""

from typing import TypedDict, Optional, List, Dict, Any
from enum import Enum


class PromptType(str, Enum):
    """Classification of user input type"""
    CHITCHAT = "chitchat"                    # Hello, hi, thanks, bye
    REQUEST = "request"                      # Find, get, search, show
    INSTRUCTION = "instruction"              # Other questions
    AMBIGUOUS = "ambiguous"                  # Unclear references, missing context
    FILE_ONLY = "file_only"                  # No prompt, only files
    PROMPT_AND_FILE = "prompt_and_file"      # Both prompt and files


class EmbeddingMethod(str, Enum):
    """Method for embedding file data"""
    SINGLE_DENSE = "single_dense"            # <5KB
    MULTIDIMENSIONAL = "multidimensional"    # <50KB
    HIERARCHICAL = "hierarchical"            # >50KB


class DataType(str, Enum):
    """Data types detected in search results"""
    TABLE = "table"                          # Structured table data
    NUMERIC = "numeric"                      # Numeric/calculation data
    TEXT = "text"                            # Prose/explanatory text
    MIXED = "mixed"                          # Multiple data types


class RetrievalStrategy(str, Enum):
    """Strategy for RAG retrieval"""
    PERSONAL_ONLY = "personal_only"          # Search personal vectordb only
    PERSONAL_WITH_FALLBACK = "personal_with_fallback"  # Personal first, then global
    DUAL = "dual"                            # Search both simultaneously


class WorkflowState(TypedDict):
    """
    State dictionary for the 10-node LangGraph workflow.
    
    Flows through:
    CLASSIFY → (CHITCHAT_HANDLER | REWRITE_PROMPT | EXTRACT_DATA) 
    → ... → FILTER_SEARCH → SELECT_TOOLS → GENERATE_ANSWER → END
    """
    
    # ===== CONTEXT INFO =====
    user_id: Optional[str]                  # User identifier
    session_id: Optional[str]               # Session identifier
    
    # ===== INPUT FIELDS =====
    user_prompt: Optional[str]              # Original user question
    uploaded_files: Optional[List[Dict]]    # Files uploaded with metadata
    conversation_history: List[Dict]        # Previous messages in conversation
    
    # ===== CLASSIFICATION PHASE (CLASSIFY node) =====
    prompt_type: Optional[PromptType]       # CHITCHAT/REQUEST/INSTRUCTION/AMBIGUOUS/FILE_ONLY/PROMPT_AND_FILE
    needs_file_processing: bool             # Whether files need extraction
    is_chitchat: bool                       # Flag set by PROMPT_HANDLER for chitchat queries
    
    # ===== REWRITING PHASE (REWRITE_EVALUATION & REWRITE_* nodes) =====
    needs_rewrite: bool                     # Whether query needs rewriting
    rewrite_context_type: Optional[str]     # "file_context" or "conversation_context"
    rewritten_prompt: Optional[str]         # Disambiguated query using context
    
    # ===== FILE EXTRACTION PHASE (EXTRACT_DATA node) =====
    extracted_file_data: Optional[Dict]     # Structured data from files
    file_metadata: List[Dict]               # File metadata from extraction (path, name, type, size)
    
    # ===== FILE INGESTION PHASE (INGEST_FILE node) =====
    embedding_method: Optional[EmbeddingMethod]  # Selected embedding method
    ingested_file_ids: List[str]                 # IDs of ingested files
    files_ingested: bool                    # Whether files were successfully ingested
    ingested_chunks: int                    # Number of chunks ingested
    
    # ===== RETRIEVAL PHASE (RETRIEVE_PERSONAL & RETRIEVE_GLOBAL nodes) =====
    personal_semantic_results: List[Dict]   # Semantic search results from personal vectordb
    personal_keyword_results: List[Dict]    # Keyword search results from personal vectordb
    global_semantic_results: List[Dict]     # Semantic search results from global vectordb (fallback)
    global_keyword_results: List[Dict]      # Keyword search results from global vectordb (fallback)
    rag_enabled: bool                       # Whether RAG retrieval is enabled (from RETRIEVE_OR_GENERATE)
    
    # ===== FILTERING PHASE (FILTER_AND_RANK node) =====
    best_search_results: List[Dict]         # Ranked, deduplicated, top results from RRF
    search_metadata: Dict[str, Any]         # RRF scores, ranking info, fallback sources used
    
    # ===== ANALYSIS PHASE (ANALYZE_RETRIEVED_RESULTS node) =====
    has_table_data: bool                    # Whether results contain table data
    has_numeric_data: bool                  # Whether results contain numeric/calculation data
    text_only: bool                         # Whether results are text-only
    detected_data_types: List[DataType]     # Specific data types detected
    
    # ===== TOOL SELECTION PHASE (TOOL_SELECTION node) =====
    selected_tools: List[str]               # List of tools to use (e.g., ['calculator', 'table_processor'])
    
    # ===== FINAL GENERATION PHASE (GENERATE_ANSWER node) =====
    generated_answer: str                   # Final response to user
    
    # ===== TOOL EXECUTION PHASE (EXECUTE_TOOLS node) =====
    tool_results: Dict[str, Any]            # Tool execution results {tool_name: result}
    combined_tool_output: str                # Formatted combined output from all tools
    
    # ===== OUTPUT FORMATTING PHASE (FORMAT_OUTPUT node) =====
    formatted_answer: str                   # Final formatted response with tables, calculations
    
    # ===== METADATA =====
    metadata: Dict[str, Any]                # Tracking info (retrieval_used, tools_used, etc.)


def create_initial_state(
    user_prompt: Optional[str] = None,
    uploaded_files: Optional[List[Dict]] = None,
    conversation_history: Optional[List[Dict]] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None
) -> WorkflowState:
    """
    Create initial WorkflowState for a new query.
    
    Args:
        user_prompt: User's question
        uploaded_files: Files uploaded by user
        conversation_history: Previous messages
        user_id: User identifier
        session_id: Session identifier
        
    Returns:
        Initialized WorkflowState ready for CLASSIFY node
    """
    return WorkflowState(
        # Context
        user_id=user_id,
        session_id=session_id,
        
        # Input
        user_prompt=user_prompt,
        uploaded_files=uploaded_files or [],
        conversation_history=conversation_history or [],
        
        # Classification
        prompt_type=None,
        needs_file_processing=False,
        is_chitchat=False,
        
        # Rewriting
        needs_rewrite=False,
        rewrite_context_type=None,
        rewritten_prompt=None,
        
        # File Extraction
        extracted_file_data=None,
        file_metadata=[],
        
        # File Ingestion
        embedding_method=None,
        ingested_file_ids=[],
        files_ingested=False,
        ingested_chunks=0,
        
        # Retrieval
        personal_semantic_results=[],
        personal_keyword_results=[],
        global_semantic_results=[],
        global_keyword_results=[],
        rag_enabled=False,
        
        # Filtering
        best_search_results=[],
        search_metadata={},
        
        # Analysis
        has_table_data=False,
        has_numeric_data=False,
        text_only=False,
        detected_data_types=[],
        
        # Tool Selection
        selected_tools=[],
        
        # Tool Execution
        tool_results={},
        combined_tool_output="",
        
        # Output Formatting
        formatted_answer="",
        
        # Final Answer
        generated_answer="",
        
        # Metadata
        metadata={
            "retrieval_used": False,
            "tools_used": [],
            "embedding_method_selected": None,
            "search_results_found": 0,
            "data_types_detected": [],
            "rag_strategy_used": None,
            "fallback_to_global": False
        }
    )

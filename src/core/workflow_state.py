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
    FILE_ONLY = "file_only"                  # No prompt, only files
    PROMPT_AND_FILE = "prompt_and_file"      # Both prompt and files


class EmbeddingMethod(str, Enum):
    """Method for embedding file data"""
    SINGLE_DENSE = "single_dense"            # <5KB
    MULTIDIMENSIONAL = "multidimensional"    # <50KB
    HIERARCHICAL = "hierarchical"            # >50KB


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
    prompt_type: Optional[PromptType]       # CHITCHAT/REQUEST/INSTRUCTION/FILE_ONLY/PROMPT_AND_FILE
    needs_file_processing: bool             # Whether files need extraction
    
    # ===== REWRITING PHASE (REWRITE_PROMPT node) =====
    rewritten_prompt: Optional[str]         # Disambiguated query using context
    
    # ===== FILE EXTRACTION PHASE (EXTRACT_DATA node) =====
    extracted_file_data: Optional[Dict]     # Structured data from files
    
    # ===== FILE INGESTION PHASE (INGEST_FILE node) =====
    embedding_method: Optional[EmbeddingMethod]  # Selected embedding method
    ingested_file_ids: List[str]                 # IDs of ingested files
    
    # ===== RETRIEVAL PHASE (RETRIEVE_PERSONAL & RETRIEVE_GLOBAL nodes) =====
    personal_semantic_results: List[Dict]   # Semantic search results from personal vectordb
    personal_keyword_results: List[Dict]    # Keyword search results from personal vectordb
    global_semantic_results: List[Dict]     # Semantic search results from global vectordb
    global_keyword_results: List[Dict]      # Keyword search results from global vectordb
    
    # ===== FILTERING PHASE (FILTER_SEARCH node) =====
    best_search_results: List[Dict]         # Ranked, deduplicated, top results from RRF
    
    # ===== TOOL SELECTION PHASE (SELECT_TOOLS node) =====
    selected_tools: List[str]               # List of tools to use (e.g., ['calculator', 'table_processor'])
    
    # ===== FINAL GENERATION PHASE (GENERATE_ANSWER node) =====
    generated_answer: str                   # Final response to user
    
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
        
        # Rewriting
        rewritten_prompt=None,
        
        # File Extraction
        extracted_file_data=None,
        
        # File Ingestion
        embedding_method=None,
        ingested_file_ids=[],
        
        # Retrieval
        personal_semantic_results=[],
        personal_keyword_results=[],
        global_semantic_results=[],
        global_keyword_results=[],
        
        # Filtering
        best_search_results=[],
        
        # Tool Selection
        selected_tools=[],
        
        # Final Answer
        generated_answer="",
        
        # Metadata
        metadata={
            "retrieval_used": False,
            "tools_used": [],
            "embedding_method_selected": None,
            "search_results_found": 0
        }
    )

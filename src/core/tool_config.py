"""Tool registry and behavior configuration for financial agent."""

from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class ToolRegistry(str, Enum):
    """Available tool categories."""
    VNSTOCK = "vnstock_tools"
    TECHNICAL = "technical_tools"


class QueryRewriteConfig(BaseModel):
    """Configuration for query rewriting pipeline step."""
    enabled: bool = True
    ambiguous_words: List[str] = Field(
        default=[
            "it", "this", "that", "it's", "they", "them",
            "those", "these", "nó", "cái này", "cái kia"
        ]
    )
    context_depth: int = 2


class RAGFilterConfig(BaseModel):
    """Configuration for RAG document filtering."""
    enabled: bool = True
    min_relevance_threshold: float = 0.3
    max_documents: int = 5
    similarity_metric: str = "bm25"


class SummarizationConfig(BaseModel):
    """Configuration for result summarization."""
    enabled: bool = True
    length_threshold: int = 500
    max_summary_length: int = 200
    language: str = "vi"


class ToolsConfig(BaseModel):
    """Master configuration for tools and agent pipeline behavior."""

    # Tool Registry Control - only query-time tools (file processing handled by pipeline)
    enabled_tools: List[ToolRegistry] = Field(
        default=[
            ToolRegistry.VNSTOCK,
            ToolRegistry.TECHNICAL,
        ]
    )

    # Feature Flags
    allow_tool_calls: bool = True
    allow_rag: bool = True

    # Sub-configs for pipeline steps
    query_rewrite: QueryRewriteConfig = Field(default_factory=QueryRewriteConfig)
    rag_filter: RAGFilterConfig = Field(default_factory=RAGFilterConfig)
    summarization: SummarizationConfig = Field(default_factory=SummarizationConfig)

    # Response Validation
    validate_responses: bool = True
    max_tool_result_length: int = 8000

    # Logging
    log_tool_calls: bool = True
    log_reasoning: bool = True

    class Config:
        use_enum_values = True


# Default instance with standard settings
DEFAULT_TOOLS_CONFIG = ToolsConfig()

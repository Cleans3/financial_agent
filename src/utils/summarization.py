import re
from typing import Optional, List, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.language_models import BaseLLM


async def summarize_messages(messages: List[BaseMessage], llm: BaseLLM, num_messages_to_compress: int = 5) -> str:
    """Compress multiple messages into 1-2 bullet points.
    
    Args:
        messages: List of messages to summarize (e.g., last 5 messages before current)
        llm: Language model to use for summarization
        num_messages_to_compress: How many messages to target for compression
    
    Returns:
        Markdown bullet point summary
    """
    if len(messages) <= 2:
        return None
    
    msg_text = "\n".join([f"{msg.type}: {msg.content}" for msg in messages[-num_messages_to_compress:]])
    
    prompt = f"""Summarize this conversation exchange into 1-2 concise bullet points. Focus on key questions, decisions, and context.

Conversation:
{msg_text}

Format: Use markdown bullet points (- item). Be very concise."""
    
    response = await llm.ainvoke(prompt)
    return response.content


def summarize_tool_result(result: Dict[str, Any], llm: BaseLLM) -> Optional[str]:
    """Generate 1-sentence summary for tool results >500 chars.
    
    Args:
        result: Tool result dictionary
        llm: Language model to use for summarization
    
    Returns:
        1-sentence tl;dr or None if too short
    """
    result_str = str(result)
    if len(result_str) <= 500:
        return None
    
    prompt = f"""Summarize this financial tool result in exactly 1 sentence. Focus on the most important finding.

Result:
{result_str[:1000]}...

Provide ONLY the 1-sentence summary, nothing else."""
    
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return None


def extract_financial_metrics(text: str) -> Dict[str, Any]:
    """Extract financial metrics and KPIs from text using regex patterns.
    
    Args:
        text: Document or result text to extract from
    
    Returns:
        Dictionary with extracted metrics
    """
    metrics = {}
    
    # Common financial patterns
    patterns = {
        'revenue': r'(?:revenue|sales?|turnover)[\s:]*(?:VND|USD|₫|\$)?\s*([\d,\.]+)\s*(?:billion|million|bn|mn)',
        'profit': r'(?:net profit|earnings|income)[\s:]*(?:VND|USD|₫|\$)?\s*([\d,\.]+)\s*(?:billion|million|bn|mn)',
        'growth': r'(?:growth|increase|rise)[\s:]*([+-]?\d+\.?\d*)%',
        'price': r'(?:price|stock price)[\s:]*(?:VND|₫)?\s*([\d,\.]+)',
        'date': r'(?:Q[1-4]\s*\d{4}|\d{1,2}\/\d{1,2}\/\d{4}|\d{4}-\d{2}-\d{2})',
    }
    
    for key, pattern in patterns.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            metrics[key] = matches[:3]  # Keep top 3 matches
    
    return metrics


def create_enhanced_tool_result(
    data: Any,
    tool_name: str,
    llm: BaseLLM,
    reasoning: str = "",
    raw_result: Optional[Dict] = None
) -> Dict[str, Any]:
    """Create enhanced tool result with reasoning and summary.
    
    Args:
        data: Main tool result data
        tool_name: Name of the tool called
        llm: Language model to use for summarization
        reasoning: Why was this tool called, what assumptions
        raw_result: Original full result dict
    
    Returns:
        Enhanced result dict with summary and context
    """
    result_str = str(data)
    summary = summarize_tool_result(raw_result or {"data": data}, llm)
    metrics = extract_financial_metrics(result_str) if len(result_str) > 300 else {}
    
    return {
        "data": data,
        "tool": tool_name,
        "reasoning": reasoning,
        "summary": summary,
        "metrics": metrics,
        "full_context": raw_result or {"data": data}
    }


def create_rag_summary(document_text: str, relevance_score: float) -> Dict[str, Any]:
    """Create extractive summary for RAG document.
    
    Args:
        document_text: Full document text
        relevance_score: Cosine similarity score
    
    Returns:
        Dict with full text, metrics, and summary
    """
    metrics = extract_financial_metrics(document_text)
    summary = None
    
    if len(document_text) > 2000:
        sentences = document_text.split('. ')[:5]  # First 5 sentences
        summary = '. '.join(sentences) + '.'
    
    return {
        "full_text": document_text,
        "metrics_summary": metrics,
        "summary": summary,
        "relevance_score": relevance_score,
        "length": len(document_text)
    }


def estimate_message_tokens(messages: List[BaseMessage]) -> int:
    """Rough estimate of token count (4 chars ≈ 1 token).
    
    Args:
        messages: List of messages
    
    Returns:
        Estimated token count
    """
    total_chars = sum(len(str(msg.content)) for msg in messages)
    return total_chars // 4


async def should_compress_history(messages: List[BaseMessage], context_limit: int = 6000) -> bool:
    """Check if conversation should be compressed.
    
    Args:
        messages: Current message list
        context_limit: Token limit (default 6000)
    
    Returns:
        True if compression recommended
    """
    return estimate_message_tokens(messages) > context_limit * 0.8

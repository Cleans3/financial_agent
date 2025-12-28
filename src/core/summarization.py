from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from enum import Enum
from src.llm.llm_factory import LLMFactory
from src.core.config import settings
import re
import logging
from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)


class SummarizationMode(str, Enum):
    ALWAYS = "always"
    ON_DEMAND = "on-demand"
    NEVER = "never"


class SummarizationStrategy(ABC):
    @abstractmethod
    def summarize(self, text: str, context: Optional[Dict] = None) -> str:
        pass
    
    @abstractmethod
    def should_summarize(self, text: str, explicit_request: bool = False) -> bool:
        pass


class ExtractiveMetricsSummarization(SummarizationStrategy):
    METRIC_PATTERNS = {
        "revenue": r"revenue[:\s]+\$?([\d,.]+)(?:\s+(?:billion|million|thousand|b|m|k))?",
        "net_income": r"net\s+income[:\s]+\$?([\d,.]+)",
        "eps": r"(?:eps|earnings\s+per\s+share)[:\s]+\$?([\d,.]+)",
        "roa": r"roa[:\s]+([0-9.]+)%?",
        "roe": r"roe[:\s]+([0-9.]+)%?",
        "asset": r"(?:total\s+)?assets?[:\s]+\$?([\d,.]+)",
        "equity": r"(?:total\s+)?equity[:\s]+\$?([\d,.]+)",
        "debt": r"(?:total\s+)?debt[:\s]+\$?([\d,.]+)",
    }
    
    def extract_metrics(self, text: str) -> Dict[str, List[str]]:
        metrics = {}
        for metric_name, pattern in self.METRIC_PATTERNS.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                metrics[metric_name] = matches
        logger.debug(f"[SUMMARIZE] Extractive: Found {len(metrics)} metric types with {sum(len(v) for v in metrics.values())} total matches")
        return metrics
    
    def summarize(self, text: str, context: Optional[Dict] = None) -> str:
        logger.info(f"[SUMMARIZE] ExtractiveMetrics: Starting summarization for {len(text)} chars")
        metrics = self.extract_metrics(text)
        if not metrics:
            logger.debug(f"[SUMMARIZE] ExtractiveMetrics: No metrics found in text")
            return ""
        
        summary_parts = ["📊 Key Metrics:"]
        for metric_name, values in metrics.items():
            if values:
                summary_parts.append(f"• {metric_name.replace('_', ' ').title()}: {', '.join(values[:2])}")
        
        result = "\n".join(summary_parts)
        logger.info(f"[SUMMARIZE] ExtractiveMetrics: ✓ Generated summary with {len(metrics)} metrics")
        return result
    
    def should_summarize(self, text: str, explicit_request: bool = False) -> bool:
        if explicit_request:
            return True
        return len(text) > 500 and bool(self.extract_metrics(text))


class ComparativeAnalysisSummarization(SummarizationStrategy):
    def __init__(self):
        self.llm = LLMFactory.get_llm()
    
    def summarize(self, text: str, context: Optional[Dict] = None) -> str:
        logger.info(f"[SUMMARIZE] ComparativeAnalysis: Starting comparison for {len(text)} chars")
        previous_data = context.get("previous_data", "") if context else ""
        
        if not previous_data:
            logger.debug(f"[SUMMARIZE] ComparativeAnalysis: No previous data in context, skipping")
            return ""
        
        prompt = f"""Compare the following financial data and highlight key changes:

Previous Period:
{previous_data}

Current Period:
{text}

Provide a 2-3 sentence analysis of major changes (YoY comparisons, trends). Format: key metrics → % change."""
        
        try:
            response = self.llm.invoke(prompt)
            result = response.content if hasattr(response, 'content') else str(response)
            logger.info(f"[SUMMARIZE] ComparativeAnalysis: ✓ Generated comparison analysis")
            return result
        except Exception as e:
            logger.warning(f"[SUMMARIZE] ComparativeAnalysis: Failed - {e}")
            return ""
    
    def should_summarize(self, text: str, explicit_request: bool = False) -> bool:
        return explicit_request or "compared" in text.lower() or "vs" in text.lower()


class RiskFocusedSummarization(SummarizationStrategy):
    RISK_KEYWORDS = [
        "debt", "leverage", "default", "loss", "volatility", "risk",
        "impairment", "provision", "charge", "writedown", "liquidity",
        "crisis", "downgrade", "stress", "exposure"
    ]
    
    def summarize(self, text: str, context: Optional[Dict] = None) -> str:
        logger.info(f"[SUMMARIZE] RiskFocused: Analyzing {len(text)} chars for risk keywords")
        risk_sentences = []
        for sentence in text.split('.'):
            sentence_lower = sentence.lower()
            if any(kw in sentence_lower for kw in self.RISK_KEYWORDS):
                risk_sentences.append(sentence.strip())
        
        if not risk_sentences:
            logger.debug(f"[SUMMARIZE] RiskFocused: No risk keywords found")
            return ""
        
        summary_parts = ["⚠️ Risk Highlights:"]
        for sent in risk_sentences[:3]:
            if sent:
                summary_parts.append(f"• {sent}")
        
        result = "\n".join(summary_parts)
        logger.info(f"[SUMMARIZE] RiskFocused: ✓ Found {len(risk_sentences)} risk-related sentences")
        return result
    
    def should_summarize(self, text: str, explicit_request: bool = False) -> bool:
        if explicit_request:
            return True
        return any(kw in text.lower() for kw in self.RISK_KEYWORDS)


class AnomalyDetectionSummarization(SummarizationStrategy):
    def __init__(self):
        self.llm = LLMFactory.get_llm()
    
    def summarize(self, text: str, context: Optional[Dict] = None) -> str:
        logger.info(f"[SUMMARIZE] AnomalyDetection: Analyzing {len(text)} chars for anomalies")
        historical_data = context.get("historical_avg", "") if context else ""
        
        prompt = f"""Identify anomalies or unusual patterns in this financial data:

Data:
{text}

Historical Reference (if any):
{historical_data}

List 2-3 unusual findings (unexpected changes, outliers). Format: • Anomaly: explanation"""
        
        try:
            response = self.llm.invoke(prompt)
            result = response.content if hasattr(response, 'content') else str(response)
            logger.info(f"[SUMMARIZE] AnomalyDetection: ✓ Generated anomaly analysis")
            return result
        except Exception as e:
            logger.warning(f"[SUMMARIZE] AnomalyDetection: Failed - {e}")
            return ""
    
    def should_summarize(self, text: str, explicit_request: bool = False) -> bool:
        return explicit_request or any(kw in text.lower() for kw in ["unusual", "anomaly", "unexpected"])


class HybridSummarization(SummarizationStrategy):
    def __init__(self):
        self.extractive = ExtractiveMetricsSummarization()
        self.comparative = ComparativeAnalysisSummarization()
        self.risk = RiskFocusedSummarization()
        self.anomaly = AnomalyDetectionSummarization()
    
    def summarize(self, text: str, context: Optional[Dict] = None) -> str:
        logger.info(f"[SUMMARIZE] Hybrid: Starting multi-strategy summarization for {len(text)} chars")
        summaries = []
        
        if self.extractive.should_summarize(text, False):
            metrics_summary = self.extractive.summarize(text, context)
            if metrics_summary:
                summaries.append(metrics_summary)
        
        if self.risk.should_summarize(text, False):
            risk_summary = self.risk.summarize(text, context)
            if risk_summary:
                summaries.append(risk_summary)
        
        if context and "previous_data" in context:
            comp_summary = self.comparative.summarize(text, context)
            if comp_summary:
                summaries.append(f"📈 Comparison: {comp_summary}")
        
        result = "\n\n".join(summaries)
        logger.info(f"[SUMMARIZE] Hybrid: ✓ Combined {len(summaries)} summary components")
        return result
    
    def should_summarize(self, text: str, explicit_request: bool = False) -> bool:
        return explicit_request or len(text) > 500


def get_summarization_strategy(strategy: str = None) -> SummarizationStrategy:
    strategy = strategy or settings.SUMMARIZE_MODE
    logger.info(f"[SUMMARIZE] Strategy selected: {strategy}")
    
    if strategy == "always":
        logger.debug(f"[SUMMARIZE] Using HybridSummarization (always mode)")
        return HybridSummarization()
    elif strategy == "on-demand":
        logger.debug(f"[SUMMARIZE] Using HybridSummarization (on-demand mode)")
        return HybridSummarization()
    else:
        logger.debug(f"[SUMMARIZE] Using ExtractiveMetricsSummarization (never/fallback mode)")
        return ExtractiveMetricsSummarization()


def should_summarize_response(text: str, mode: str = None) -> bool:
    mode = mode or settings.SUMMARIZE_MODE
    text_len = len(text)
    
    if mode == "never":
        logger.debug(f"[SUMMARIZE] Mode=never: Summarization disabled")
        return False
    elif mode == "always":
        should_summarize = text_len > 500
        logger.debug(f"[SUMMARIZE] Mode=always: {text_len} chars, should_summarize={should_summarize}")
        return should_summarize
    else:
        logger.debug(f"[SUMMARIZE] Mode={mode}: Summarization disabled")
        return False


# ============== UTILITY FUNCTIONS (merged from utils/summarization.py) ==============

async def summarize_messages(messages: List[BaseMessage], llm, num_messages_to_compress: int = 5) -> Optional[str]:
    """Compress multiple messages into 1-2 bullet points."""
    logger.info(f"[SUMMARIZE] Messages: Compressing {len(messages)} messages (compress last {num_messages_to_compress})")
    if len(messages) <= 2:
        logger.debug(f"[SUMMARIZE] Messages: Skipped (only {len(messages)} messages)")
        return None
    
    msg_text = "\n".join([f"{msg.type}: {msg.content}" for msg in messages[-num_messages_to_compress:]])
    
    prompt = f"""Summarize this conversation exchange into 1-2 concise bullet points. Focus on key questions, decisions, and context.

Conversation:
{msg_text}

Format: Use markdown bullet points (- item). Be very concise."""
    
    response = await llm.ainvoke(prompt)
    logger.info(f"[SUMMARIZE] Messages: ✓ Compressed to summary")
    return response.content


def summarize_tool_result(result: Dict[str, Any], llm) -> Optional[str]:
    """Generate 1-sentence summary for tool results >500 chars."""
    result_str = str(result)
    if len(result_str) <= 500:
        logger.debug(f"[SUMMARIZE] ToolResult: Skipped ({len(result_str)} chars < 500)")
        return None
    logger.info(f"[SUMMARIZE] ToolResult: Summarizing {len(result_str)} chars")
    
    prompt = f"""Summarize this financial tool result in exactly 1 sentence. Focus on the most important finding.

Result:
{result_str[:1000]}...

Provide ONLY the 1-sentence summary, nothing else."""
    
    try:
        response = llm.invoke(prompt)
        logger.info(f"[SUMMARIZE] ToolResult: ✓ Generated 1-sentence summary")
        return response.content
    except Exception as e:
        logger.warning(f"[SUMMARIZE] ToolResult: Failed - {e}")
        return None


def extract_financial_metrics(text: str) -> Dict[str, Any]:
    """Extract financial metrics and KPIs from text using regex patterns."""
    logger.debug(f"[SUMMARIZE] Extracting metrics from {len(text)} chars")
    metrics = {}
    
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
            metrics[key] = matches[:3]
    
    logger.debug(f"[SUMMARIZE] Extracted {len(metrics)} metric types with {sum(len(v) for v in metrics.values())} total values")
    return metrics


def create_enhanced_tool_result(
    data: Any,
    tool_name: str,
    llm,
    reasoning: str = "",
    raw_result: Optional[Dict] = None
) -> Dict[str, Any]:
    """Create enhanced tool result with reasoning and summary."""
    logger.info(f"[SUMMARIZE] EnhancedResult: Creating enhanced result for tool={tool_name}")
    result_str = str(data)
    summary = summarize_tool_result(raw_result or {"data": data}, llm)
    metrics = extract_financial_metrics(result_str) if len(result_str) > 300 else {}
    logger.info(f"[SUMMARIZE] EnhancedResult: ✓ Extracted {len(metrics)} metrics")
    
    return {
        "data": data,
        "tool": tool_name,
        "reasoning": reasoning,
        "summary": summary,
        "metrics": metrics,
        "full_context": raw_result or {"data": data}
    }


def create_rag_summary(document_text: str, relevance_score: float) -> Dict[str, Any]:
    """Create extractive summary for RAG document."""
    logger.info(f"[SUMMARIZE] RAGSummary: Creating summary for {len(document_text)} chars (relevance={relevance_score:.2f})")
    metrics = extract_financial_metrics(document_text)
    summary = None
    
    if len(document_text) > 2000:
        sentences = document_text.split('. ')[:5]
        summary = '. '.join(sentences) + '.'
        logger.debug(f"[SUMMARIZE] RAGSummary: Extracted first 5 sentences as summary")
    
    logger.info(f"[SUMMARIZE] RAGSummary: ✓ Created summary with {len(metrics)} metrics")
    return {
        "full_text": document_text,
        "metrics_summary": metrics,
        "summary": summary,
        "relevance_score": relevance_score,
        "length": len(document_text)
    }


def estimate_message_tokens(messages: List[BaseMessage]) -> int:
    """Rough estimate of token count (4 chars ≈ 1 token)."""
    total_chars = sum(len(str(msg.content)) for msg in messages)
    token_estimate = total_chars // 4
    logger.debug(f"[SUMMARIZE] EstimateTokens: {total_chars} chars ≈ {token_estimate} tokens")
    return token_estimate


async def should_compress_history(messages: List[BaseMessage], context_limit: int = 6000) -> bool:
    """Check if conversation should be compressed."""
    token_count = estimate_message_tokens(messages)
    should_compress = token_count > context_limit * 0.8
    logger.info(f"[SUMMARIZE] CompressHistory: {token_count} tokens (limit={context_limit}), should_compress={should_compress}")
    return should_compress

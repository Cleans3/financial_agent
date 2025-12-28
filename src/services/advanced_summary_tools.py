"""
Advanced Summary Tools - 5 techniques for financial data summarization

Techniques:
1. Comparative Analysis - Highlights what changed vs just stating values
2. Anomaly Detection - Flags unusual patterns, reconciliation mismatches
3. Materiality-Weighted - Allocates summary space by metric importance
4. Narrative Arc - Structures as Setup -> Conflict -> Resolution story
5. Key Questions - Directly answers: Growth? Margins? Profitability? Sustainability? Guidance?

Each tool is self-contained and can be called independently or combined.
"""

import logging
import json
from typing import List, Dict, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MetricData:
    """Metric data extracted from chunks"""
    name: str
    current_value: Optional[str] = None
    prior_value: Optional[str] = None
    change_pct: Optional[float] = None
    period: Optional[str] = None
    period_prior: Optional[str] = None
    narrative: Optional[str] = None
    raw_text: Optional[str] = None


class SummaryTechnique(str, Enum):
    """Available summary techniques"""
    COMPARATIVE = "comparative_analysis"
    ANOMALY = "anomaly_detection"
    MATERIALITY = "materiality_weighted"
    NARRATIVE = "narrative_arc"
    KEY_QUESTIONS = "key_questions"


class ComparativeAnalysisSummarizer:
    """
    Technique 1: Comparative Analysis
    Highlights what CHANGED rather than just stating values
    
    Example:
    Basic: "Revenue was $500M"
    Advanced: "Revenue accelerated 12.1% to $500M, driven by pricing actions"
    """
    
    @staticmethod
    def summarize(metric_texts: List[str], query: str = "") -> Dict:
        """
        Generate comparative analysis summary
        
        Args:
            metric_texts: List of metric chunk texts
            query: Original user query for context
            
        Returns:
            Dictionary with comparative insights
        """
        logger.info("[SUMMARY:COMPARATIVE] Starting comparative analysis")
        
        # Extract metrics with trend information
        metrics = ComparativeAnalysisSummarizer._extract_metrics_with_trends(metric_texts)
        
        # Generate comparative narrative
        summary_parts = []
        
        for metric in metrics:
            if metric.change_pct is not None:
                # Focus on change, not absolute value
                direction = "accelerated" if metric.change_pct > 0 else "decelerated"
                materiality = "significant" if abs(metric.change_pct) > 5 else "modest"
                
                if metric.narrative:
                    summary_parts.append(
                        f"{metric.name.title()} {direction} {abs(metric.change_pct):.1f}% to {metric.current_value} "
                        f"({materiality} change). {metric.narrative}"
                    )
                else:
                    summary_parts.append(
                        f"{metric.name.title()} {direction} {abs(metric.change_pct):.1f}% "
                        f"({materiality} change)"
                    )
        
        logger.info(f"[SUMMARY:COMPARATIVE] ✓ Generated {len(summary_parts)} comparative insights")
        
        return {
            "technique": SummaryTechnique.COMPARATIVE,
            "summary": " ".join(summary_parts),
            "metrics_analyzed": len(metrics),
            "insights": summary_parts
        }
    
    @staticmethod
    def _extract_metrics_with_trends(texts: List[str]) -> List[MetricData]:
        """Extract metrics with trend information"""
        metrics = []
        
        # Look for patterns like "revenue grew 12%", "margin expanded 130bps"
        import re
        
        for text in texts:
            text_lower = text.lower()
            
            # Pattern: metric grew/expanded/increased X%
            pattern = r"(\w+)\s+(grew|expanded|increased|accelerated|decelerated|declined)\s+([\d.]+)%"
            matches = re.findall(pattern, text_lower)
            
            for metric_name, direction, change_str in matches:
                change = float(change_str)
                if direction in ["grew", "expanded", "increased", "accelerated"]:
                    change = abs(change)
                else:
                    change = -abs(change)
                
                # Extract current value
                value_pattern = f"{metric_name}[:\\s]+([\\$]?[\\d,.]+)"
                value_match = re.search(value_pattern, text, re.IGNORECASE)
                current_value = value_match.group(1) if value_match else None
                
                metrics.append(MetricData(
                    name=metric_name,
                    current_value=current_value,
                    change_pct=change,
                    raw_text=text[:200]  # First 200 chars for context
                ))
        
        return metrics


class AnomalyDetectionSummarizer:
    """
    Technique 2: Anomaly Detection
    Flags unusual patterns, reconciliation issues, and concerns
    
    Examples:
    - Revenue grew 12% but customers down 5% (mix shift)
    - Margin 18.4% vs historical avg 15.2% (3x volatility)
    - Drivers: +5% volume + 4% pricing = 9%, claimed 12% (1% unaccounted)
    """
    
    @staticmethod
    def summarize(metric_texts: List[str], query: str = "") -> Dict:
        """
        Detect and report anomalies
        
        Args:
            metric_texts: List of metric chunk texts
            query: Original user query for context
            
        Returns:
            Dictionary with anomalies and flags
        """
        logger.info("[SUMMARY:ANOMALY] Starting anomaly detection")
        
        anomalies = []
        
        # Check 1: Numerical reconciliation
        reconciliation_issues = AnomalyDetectionSummarizer._check_reconciliation(metric_texts)
        anomalies.extend(reconciliation_issues)
        
        # Check 2: Contra-intuitive patterns
        contra_patterns = AnomalyDetectionSummarizer._detect_contra_intuitive(metric_texts)
        anomalies.extend(contra_patterns)
        
        # Check 3: Trend consistency
        trend_issues = AnomalyDetectionSummarizer._check_trend_consistency(metric_texts)
        anomalies.extend(trend_issues)
        
        logger.info(f"[SUMMARY:ANOMALY] ✓ Detected {len(anomalies)} anomalies")
        
        return {
            "technique": SummaryTechnique.ANOMALY,
            "anomalies_found": len(anomalies) > 0,
            "anomalies": anomalies,
            "summary": "No anomalies detected" if not anomalies else 
                      f"⚠ {len(anomalies)} anomalies detected - review flagged items",
            "severity": "HIGH" if any(a.get("severity") == "HIGH" for a in anomalies) else "MEDIUM" if anomalies else "NONE"
        }
    
    @staticmethod
    def _check_reconciliation(texts: List[str]) -> List[Dict]:
        """Check if drivers mathematically reconcile"""
        issues = []
        
        import re
        
        for text in texts:
            # Look for "drivers: +5% + 4% + 2% = 11%, claimed 12%"
            pattern = r"drivers?[:\s]+([\d\+\-\s%.]*)\s*(?:claimed|actual|reported)\s+([\d.]+)%"
            match = re.search(pattern, text, re.IGNORECASE)
            
            if match:
                drivers_str = match.group(1)
                reported_str = match.group(2)
                
                # Try to sum drivers
                numbers = re.findall(r"[\+\-]?\s*(\d+\.?\d*)", drivers_str)
                if numbers:
                    driver_sum = sum(float(n) for n in numbers)
                    reported = float(reported_str)
                    
                    if abs(driver_sum - reported) > 0.5:  # More than 0.5% difference
                        issues.append({
                            "type": "reconciliation_mismatch",
                            "severity": "HIGH",
                            "description": f"Drivers sum to {driver_sum:.1f}% but {reported}% reported",
                            "implication": f"Unaccounted {reported - driver_sum:.1f}% growth - investigate"
                        })
        
        return issues
    
    @staticmethod
    def _detect_contra_intuitive(texts: List[str]) -> List[Dict]:
        """Detect contradictory signals like revenue up but customers down"""
        issues = []
        
        import re
        
        for text in texts:
            # Look for patterns like "revenue grew X% but customers declined Y%"
            if "but" in text.lower():
                sentences = re.split(r"[.!?]", text)
                for sentence in sentences:
                    if "revenue" in sentence.lower() and "customer" in sentence.lower():
                        if any(word in sentence.lower() for word in ["up", "grew", "increase"]):
                            if any(word in sentence.lower() for word in ["down", "decline", "decrease"]):
                                issues.append({
                                    "type": "contra_intuitive",
                                    "severity": "MEDIUM",
                                    "description": "Revenue growth with customer decline suggests mix shift",
                                    "implication": "Quality of growth may be impacted by customer composition"
                                })
        
        return issues
    
    @staticmethod
    def _check_trend_consistency(texts: List[str]) -> List[Dict]:
        """Check for erratic trends vs consistent momentum"""
        issues = []
        
        import re
        
        for text in texts:
            # Look for "Q1 12%, Q2 10.8%, Q3 12.1%" pattern
            pattern = r"([QqFf][1-4])\s+([0-9.]+)%"
            matches = re.findall(pattern, text)
            
            if len(matches) >= 3:
                values = [float(m[1]) for m in matches]
                
                # Check for volatility
                if len(values) >= 3:
                    diffs = [values[i+1] - values[i] for i in range(len(values)-1)]
                    volatility = sum(abs(d) for d in diffs) / len(diffs)
                    
                    if volatility > 1.5:  # High volatility
                        issues.append({
                            "type": "trend_volatility",
                            "severity": "MEDIUM",
                            "description": f"Erratic growth pattern: {[f'{v:.1f}%' for v in values]}",
                            "implication": "Momentum appears inconsistent despite stated strength"
                        })
        
        return issues


class MaterialityWeightedSummarizer:
    """
    Technique 3: Materiality-Weighted Summarization
    Allocates summary space by metric importance
    
    High materiality (90+): 40% of summary
    Medium (70-89): 20% of summary  
    Low (50-69): 10% of summary
    """
    
    @staticmethod
    def summarize(metric_texts: List[str], query: str = "") -> Dict:
        """
        Generate summary with materiality weighting
        
        Args:
            metric_texts: List of metric chunk texts
            query: Original user query for context
            
        Returns:
            Dictionary with materiality-weighted summary
        """
        logger.info("[SUMMARY:MATERIALITY] Starting materiality weighting")
        
        # Score each metric by materiality
        scored_metrics = MaterialityWeightedSummarizer._score_materiality(metric_texts)
        
        # Allocate summary space by materiality
        high = [m for m in scored_metrics if m["score"] >= 90]
        medium = [m for m in scored_metrics if 70 <= m["score"] < 90]
        low = [m for m in scored_metrics if 50 <= m["score"] < 70]
        
        summary_parts = []
        
        # High materiality: 40% - lead with these
        if high:
            summary_parts.append("KEY DRIVERS: " + ", ".join(m["metric"] for m in high))
        
        # Medium: 20% - include but secondary
        if medium:
            summary_parts.append("SUPPORTING METRICS: " + ", ".join(m["metric"] for m in medium))
        
        # Low: 10% - mention briefly
        if low:
            summary_parts.append("Other metrics: " + ", ".join(m["metric"] for m in low))
        
        logger.info(f"[SUMMARY:MATERIALITY] ✓ Weighted {len(scored_metrics)} metrics: "
                   f"{len(high)} high, {len(medium)} medium, {len(low)} low")
        
        return {
            "technique": SummaryTechnique.MATERIALITY,
            "summary": " | ".join(summary_parts),
            "materiality_breakdown": {
                "high_priority": [m["metric"] for m in high],
                "medium_priority": [m["metric"] for m in medium],
                "low_priority": [m["metric"] for m in low]
            },
            "allocation": {
                "high": f"{len(high) / len(scored_metrics) * 100:.0f}%" if scored_metrics else "0%",
                "medium": f"{len(medium) / len(scored_metrics) * 100:.0f}%" if scored_metrics else "0%",
                "low": f"{len(low) / len(scored_metrics) * 100:.0f}%" if scored_metrics else "0%"
            }
        }
    
    @staticmethod
    def _score_materiality(texts: List[str]) -> List[Dict]:
        """Score each metric by materiality (impact on valuation, user interest, etc)"""
        metrics_scores = {
            "revenue": 95,
            "profit": 90,
            "margin": 85,
            "earnings": 90,
            "cash flow": 85,
            "debt": 75,
            "equity": 70,
            "customer": 75,
            "growth": 80,
            "guidance": 80,
            "dividend": 65,
        }
        
        found_metrics = []
        
        for text in texts:
            text_lower = text.lower()
            
            for metric_name, score in metrics_scores.items():
                if metric_name in text_lower:
                    found_metrics.append({
                        "metric": metric_name.upper(),
                        "score": score,
                        "source": text[:100]  # First 100 chars
                    })
        
        # Remove duplicates keeping highest score
        unique = {}
        for m in found_metrics:
            key = m["metric"]
            if key not in unique or m["score"] > unique[key]["score"]:
                unique[key] = m
        
        return sorted(unique.values(), key=lambda x: x["score"], reverse=True)


class NarrativeArcSummarizer:
    """
    Technique 4: Narrative Arc Structure
    Setup -> Conflict -> Resolution (3-act storytelling)
    
    ACT 1: What was the context/expectation?
    ACT 2: What actually happened?
    ACT 3: What does it mean?
    """
    
    @staticmethod
    def summarize(metric_texts: List[str], query: str = "") -> Dict:
        """
        Generate narrative arc summary
        
        Args:
            metric_texts: List of metric chunk texts
            query: Original user query for context
            
        Returns:
            Dictionary with three-act narrative
        """
        logger.info("[SUMMARY:NARRATIVE] Starting narrative arc construction")
        
        # Extract context and build narrative
        setup = NarrativeArcSummarizer._extract_setup(metric_texts)
        conflict = NarrativeArcSummarizer._extract_conflict(metric_texts)
        resolution = NarrativeArcSummarizer._extract_resolution(metric_texts)
        
        # Compose narrative
        narrative = f"{setup} {conflict} {resolution}"
        
        logger.info("[SUMMARY:NARRATIVE] ✓ Narrative arc constructed")
        
        return {
            "technique": SummaryTechnique.NARRATIVE,
            "summary": narrative,
            "acts": {
                "setup": setup,
                "conflict": conflict,
                "resolution": resolution
            },
            "narrative_flow": "Complete" if all([setup, conflict, resolution]) else "Partial"
        }
    
    @staticmethod
    def _extract_setup(texts: List[str]) -> str:
        """Extract context/setup from texts"""
        # Look for guidance, expectations, context
        combined = " ".join(texts)
        
        if "guidance" in combined.lower() or "expect" in combined.lower():
            return "The company faced questions about growth sustainability and margin resilience."
        elif "concern" in combined.lower() or "pressure" in combined.lower():
            return "Against competitive headwinds, the company needed to demonstrate execution."
        else:
            return "In the reported period, the company faced market challenges."
    
    @staticmethod
    def _extract_conflict(texts: List[str]) -> str:
        """Extract development/conflict from texts"""
        combined = " ".join(texts)
        
        if "growth" in combined.lower() and "margin" in combined.lower():
            if "accelerat" in combined.lower():
                return "However, results showed accelerating growth alongside margin expansion, demonstrating both topline momentum and operational leverage."
            else:
                return "The company delivered steady growth with margin improvement, indicating balanced execution."
        elif any(word in combined.lower() for word in ["beat", "outperform", "exceed"]):
            return "Actual results exceeded expectations on multiple fronts."
        else:
            return "Results showed mixed performance across key metrics."
    
    @staticmethod
    def _extract_resolution(texts: List[str]) -> str:
        """Extract resolution/implications from texts"""
        combined = " ".join(texts)
        
        if "sustainable" in combined.lower() or "structural" in combined.lower():
            return "This suggests the company possesses structural advantages positioning it well for sustained growth."
        elif "concern" in combined.lower() or "risk" in combined.lower():
            return "However, lingering concerns around execution risks warrant continued monitoring."
        else:
            return "The results raise confidence in the company's competitive positioning and ability to create value."


class KeyQuestionsAnsweringSummarizer:
    """
    Technique 5: Key Questions Answered
    Directly answer what analysts actually want to know:
    1. Is growth fast? Accelerating?
    2. Are margins expanding?
    3. Is growth profitable?
    4. Is this sustainable?
    5. Did they beat guidance?
    """
    
    @staticmethod
    def summarize(metric_texts: List[str], query: str = "") -> Dict:
        """
        Answer key analyst questions
        
        Args:
            metric_texts: List of metric chunk texts
            query: Original user query for context
            
        Returns:
            Dictionary with answers to key questions
        """
        logger.info("[SUMMARY:KEY_QUESTIONS] Starting key questions answering")
        
        answers = {
            "growth_speed": KeyQuestionsAnsweringSummarizer._answer_growth_speed(metric_texts),
            "margin_trend": KeyQuestionsAnsweringSummarizer._answer_margins(metric_texts),
            "profitability": KeyQuestionsAnsweringSummarizer._answer_profitability(metric_texts),
            "sustainability": KeyQuestionsAnsweringSummarizer._answer_sustainability(metric_texts),
            "guidance": KeyQuestionsAnsweringSummarizer._answer_guidance(metric_texts)
        }
        
        # Compose summary from answers
        summary_parts = [v for v in answers.values() if v]
        
        logger.info("[SUMMARY:KEY_QUESTIONS] ✓ Answered all key analyst questions")
        
        return {
            "technique": SummaryTechnique.KEY_QUESTIONS,
            "summary": " ".join(summary_parts),
            "answers": {
                "growth_speed": answers["growth_speed"],
                "margin_expansion": answers["margin_trend"],
                "earnings_quality": answers["profitability"],
                "sustainability": answers["sustainability"],
                "guidance_beat_miss": answers["guidance"]
            }
        }
    
    @staticmethod
    def _answer_growth_speed(texts: List[str]) -> str:
        """Q1: How fast is it growing? Accelerating?"""
        combined = " ".join(texts)
        
        if "accelerat" in combined.lower():
            return "Growth is accelerating with strong momentum. "
        elif "deceler" in combined.lower() or "slow" in combined.lower():
            return "Growth is decelerating and losing momentum. "
        elif any(word in combined.lower() for word in ["12%", "double digit", "strong"]):
            return "Growth remains solid and in double digits. "
        else:
            return "Growth is steady. "
    
    @staticmethod
    def _answer_margins(texts: List[str]) -> str:
        """Q2: Are margins expanding?"""
        combined = " ".join(texts)
        
        if "margin expand" in combined.lower() or "margin up" in combined.lower():
            return "Margins are expanding, driven by operational leverage. "
        elif "margin compress" in combined.lower() or "margin down" in combined.lower():
            return "Margins are under pressure from cost inflation. "
        else:
            return "Margins are stable. "
    
    @staticmethod
    def _answer_profitability(texts: List[str]) -> str:
        """Q3: Is growth profitable? Converting to cash?"""
        combined = " ".join(texts)
        
        if "fcf" in combined.lower() or "cash flow" in combined.lower():
            if "outpacing" in combined.lower() or "exceed" in combined.lower():
                return "FCF growth outpaces earnings, indicating high-quality profits. "
            else:
                return "Cash generation supports earnings growth. "
        else:
            return "Earnings appear profitable. "
    
    @staticmethod
    def _answer_sustainability(texts: List[str]) -> str:
        """Q4: Is this repeatable/sustainable?"""
        combined = " ".join(texts)
        
        if "structural" in combined.lower():
            return "The improvements appear structural and sustainable. "
        elif "one-time" in combined.lower() or "temporary" in combined.lower():
            return "Some benefits may be temporary or one-time in nature. "
        else:
            return "Sustainability depends on continued execution. "
    
    @staticmethod
    def _answer_guidance(texts: List[str]) -> str:
        """Q5: Did they beat/miss guidance?"""
        combined = " ".join(texts)
        
        if "beat" in combined.lower() and "guidance" in combined.lower():
            return "Results beat guidance significantly, signaling conservative prior expectations or strong execution. "
        elif "miss" in combined.lower() and "guidance" in combined.lower():
            return "Results missed guidance, raising execution questions. "
        else:
            return "Results aligned with expectations. "


def select_summary_technique(query: str, retrieved_chunks: List[Dict]) -> SummaryTechnique:
    """
    Select appropriate summary technique based on query type
    
    Args:
        query: User query
        retrieved_chunks: Retrieved chunks from vector DB
        
    Returns:
        Selected technique
    """
    query_lower = query.lower()
    
    # Rule-based selection
    if any(word in query_lower for word in ["summary", "overview", "takeaway"]):
        return SummaryTechnique.MATERIALITY  # Material metrics only
    
    if any(word in query_lower for word in ["change", "accelerat", "deceler", "vs", "versus", "compared"]):
        return SummaryTechnique.COMPARATIVE  # Show what changed
    
    if any(word in query_lower for word in ["unusual", "anomaly", "concern", "risk", "red flag"]):
        return SummaryTechnique.ANOMALY  # Flag problems
    
    if any(word in query_lower for word in ["what happened", "story", "narrative", "explain"]):
        return SummaryTechnique.NARRATIVE  # Tell a story
    
    if any(word in query_lower for word in ["sustainable", "profitable", "growth", "guidance", "beat"]):
        return SummaryTechnique.KEY_QUESTIONS  # Answer key Qs
    
    # Default based on chunk count
    if len(retrieved_chunks) > 5:
        return SummaryTechnique.MATERIALITY  # Too much data, focus on what matters
    
    return SummaryTechnique.KEY_QUESTIONS  # Default to key questions


def apply_summary_tool(technique: SummaryTechnique,
                       metric_texts: List[str],
                       query: str = "") -> Dict:
    """
    Apply selected summary technique
    
    Args:
        technique: Technique to apply
        metric_texts: List of metric chunk texts
        query: Original user query
        
    Returns:
        Summary result
    """
    logger.info(f"[SUMMARY:APPLY] Applying {technique.value} technique")
    
    if technique == SummaryTechnique.COMPARATIVE:
        result = ComparativeAnalysisSummarizer.summarize(metric_texts, query)
    elif technique == SummaryTechnique.ANOMALY:
        result = AnomalyDetectionSummarizer.summarize(metric_texts, query)
    elif technique == SummaryTechnique.MATERIALITY:
        result = MaterialityWeightedSummarizer.summarize(metric_texts, query)
    elif technique == SummaryTechnique.NARRATIVE:
        result = NarrativeArcSummarizer.summarize(metric_texts, query)
    elif technique == SummaryTechnique.KEY_QUESTIONS:
        result = KeyQuestionsAnsweringSummarizer.summarize(metric_texts, query)
    else:
        logger.warning(f"[SUMMARY:APPLY] Unknown technique: {technique}")
        result = KeyQuestionsAnsweringSummarizer.summarize(metric_texts, query)
    
    logger.info(f"[SUMMARY:APPLY] ✓ Applied {technique.value}")
    return result

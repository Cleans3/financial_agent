"""
Prompt Classifier - Categorizes user input into 5 types
Implements: PROMPT_HANDLER node logic
"""

import logging
import re
from typing import Tuple, Optional
from .workflow_state import PromptType

logger = logging.getLogger(__name__)


class PromptClassifier:
    """
    Classifies user prompts into 5 categories:
    - CHITCHAT: Greetings, small talk (no financial question)
    - REQUEST: Asking for data, reports, analysis
    - INSTRUCTION: Other questions or commands
    - AMBIGUOUS: Unclear references, missing context
    - FILE_ONLY: Empty prompt, only files provided
    
    Uses rule-based detection for simple cases and LLM for complex classification.
    """
    
    def __init__(self, llm=None):
        """
        Initialize classifier.
        
        Args:
            llm: Optional LLM instance for complex classifications
        """
        self.llm = llm
        self.logger = logging.getLogger(__name__)
    
    async def classify(self, prompt: Optional[str], has_files: bool = False) -> Tuple[PromptType, float]:
        """
        Classify prompt into one of 5 types.
        
        Args:
            prompt: User's input text
            has_files: Whether files were uploaded
            
        Returns:
            Tuple of (prompt_type, confidence_score)
            Confidence: 0.0-1.0, higher = more confident
        """
        # Handle file-only case
        if not prompt or prompt.strip() == "":
            if has_files:
                self.logger.info("Classified as FILE_ONLY (no prompt, files provided)")
                return (PromptType.FILE_ONLY, 0.95)
            else:
                self.logger.warning("No prompt and no files - treating as CHITCHAT")
                return (PromptType.CHITCHAT, 0.5)
        
        # Try rule-based detection first
        prompt_type = self._detect_pattern(prompt)
        if prompt_type is not None:
            self.logger.info(f"Rule-based classification: {prompt_type.value}")
            return (prompt_type, 0.85)  # High confidence for pattern matches
        
        # Fall back to LLM if available
        if self.llm:
            return await self._classify_with_llm(prompt)
        
        # Default fallback
        self.logger.info("Defaulting to INSTRUCTION (no pattern match, no LLM)")
        return (PromptType.INSTRUCTION, 0.5)
    
    def _detect_pattern(self, prompt: str) -> Optional[PromptType]:
        """
        Rule-based pattern detection for common cases.
        
        Args:
            prompt: User's input text
            
        Returns:
            PromptType if pattern matched, None otherwise
        """
        prompt_lower = prompt.lower().strip()
        
        # === CHITCHAT DETECTION ===
        # Greetings and small talk
        greetings = [
            "hello", "hi", "hey", "howdy", "greetings",
            "thanks", "thank you", "thx", "appreciate",
            "cảm ơn", "chào", "xin chào", "chào bạn",
            "bye", "goodbye", "see you", "take care",
            "how are you", "bạn khỏe không", "như thế nào"
        ]
        
        # Check if prompt is primarily a greeting
        if any(greeting in prompt_lower for greeting in greetings):
            # Make sure it's not a greeting + real question
            if len(prompt_lower) < 50:  # Short messages are likely greetings
                self.logger.debug(f"Detected greeting pattern: {prompt_lower}")
                return PromptType.CHITCHAT
        
        # === REQUEST DETECTION ===
        # Action words indicating data/report requests
        request_verbs = [
            "find", "get", "show", "list", "fetch",
            "retrieve", "search", "query", "lookup",
            "give me", "tell me", "provide",
            "what is", "what are", "which",
            "tìm", "lấy", "hiển thị", "liệt kê"
        ]
        
        if any(verb in prompt_lower for verb in request_verbs):
            # Make sure it's not ambiguous reference after the request
            if not self._has_ambiguous_reference(prompt_lower):
                self.logger.debug(f"Detected request pattern: {prompt_lower}")
                return PromptType.REQUEST
        
        # === AMBIGUOUS DETECTION ===
        # Pronouns and unclear references without context
        if self._has_ambiguous_reference(prompt_lower):
            self.logger.debug(f"Detected ambiguous pattern: {prompt_lower}")
            return PromptType.AMBIGUOUS
        
        # No pattern matched
        return None
    
    def _has_ambiguous_reference(self, prompt_lower: str) -> bool:
        """
        Check if prompt has ambiguous pronouns or references.
        
        Args:
            prompt_lower: Lowercase prompt text
            
        Returns:
            True if ambiguous references detected
        """
        # Ambiguous pronouns and references
        ambiguous_patterns = [
            r'\b(it|they|their|them|that|this|those|these)\b\s+(?!(\.|\?|company|corporation|bank))',
            r'\bwhat\'?s (it|that|this)\b',
            r'\bshow\s+(it|them|that)\b',
            r'\bcompare\s+(it|them|that)\b',
        ]
        
        for pattern in ambiguous_patterns:
            if re.search(pattern, prompt_lower):
                return True
        
        return False
    
    async def _classify_with_llm(self, prompt: str) -> Tuple[PromptType, float]:
        """
        Use LLM for classification of complex cases.
        
        Args:
            prompt: User's input text
            
        Returns:
            Tuple of (prompt_type, confidence_score)
        """
        try:
            from langchain_core.prompts import ChatPromptTemplate
            
            # Create classification prompt
            classification_prompt = ChatPromptTemplate.from_template("""
Classify this user prompt into ONE category:

Categories:
- CHITCHAT: Greetings, small talk, no financial question
- REQUEST: Asking for data, reports, or financial analysis
- INSTRUCTION: Other questions or commands related to finance
- AMBIGUOUS: Unclear references, pronouns without context
- FILE_ONLY: Should not reach here (handled separately)

Prompt: "{prompt}"

Respond with JSON: {{"type": "CHITCHAT|REQUEST|INSTRUCTION|AMBIGUOUS", "confidence": 0.0-1.0}}
""")
            
            chain = classification_prompt | self.llm
            response = await chain.ainvoke({"prompt": prompt})
            
            # Parse response
            import json
            result = json.loads(response.content)
            
            prompt_type = PromptType[result["type"].upper()]
            confidence = float(result.get("confidence", 0.5))
            
            self.logger.info(f"LLM classification: {prompt_type.value} (confidence: {confidence})")
            return (prompt_type, confidence)
        
        except Exception as e:
            self.logger.warning(f"LLM classification failed: {e}, defaulting to INSTRUCTION")
            return (PromptType.INSTRUCTION, 0.5)
    
    async def classify_with_confidence(self, prompt: Optional[str], 
                                       has_files: bool = False,
                                       confidence_threshold: float = 0.7) -> Tuple[PromptType, float, bool]:
        """
        Classify with confidence threshold check.
        
        Args:
            prompt: User's input text
            has_files: Whether files were uploaded
            confidence_threshold: Minimum confidence required (0.0-1.0)
            
        Returns:
            Tuple of (prompt_type, confidence_score, meets_threshold)
        """
        prompt_type, confidence = await self.classify(prompt, has_files)
        meets_threshold = confidence >= confidence_threshold
        
        return (prompt_type, confidence, meets_threshold)

"""
Query Rewriter - Refines ambiguous or incomplete queries
Implements: REWRITE_EVALUATION, REWRITE_FILE_CONTEXT, REWRITE_CONVERSATION_CONTEXT node logic
"""

import logging
import re
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class QueryRewriter:
    """
    Rewrites ambiguous or incomplete user queries by incorporating context.
    
    Three rewriting strategies:
    1. File context: Inject uploaded file names/metadata into query
    2. Conversation context: Resolve pronouns using conversation history
    3. Session context: Reference previous queries in current session
    """
    
    def __init__(self, llm=None):
        """
        Initialize query rewriter.
        
        Args:
            llm: Optional LLM instance for complex rewrites
        """
        self.llm = llm
        self.logger = logging.getLogger(__name__)
        
        # Regex patterns for ambiguity detection
        self.pronoun_pattern = re.compile(r'\b(it|it\'s|they|their|this|that|its)\b')
        self.reference_pattern = re.compile(r'\b(the company|the data|the document|the file|that report)\b')
        self.vague_pattern = re.compile(r'\b(some|many|any|all|each)\s+\w+\b')
    
    async def evaluate_need_for_rewriting(
        self,
        prompt: str,
        has_files: bool = False,
        conversation_history: Optional[List[Dict]] = None
    ) -> bool:
        """
        Determine if query needs rewriting (contains ambiguous references).
        
        Args:
            prompt: User's input query
            has_files: Whether files were uploaded
            conversation_history: Previous conversation context
            
        Returns:
            True if rewriting recommended, False otherwise
        """
        if not prompt or len(prompt.strip()) == 0:
            return False
        
        prompt_lower = prompt.lower()
        
        # Check 1: Has uploaded files but prompt doesn't reference them
        if has_files and not self._mentions_files(prompt):
            self.logger.debug("Query lacks file references despite files uploaded")
            return True
        
        # Check 2: Contains pronouns without clear antecedent
        if self.pronoun_pattern.search(prompt_lower):
            # Check if there's clear previous context
            if not conversation_history or len(conversation_history) < 2:
                self.logger.debug("Query has pronouns but no conversation history")
                return True
        
        # Check 3: Vague references to data
        if self.reference_pattern.search(prompt_lower):
            if not conversation_history or len(conversation_history) < 2:
                self.logger.debug("Query has vague data references")
                return True
        
        # Check 4: Vague quantifiers without context
        vague_matches = self.vague_pattern.findall(prompt_lower)
        if len(vague_matches) >= 2:
            self.logger.debug(f"Query has multiple vague references: {vague_matches}")
            return True
        
        return False
    
    async def rewrite_with_file_context(
        self,
        prompt: str,
        file_metadata: List[Dict[str, Any]]
    ) -> str:
        """
        Inject file context into query to resolve "it", "the document", etc.
        
        Args:
            prompt: Original user query
            file_metadata: List of uploaded files with metadata
                [{"filename": str, "size": int, "content_type": str, "doc_id": str}, ...]
                
        Returns:
            Rewritten query incorporating file information
        """
        if not file_metadata:
            return prompt
        
        # Extract file names and key info
        file_names = [f.get("filename", f.get("name", "file")) for f in file_metadata]
        file_count = len(file_metadata)
        
        # Check if prompt mentions generic file references
        if re.search(r'\b(it|that|the document|the file|the data)\b', prompt, re.IGNORECASE):
            # Build file context string
            if file_count == 1:
                file_context = f"from the uploaded file '{file_names[0]}'"
            else:
                files_str = "', '".join(file_names[:3])  # Max 3 file names
                if file_count > 3:
                    files_str += f"' and {file_count - 3} more files"
                file_context = f"from the uploaded files '{files_str}'"
            
            # Insert context into prompt
            rewritten = f"{prompt} (referring to the data {file_context})"
            self.logger.info(f"Rewritten with file context: {rewritten[:100]}...")
            return rewritten
        
        return prompt
    
    async def rewrite_with_conversation_context(
        self,
        prompt: str,
        conversation_history: List[Dict[str, str]]
    ) -> str:
        """
        Resolve pronouns and references using conversation history.
        
        Args:
            prompt: Original user query with ambiguous references
            conversation_history: Previous messages
                [{"role": "user"|"assistant", "content": str}, ...]
                
        Returns:
            Rewritten query with resolved references
        """
        if not conversation_history or len(conversation_history) < 2:
            return prompt
        
        # Extract recent context (last 2-3 messages)
        recent_context = []
        for msg in conversation_history[-3:]:
            if msg.get("role") == "user":
                recent_context.append(msg.get("content", ""))
        
        # Check for pronouns that need resolution
        pronouns_found = self.pronoun_pattern.findall(prompt)
        if not pronouns_found:
            return prompt
        
        # Use LLM to resolve references if available
        if self.llm:
            rewritten = await self._resolve_with_llm(prompt, recent_context)
            if rewritten and rewritten != prompt:
                self.logger.info(f"Rewritten with conversation context via LLM")
                return rewritten
        
        # Fallback: Simple heuristic-based resolution
        rewritten = await self._resolve_with_heuristics(prompt, recent_context)
        self.logger.info(f"Rewritten with heuristic resolution: {rewritten[:100]}...")
        return rewritten
    
    async def _resolve_with_llm(self, prompt: str, context: List[str]) -> Optional[str]:
        """
        Use LLM to resolve pronouns and ambiguous references.
        
        Args:
            prompt: Query with ambiguous references
            context: Recent conversation messages
            
        Returns:
            Rewritten query or None if resolution failed
        """
        if not self.llm:
            return None
        
        context_str = "\n".join(context) if context else "No previous context"
        
        resolution_prompt = f"""Given this conversation context and the current query, 
resolve any ambiguous pronouns or references:

Recent Context:
{context_str}

Current Query: {prompt}

Provide ONLY the rewritten query without pronouns or ambiguous references.
If no ambiguity detected, return the original query exactly.

Rewritten Query:"""
        
        try:
            response = await self.llm.ainvoke({"input": resolution_prompt})
            rewritten = response.get("output", prompt).strip()
            
            # Ensure output is reasonable length (not hallucinated)
            if len(rewritten) > 3 * len(prompt):
                self.logger.warning("LLM rewrite too long, using original")
                return None
            
            return rewritten if rewritten else prompt
        except Exception as e:
            self.logger.error(f"LLM resolution failed: {e}")
            return None
    
    async def _resolve_with_heuristics(self, prompt: str, context: List[str]) -> str:
        """
        Simple heuristic-based pronoun resolution.
        
        Args:
            prompt: Query with pronouns
            context: Recent messages
            
        Returns:
            Rewritten query with resolved references
        """
        rewritten = prompt
        
        # Extract key nouns from context
        key_nouns = []
        for msg in context:
            # Simple extraction: assume capitalized words are entities
            matches = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b', msg)
            key_nouns.extend(matches)
        
        if not key_nouns:
            return prompt
        
        # Replace first-person pronouns with most recent noun from context
        most_recent_noun = key_nouns[-1] if key_nouns else "the data"
        
        # Replace pronouns (simple replacements)
        replacements = {
            r'\bit\b': f"the {most_recent_noun.lower()}",
            r'\bits\b': f"the {most_recent_noun.lower()}'s",
            r'\bthis\b': f"the {most_recent_noun}",
            r'\bthat\b': f"the {most_recent_noun}",
        }
        
        for pattern, replacement in replacements.items():
            rewritten = re.sub(pattern, replacement, rewritten, flags=re.IGNORECASE)
        
        return rewritten
    
    def _mentions_files(self, prompt: str) -> bool:
        """
        Check if query mentions uploaded files.
        
        Args:
            prompt: User query
            
        Returns:
            True if query references uploaded files
        """
        file_mentions = [
            r'\b(file|files|document|documents|data|upload|uploaded)\b',
            r'\b(in (the )?(uploaded )?files?)\b',
            r'\b(from (my )?files?)\b',
        ]
        
        for pattern in file_mentions:
            if re.search(pattern, prompt, re.IGNORECASE):
                return True
        
        return False
    
    def get_rewriting_summary(self) -> Dict[str, Any]:
        """
        Get summary of rewrites performed in this session.
        
        Returns:
            Dict with statistics about rewriting
        """
        return {
            "description": "Query Rewriter for resolving ambiguous references",
            "strategies": [
                "file_context: Inject file names into queries",
                "conversation_context: Resolve pronouns from history",
                "heuristic_resolution: Rule-based pronoun replacement",
                "llm_resolution: LLM-based smart resolution"
            ]
        }

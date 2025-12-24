import logging
from typing import List, Dict, Optional, Tuple
from langchain_core.messages import HumanMessage, AIMessage
from src.core.config import settings
from src.llm.llm_factory import get_llm

logger = logging.getLogger(__name__)


class SmartQueryProcessor:
    """Process queries with unfulfilled request detection and rewriting"""
    
    def __init__(self):
        self.llm = get_llm()
    
    def detect_unfulfilled_requests(self, messages: List) -> List[Dict]:
        """
        Find previous questions in conversation that were not fully answered
        
        Returns: List of unfulfilled requests with question and context
        """
        unfulfilled = []
        
        for i, msg in enumerate(messages):
            if isinstance(msg, HumanMessage) and i + 1 < len(messages):
                next_msg = messages[i + 1]
                
                if isinstance(next_msg, AIMessage):
                    response = next_msg.content if isinstance(next_msg.content, str) else str(next_msg.content)
                    
                    insufficient_indicators = [
                        "i don't have enough data",
                        "không có đủ dữ liệu",
                        "need more information",
                        "không đủ thông tin",
                        "please provide",
                        "vui lòng cung cấp",
                        "unable to answer",
                        "không thể trả lời",
                        "require additional data",
                        "cần thêm dữ liệu"
                    ]
                    
                    if any(indicator in response.lower() for indicator in insufficient_indicators):
                        unfulfilled.append({
                            'question': msg.content,
                            'context': response,
                            'index': i
                        })
        
        return unfulfilled
    
    def rewrite_prompt_for_unfulfilled(self, 
                                      original_query: str,
                                      unfulfilled_requests: List[Dict]) -> str:
        """
        Rewrite query to address unfulfilled requests when file is uploaded
        
        Example: User asked "What's the revenue trend?" → Agent couldn't answer
                 User uploads file → Rewritten: "Based on uploaded file, what's the revenue trend?"
        """
        if not unfulfilled_requests:
            return original_query
        
        recent_request = unfulfilled_requests[0]
        original_question = recent_request['question']
        
        try:
            prompt = f"""A user previously asked: "{original_question}"
The assistant could not answer due to lack of data.
Now the user has uploaded a relevant document and the query is: "{original_query}"

Rewrite the user's current query to explicitly address their original unfulfilled request.
The rewritten query should:
1. Reference the newly uploaded document
2. Address the original unanswered question
3. Be natural and concise

Return only the rewritten query, no explanation."""
            
            response = self.llm.invoke(prompt)
            rewritten = response.content if hasattr(response, 'content') else str(response)
            logger.info(f"Rewrote query for unfulfilled request")
            return rewritten.strip()
        
        except Exception as e:
            logger.warning(f"Failed to rewrite query: {e}")
            return original_query
    
    def process_file_upload_query(self,
                                 original_query: Optional[str],
                                 conversation_messages: List,
                                 file_summary: Optional[str]) -> str:
        """
        Process query when file is uploaded
        
        If no explicit query: check for unfulfilled requests and rewrite prompt
        If explicit query: enhance with file context
        """
        if original_query:
            context_prefix = f"Given the uploaded document with summary: {file_summary}\n\n" if file_summary else ""
            return context_prefix + original_query
        
        unfulfilled = self.detect_unfulfilled_requests(conversation_messages)
        if unfulfilled:
            return self.rewrite_prompt_for_unfulfilled(
                "Analyze the uploaded document",
                unfulfilled
            )
        
        return "Analyze and summarize the uploaded document"


def get_query_processor() -> SmartQueryProcessor:
    return SmartQueryProcessor()

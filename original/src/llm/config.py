"""
LLM Configuration Module
Hỗ trợ Google Gemini và Ollama
"""

import os
from dotenv import load_dotenv

load_dotenv()


class LLMConfig:
    """LLM configuration settings"""
    
    # Provider selection
    PROVIDER = os.getenv("LLM_PROVIDER", "gemini")  # "gemini" or "ollama"
    
    # Gemini settings
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    GEMINI_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash")
    
    # Ollama settings
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
    
    # Generation settings
    TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))
    
    @classmethod
    def validate(cls):
        """Validate configuration"""
        if cls.PROVIDER == "gemini" and not cls.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is required for Gemini provider")
        return True

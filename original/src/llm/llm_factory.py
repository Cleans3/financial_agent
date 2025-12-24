"""
LLM Factory - Hỗ trợ nhiều LLM providers (Ollama, Gemini)
"""

import logging
from langchain_core.language_models.chat_models import BaseChatModel
from .config import LLMConfig

logger = logging.getLogger(__name__)


class LLMFactory:
    """
    Factory để tạo LLM instance dựa trên cấu hình
    
    Hỗ trợ:
    - Ollama (local)
    - Google Gemini (API)
    
    Cấu hình qua biến môi trường trong file .env:
    - LLM_PROVIDER: "ollama" hoặc "gemini" (mặc định: "gemini")
    - OLLAMA_MODEL: model name cho Ollama (mặc định: "qwen3:8b")
    - OLLAMA_BASE_URL: URL của Ollama server (mặc định: "http://localhost:11434")
    - GOOGLE_API_KEY: API key cho Gemini (bắt buộc nếu dùng gemini)
    - LLM_MODEL: model name cho Gemini (mặc định: "gemini-2.0-flash")
    - LLM_TEMPERATURE: temperature cho generation (mặc định: 0.3)
    - LLM_MAX_TOKENS: max tokens (mặc định: 2048)
    """
    
    @staticmethod
    def get_llm() -> BaseChatModel:
        """
        Tạo và trả về LLM instance dựa trên cấu hình
        
        Returns:
            BaseChatModel instance (ChatOllama hoặc ChatGoogleGenerativeAI)
        """
        provider = LLMConfig.PROVIDER.lower()
        
        logger.info(f"Initializing LLM with provider: {provider}")
        
        if provider == "gemini":
            return LLMFactory._get_gemini_llm()
        elif provider == "ollama":
            return LLMFactory._get_ollama_llm()
        else:
            logger.warning(f"Unknown provider '{provider}', falling back to Gemini")
            return LLMFactory._get_gemini_llm()
    
    @staticmethod
    def _get_ollama_llm() -> BaseChatModel:
        """
        Tạo ChatOllama instance
        
        Returns:
            ChatOllama configured instance
        """
        from langchain_ollama import ChatOllama
        
        model = LLMConfig.OLLAMA_MODEL
        base_url = LLMConfig.OLLAMA_BASE_URL
        
        logger.info(f"Creating Ollama LLM: model={model}, base_url={base_url}")
        
        llm = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=LLMConfig.TEMPERATURE,
            num_predict=LLMConfig.MAX_TOKENS,
        )
        
        logger.info("Ollama LLM created successfully")
        return llm
    
    @staticmethod
    def _get_gemini_llm() -> BaseChatModel:
        """
        Tạo ChatGoogleGenerativeAI instance
        
        Returns:
            ChatGoogleGenerativeAI configured instance
            
        Raises:
            ValueError: Nếu GOOGLE_API_KEY không được set
        """
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        # Validate config
        LLMConfig.validate()
        
        model = LLMConfig.GEMINI_MODEL
        api_key = LLMConfig.GOOGLE_API_KEY
        
        logger.info(f"Creating Gemini LLM: model={model}")
        
        llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=LLMConfig.TEMPERATURE,
            max_tokens=LLMConfig.MAX_TOKENS,
        )
        
        logger.info("Gemini LLM created successfully")
        return llm


__all__ = ["LLMFactory"]

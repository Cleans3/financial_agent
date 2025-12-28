import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    PROJECT_NAME: str = "Financial Agent"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql://financial_user:financial_password@localhost:5432/financial_agent"
    )
    
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "dev-secret-key-change-in-production")
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_MINUTES: int = 60
    JWT_REFRESH_EXPIRATION_DAYS: int = 7
    
    ADMIN_USERNAME: str = os.getenv("ADMIN_USERNAME", "admin")
    ADMIN_PASSWORD: str = os.getenv("ADMIN_PASSWORD", "admin123")
    
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-2.5-flash")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "4096"))
    
    # Qdrant Configuration (Cloud Migration)
    QDRANT_MODE: str = os.getenv("QDRANT_MODE", "cloud")  # 'cloud' or 'local'
    QDRANT_CLOUD_URL: str = os.getenv("QDRANT_CLOUD_URL", "")
    QDRANT_CLOUD_API_KEY: str = os.getenv("QDRANT_CLOUD_API_KEY", "")
    
    # Legacy settings (for backwards compatibility)
    QDRANT_URL: str = os.getenv("QDRANT_URL", "")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    
    # Qdrant Timeout Configuration
    QDRANT_TIMEOUT_SECONDS: int = int(os.getenv("QDRANT_TIMEOUT_SECONDS", "120"))  # 2 minutes
    QDRANT_RETRY_ATTEMPTS: int = int(os.getenv("QDRANT_RETRY_ATTEMPTS", "3"))  # Number of retries
    QDRANT_RETRY_DELAY_SECONDS: float = float(os.getenv("QDRANT_RETRY_DELAY_SECONDS", "2"))  # Initial delay
    
    # Embedding Configuration (Dual Models)
    EMBEDDING_MODEL_FINANCIAL: str = os.getenv("EMBEDDING_MODEL_FINANCIAL", "fin-e5-small")
    EMBEDDING_MODEL_GENERAL: str = os.getenv("EMBEDDING_MODEL_GENERAL", "sentence-transformers/all-MiniLM-L6-v2")
    CHUNK_SIZE_TOKENS: int = int(os.getenv("CHUNK_SIZE_TOKENS", "512"))
    CHUNK_OVERLAP_TOKENS: int = int(os.getenv("CHUNK_OVERLAP_TOKENS", "50"))
    CHUNK_EMBEDDING_STRATEGY: str = os.getenv("CHUNK_EMBEDDING_STRATEGY", "single-dense")
    
    # RAG Configuration
    RAG_PRIORITY_MODE: str = os.getenv("RAG_PRIORITY_MODE", "personal-first")
    RAG_SIMILARITY_THRESHOLD: float = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.1"))
    RAG_TOP_K_RESULTS: int = int(os.getenv("RAG_TOP_K_RESULTS", "20"))
    
    # Summarization Configuration
    SUMMARIZE_MODE: str = os.getenv("SUMMARIZE_MODE", "on-demand")
    
    # Tools Configuration
    ENABLE_TOOLS: bool = os.getenv("ENABLE_TOOLS", "True").lower() == "true"
    ENABLE_RAG: bool = os.getenv("ENABLE_RAG", "True").lower() == "true"
    ENABLE_SUMMARIZATION: bool = os.getenv("ENABLE_SUMMARIZATION", "True").lower() == "true"
    ENABLE_QUERY_REWRITING: bool = os.getenv("ENABLE_QUERY_REWRITING", "True").lower() == "true"
    
    # Workflow Version Configuration (Phase 4)
    WORKFLOW_VERSION: str = os.getenv("WORKFLOW_VERSION", "v4")  # v2, v3, or v4
    CANARY_ROLLOUT_PERCENTAGE: int = int(os.getenv("CANARY_ROLLOUT_PERCENTAGE", "100"))  # 0-100
    WORKFLOW_OBSERVER_ENABLED: bool = os.getenv("WORKFLOW_OBSERVER_ENABLED", "True").lower() == "true"
    
    RAG_MIN_RELEVANCE: float = float(os.getenv("RAG_MIN_RELEVANCE", "0.3"))
    RAG_MAX_DOCUMENTS: int = int(os.getenv("RAG_MAX_DOCUMENTS", "5"))
    SUMMARIZATION_THRESHOLD: int = int(os.getenv("SUMMARIZATION_THRESHOLD", "500"))
    QUERY_CONTEXT_DEPTH: int = int(os.getenv("QUERY_CONTEXT_DEPTH", "2"))
    
    ENABLED_TOOLS: str = os.getenv("ENABLED_TOOLS", "vnstock_tools,technical_tools")
    CORS_ORIGINS_STR: str = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000,http://localhost:8000")
    RATE_LIMIT_REQUESTS: int = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
    RATE_LIMIT_PERIOD_MINUTES: int = int(os.getenv("RATE_LIMIT_PERIOD_MINUTES", "60"))
    
    class Config:
        env_file = ".env"
        extra = "ignore"
    
    @property
    def cors_origins_list(self) -> list:
        return [origin.strip() for origin in self.CORS_ORIGINS_STR.split(",")]
    
    def should_use_workflow_version(self, user_id: str = None) -> str:
        """Determine which workflow version to use based on canary rollout.
        
        Args:
            user_id: Optional user ID for consistent canary assignment
            
        Returns:
            Workflow version string: 'v2', 'v3', or 'v4'
        """
        if self.CANARY_ROLLOUT_PERCENTAGE >= 100:
            return self.WORKFLOW_VERSION
        if self.CANARY_ROLLOUT_PERCENTAGE <= 0:
            return "v3"  # Fallback to v3
        
        # Use hash of user_id for consistent assignment
        if user_id:
            hash_val = hash(user_id) % 100
            if hash_val < self.CANARY_ROLLOUT_PERCENTAGE:
                return self.WORKFLOW_VERSION
            return "v3"
        
        # Random assignment if no user_id
        import random
        return self.WORKFLOW_VERSION if random.random() * 100 < self.CANARY_ROLLOUT_PERCENTAGE else "v3"

settings = Settings()

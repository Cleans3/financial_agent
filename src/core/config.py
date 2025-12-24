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
    
    # Embedding Configuration (Dual Models)
    EMBEDDING_MODEL_FINANCIAL: str = os.getenv("EMBEDDING_MODEL_FINANCIAL", "fin-e5-small")
    EMBEDDING_MODEL_GENERAL: str = os.getenv("EMBEDDING_MODEL_GENERAL", "sentence-transformers/all-MiniLM-L6-v2")
    CHUNK_SIZE_TOKENS: int = int(os.getenv("CHUNK_SIZE_TOKENS", "512"))
    CHUNK_OVERLAP_TOKENS: int = int(os.getenv("CHUNK_OVERLAP_TOKENS", "50"))
    CHUNK_EMBEDDING_STRATEGY: str = os.getenv("CHUNK_EMBEDDING_STRATEGY", "single-dense")
    
    # RAG Configuration
    RAG_PRIORITY_MODE: str = os.getenv("RAG_PRIORITY_MODE", "personal-first")
    RAG_SIMILARITY_THRESHOLD: float = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.25"))
    RAG_TOP_K_RESULTS: int = int(os.getenv("RAG_TOP_K_RESULTS", "5"))
    
    # Summarization Configuration
    SUMMARIZE_MODE: str = os.getenv("SUMMARIZE_MODE", "on-demand")
    
    # Tools Configuration
    ENABLE_TOOLS: bool = os.getenv("ENABLE_TOOLS", "True").lower() == "true"
    ENABLE_RAG: bool = os.getenv("ENABLE_RAG", "True").lower() == "true"
    ENABLE_SUMMARIZATION: bool = os.getenv("ENABLE_SUMMARIZATION", "True").lower() == "true"
    ENABLE_QUERY_REWRITING: bool = os.getenv("ENABLE_QUERY_REWRITING", "True").lower() == "true"
    
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

settings = Settings()

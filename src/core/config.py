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
    
    CORS_ORIGINS: list = ["http://localhost:5173", "http://localhost:3000", "http://localhost:8000"]
    
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_PERIOD_MINUTES: int = 60
    
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
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

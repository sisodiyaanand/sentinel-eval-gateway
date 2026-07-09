import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Sentinel LLM Red-Teaming & Eval Gateway"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    TOXICITY_THRESHOLD: float = 0.3
    SIMILARITY_THRESHOLD: float = 0.6
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    class Config:
        env_file = ".env"

settings = Settings()
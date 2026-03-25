from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # App
    APP_NAME: str = "Machining Optimizer"
    DEBUG: bool = True

    # AI Model Settings - plug in your Groq API key here
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY", "")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY", "")

    # Default model to use (can be overridden per request)
    DEFAULT_MODEL: str = "sklearn_baseline"  # Options: sklearn_baseline, groq_llm, custom_trained

    # Paths for custom trained models
    CUSTOM_MODEL_DIR: str = "./ml_models/saved_models"

    class Config:
        env_file = ".env"

settings = Settings()
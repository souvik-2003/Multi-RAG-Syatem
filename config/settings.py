import os
from pydantic_settings import BaseSettings, SettingsConfigDict
import streamlit as st

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env')

    # OpenRouter API Configuration
    OPENROUTER_API_KEY: str = st.secrets["OPENROUTER_API_KEY"]
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    
    # Model Configuration
    PRIMARY_MODEL: str = "anthropic/claude-3-haiku"
    VERIFICATION_MODEL: str = "openai/gpt-4-turbo-preview"
    IMAGE_ANALYSIS_MODEL: str = "anthropic/claude-3-sonnet"
    
    # Vector Database Configuration
    VECTOR_DB_PATH: str = "./data/vector_db"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    
    # Agent Configuration
    MAX_RETRIES: int = 3
    CONFIDENCE_THRESHOLD: float = 0.7
    HALLUCINATION_THRESHOLD: float = 0.6
    
    # Streamlit Configuration
    UPLOAD_DIR: str = "./data/uploads"
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB

settings = Settings()
